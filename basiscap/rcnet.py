"""RC network of one net from its routed shapes (paper Section III-D).

The net is split at junctions, vias, pins and a maximum length.  Segment resistance = sheet resistance x length /
electrical width (drawn width minus a width bias); a via contributes its per-cut resistance; one constant fit on training
designs multiplies every resistance of the network.  Ground and coupling capacitances
predicted for the net's cuboids sit at the nearest nodes, so the signoff timer sees where along the net a coupling acts.

Shapes of a net (micrometres):
    dict(type='wire', layer, start=(x, y), end=(x, y), width)                      axis-aligned centerline
    dict(type='via',  lower, upper, pos=(x, y), width)
    dict(type='pin',  layer, rect=(llx, lly, urx, ury), name, direction='I'|'O'|'B', port=False, cell=None)
        name = 'instance:pin' for a cell pin, the port name for a top-level port (port=True)
"""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

CONTACT_RES = 0.001     # ohm; joins pieces of one metal shape


def _merge_collinear(wires):
    """Wires that were cut at window borders are joined back: same layer, axis, coordinate and width, touching."""
    groups = defaultdict(list)
    for w in wires:
        (sx, sy), (ex, ey), width = w["start"], w["end"], float(w["width"])
        if abs(sy - ey) < 1e-4:
            groups[(w["layer"], 0, round(sy, 4), width)].append((min(sx, ex), max(sx, ex)))
        else:
            groups[(w["layer"], 1, round(sx, 4), width)].append((min(sy, ey), max(sy, ey)))
    out = []
    for (layer, axis, const, width), segs in groups.items():
        segs.sort()
        merged = []
        for s in segs:
            if merged and s[0] <= merged[-1][1] + 1e-3:
                merged[-1] = (merged[-1][0], max(merged[-1][1], s[1]))
            else:
                merged.append(s)
        for a, b in merged:
            start, end = ((a, const), (b, const)) if axis == 0 else ((const, a), (const, b))
            out.append(dict(layer=layer, start=start, end=end, width=width))
    return out


class RCNetwork:
    """nodes: {node name: (x, y, layer)}; resistors: [(node, node, ohm)]; pins: {name: dict(kind '*I'|'*P', direction, xy, cell)}."""

    def __init__(self, net, shapes, tech, rc):
        self.net, self.tech = net, tech
        self.dbu = int(rc.dbu)
        self.max_len = float(rc.max_segment_length)
        self.width_bias = float(rc.width_bias)
        self.r_scale = float(rc.resistance_scale)
        self.snap_radius = float(getattr(rc, "pin_snap_radius", 2.0))
        self.nodes, self.node_xy, self.resistors, self.pins = {}, {}, [], {}
        self._key = {}
        self._build(shapes)

    def _node(self, xy, layer):
        key = (int(round(xy[0] * self.dbu)), int(round(xy[1] * self.dbu)), layer)
        if key not in self._key:
            name = f"{self.net}:{len(self._key) + 1}"
            self._key[key] = name
            self.nodes[name] = (xy[0], xy[1], layer)
            self.node_xy[name] = (xy[0], xy[1])
        return self._key[key]

    def _nodes_in_rect(self, layer, rect, tol=1e-4):
        llx, lly, urx, ury = rect
        return [n for n, (x, y, l) in self.nodes.items() if l == layer and llx - tol <= x <= urx + tol and lly - tol <= y <= ury + tol]

    def _build(self, shapes):
        wires = [dict(s) for s in shapes if s["type"] == "wire"]
        pins = [s for s in shapes if s["type"] == "pin"]
        vias = [s for s in shapes if s["type"] == "via"]
        for p in pins:                                           # the pin metal conducts: add its centerline as a wire
            llx, lly, urx, ury = p["rect"]
            w = min(urx - llx, ury - lly)
            if urx - llx > ury - lly:
                s, e = (llx + w / 2, (lly + ury) / 2), (urx - w / 2, (lly + ury) / 2)
            else:
                s, e = ((llx + urx) / 2, lly + w / 2), ((llx + urx) / 2, ury - w / 2)
            wires.append(dict(layer=p["layer"], start=s, end=e, width=w))
        wires = _merge_collinear(wires)

        # cut points: pin centers, vias, wire ends (a wire end on another wire is a junction of that wire)
        cuts = defaultdict(set)
        for p in pins:
            cuts[p["layer"]].add((round((p["rect"][0] + p["rect"][2]) / 2, 4), round((p["rect"][1] + p["rect"][3]) / 2, 4)))
        for v in vias:
            for layer in (v["lower"], v["upper"]):
                cuts[layer].add((round(v["pos"][0], 4), round(v["pos"][1], 4)))
        for w in wires:
            for q in (w["start"], w["end"]):
                cuts[w["layer"]].add((round(q[0], 4), round(q[1], 4)))

        for w in wires:
            layer, width = w["layer"], float(w["width"])
            L = self.tech.by_name[layer.lower()]
            (sx, sy), (ex, ey) = (round(w["start"][0], 4), round(w["start"][1], 4)), (round(w["end"][0], 4), round(w["end"][1], 4))
            axis = 0 if abs(sy - ey) < 1e-4 else 1
            on = [c for c in cuts[layer]
                  if (axis == 0 and abs(c[1] - sy) < 1e-3 and min(sx, ex) - 1e-3 <= c[0] <= max(sx, ex) + 1e-3)
                  or (axis == 1 and abs(c[0] - sx) < 1e-3 and min(sy, ey) - 1e-3 <= c[1] <= max(sy, ey) + 1e-3)]
            on.sort(key=lambda c: c[axis], reverse=(sx > ex) if axis == 0 else (sy > ey))
            pts = []                                             # (x, y, is_virtual): virtual = split by the maximum length
            for i, c in enumerate(on):
                if i > 0:
                    prev = on[i - 1]
                    length = float(np.hypot(c[0] - prev[0], c[1] - prev[1]))
                    if length < 1e-4:
                        continue
                    if length > self.max_len + 1e-4:
                        ux, uy = (c[0] - prev[0]) / length, (c[1] - prev[1]) / length
                        for j in range(1, int(length // self.max_len) + 1):
                            if length - j * self.max_len >= 1e-4:
                                pts.append((round(prev[0] + ux * j * self.max_len, 4), round(prev[1] + uy * j * self.max_len, 4), True))
                pts.append((c[0], c[1], False))
            w_eff = max(width - self.width_bias, 0.1 * width)
            for p1, p2 in zip(pts[:-1], pts[1:]):
                n1, n2 = self._node(p1[:2], layer), self._node(p2[:2], layer)
                if n1 == n2:
                    continue
                length = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                cut = (0.0 if p1[2] else width / 2.0) + (0.0 if p2[2] else width / 2.0)      # half a width at each real end
                l_eff = max(length - cut, 0.1 * width) if length > cut else length
                self.resistors.append((n1, n2, float(L["sheet_res"]) * l_eff / w_eff))

        for v in vias:
            via_layer = self.tech.via_between(v["lower"], v["upper"])
            r = float(via_layer["via_res"]) if via_layer else CONTACT_RES
            self.resistors.append((self._node(v["pos"], v["lower"]), self._node(v["pos"], v["upper"]), r))

        # all nodes inside one pin shape are one piece of metal
        for p in pins:
            hits = self._nodes_in_rect(p["layer"], p["rect"])
            for n in hits[1:]:
                self.resistors.append((hits[0], n, CONTACT_RES))

        # logical pin nodes
        by_layer = defaultdict(list)
        for n, (x, y, l) in self.nodes.items():
            by_layer[l].append((n, x, y))
        trees = {l: (cKDTree(np.array([[x, y] for _, x, y in v])), [n for n, _, _ in v]) for l, v in by_layer.items()}
        for p in pins:
            llx, lly, urx, ury = p["rect"]
            cx, cy = (llx + urx) / 2.0, (lly + ury) / 2.0
            self.pins[p["name"]] = dict(kind="*P" if p.get("port") else "*I", direction=p.get("direction", "B"), xy=(cx, cy), cell=p.get("cell"))
            self.node_xy[p["name"]] = (cx, cy)
            hits = self._nodes_in_rect(p["layer"], p["rect"])
            if not hits:
                best = None
                for l in [p["layer"]] + [l for l in trees if l != p["layer"]]:
                    if l in trees:
                        dist, i = trees[l][0].query([cx, cy])
                        if dist < self.snap_radius and (best is None or dist < best[0]):
                            best = (dist, trees[l][1][i])
                    if best is not None and l == p["layer"]:
                        break
                hits = [best[1]] if best else []
            for n in hits:
                self.resistors.append((p["name"], n, CONTACT_RES))
        self._close_islands()
        self.resistors = [(a, b, self.r_scale * r) for a, b, r in self.resistors]     # the one fitted constant

    def components(self):
        """Connected components of the resistor graph as {root: [nodes]}."""
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b, _ in self.resistors:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        groups = defaultdict(list)
        for n in list(self.nodes) + list(self.pins):
            groups[find(n)].append(n)
        return groups

    def _close_islands(self):
        """Every pin of the net must sit in one connected component: join the nearest nodes of split components."""
        while len(self.pins) >= 2:
            groups = self.components()
            root_of = {n: r for r, ns in groups.items() for n in ns}
            roots = {root_of[p] for p in self.pins}
            if len(roots) <= 1:
                return
            main = max(roots, key=lambda r: len(groups[r]))
            best = None
            for r in roots - {main}:
                for a in groups[r]:
                    for b in groups[main]:
                        d = (self.node_xy[a][0] - self.node_xy[b][0]) ** 2 + (self.node_xy[a][1] - self.node_xy[b][1]) ** 2
                        if best is None or d < best[0]:
                            best = (d, a, b)
            if best is None:
                return
            self.resistors.append((best[1], best[2], CONTACT_RES))

    # ---- capacitance placement ----------------------------------------------------------------------------------
    def place(self, cuboid_caps):
        """cuboid_caps: [dict(xyz=(x, y, z), gnd=c, cpl={aggressor net: c})] per target cuboid of this net (absolute
        coordinates).  Each entry goes to the nearest node of the cuboid's routing layer (any layer if that layer has
        no node).  Returns dict(ground=[[node, c]], coupling=[[node, aggressor net, c]])."""
        by_layer = defaultdict(list)
        for n, (x, y, l) in self.nodes.items():
            by_layer[l].append((n, x, y))
        trees = {l: (cKDTree(np.array([[x, y] for _, x, y in v])), [n for n, _, _ in v]) for l, v in by_layer.items()}
        every = (cKDTree(np.array([[x, y] for (x, y, _) in self.nodes.values()])), list(self.nodes)) if self.nodes else None
        gnd, cpl = defaultdict(float), defaultdict(float)
        for c in cuboid_caps:
            x, y, z = c["xyz"]
            layer = self.tech.layer_of_z(z, kind="routing")["name"]
            tree, names = trees.get(layer) or trees.get(layer.lower()) or every or (None, None)
            if tree is None:
                continue
            node = names[tree.query([x, y])[1]]
            gnd[node] += float(c["gnd"])
            for a, v in c["cpl"].items():
                cpl[(node, a)] += float(v)
        return dict(ground=[[n, v] for n, v in gnd.items()], coupling=[[n, a, v] for (n, a), v in cpl.items()])


def cuboid_caps(pred, batch_index, abs_boxes, net_names, own=None):
    """Model output of one window -> entries for ``RCNetwork.place``.

    pred: output of ``BasisCap.forward``; batch_index: row of the window in the batch; abs_boxes (n, 6) absolute
    cuboids; net_names [n]; own (n,) ownership share of each target cuboid (default 1)."""
    c_gnd = pred["c_gnd"][batch_index].detach().float().cpu().numpy()
    e = pred["cpl"]
    sel = (e["b_idx"] == batch_index).cpu().numpy()
    src, dst, val = e["src_idx"].cpu().numpy()[sel], e["dst_idx"].cpu().numpy()[sel], e["c_cpl"].detach().float().cpu().numpy()[sel]
    out = {}
    for i in np.nonzero(c_gnd > 0)[0]:
        share = 1.0 if own is None else float(own[i])
        if share > 0:
            out[int(i)] = dict(xyz=tuple(float(v) for v in abs_boxes[i][:3]), gnd=float(c_gnd[i]) * share, cpl=defaultdict(float))
    for s, d, v in zip(src, dst, val):
        if int(s) in out:
            share = 1.0 if own is None else float(own[s])
            out[int(s)]["cpl"][str(net_names[d])] += float(v) * share
    return list(out.values())
