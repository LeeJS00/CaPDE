"""Geometry features of the final correction: metal topology only (no extractor output, no label).

For every window (tile) of a net, the rows the tile owns (target rows of the net) are the targets; every other row is a
neighbour: another signal net, a power wire, or a piece of the same net that the tile does not own.

Per-tile features (summed over owned rows)
    A_<cls>, S_<cls>      plan area w*h and side area 2(w+h)d per class (routing layers, via, pin)
    LAT_<type>_b<i>       lateral plate proxy eps * d_overlap * L_overlap / gap to neighbours of type sig / pwr / own, per gap bin
    VUP_/VDN_<type>       vertical plate proxy eps * A_overlap / gap to neighbours above / below within the vertical window
    COVUP, COVDN          owned plan area covered by any conductor above / below within the vertical window
    NNB_sig, NNB_pwr      number of distinct neighbour rows within the near gap
    n_own, n_own_pin      owned rows, owned pin rows
    X_...                 the lateral / vertical proxies, the nearest-neighbour proxy and the free side length per class
Per (tile, aggressor net) pair proxies
    P_LAT_b0..2, P_VUP, P_VDN, P_mingap, P_rows
"""
import numpy as np

from .. import windows as Wn

TYPES = ("sig", "pwr", "own")
PAIR_NAMES = ["P_LAT_b0", "P_LAT_b1", "P_LAT_b2", "P_VUP", "P_VDN", "P_mingap", "P_rows"]
PAIR_TILE_NAMES = PAIR_NAMES + ["T_LATs_b0", "T_LATs_b1", "T_LATs_b2", "T_VUPs", "T_VDNs", "T_n_own", "T_A", "T_S"]
GAP_FLOOR = 1e-3


def classes(tech):
    return [L["name"] for L in tech.routing] + ["via", "pin"]


def tile_names(tech):
    cls = classes(tech)
    base = ([f"A_{c}" for c in cls] + [f"S_{c}" for c in cls] + [f"LAT_{t}_b{i}" for t in TYPES for i in range(3)]
            + ["VUP_sig", "VUP_pwr", "VDN_sig", "VDN_pwr", "COVUP", "COVDN", "NNB_sig", "NNB_pwr", "n_own", "n_own_pin"])
    extra = ([f"X_LAT_{t}_{b}_{c}" for t in ("sig", "pwr") for b in ("b0", "b1") for c in cls]
             + [f"X_{v}_{t}_{c}" for v in ("VUP", "VDN") for t in ("sig", "pwr") for c in cls]
             + [f"X_NN1_{c}" for c in cls] + [f"X_FREE_{c}" for c in cls])
    return base + extra


def tile_features(tile, net, tech, corr):
    """tile: dict(window (n, 10), abs_boxes (n, 6), net_names [n]).  corr: namespace with gap_bins, vertical_window, near_gap.

    Returns (vector over ``tile_names``, {aggressor net: 7 pair proxies}, bbox (x0, x1, y0, y1, zmax)) or None when the
    tile owns no row of the net."""
    c = np.asarray(tile["window"], dtype=np.float64)
    g = np.asarray(tile["abs_boxes"], dtype=np.float64)
    names = np.asarray(tile["net_names"])
    own = (c[:, Wn.ROLE] > 0.5) & (names == net)
    if not own.any():
        return None
    cls_names = classes(tech)
    nc = len(cls_names)
    idx = {n: i for i, n in enumerate(tile_names(tech))}
    out = np.zeros(len(idx))
    bins = list(zip(corr.gap_bins[:-1], corr.gap_bins[1:]))
    x, y, z, w, h, dz, eps = g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4], g[:, 5], c[:, Wn.EPS]
    T = np.where(own)[0]
    routing_index = {L["name"]: i for i, L in enumerate(tech.routing)}
    cls = []
    for t in T:
        if c[t, Wn.KIND] < 0.8:
            cls.append(nc - 1)                                                   # pin
        else:
            L = tech.layer_of_z(z[t])
            cls.append(routing_index.get(L["name"], nc - 2))                     # routing layer, else via
    cls = np.array(cls)
    A = w[T] * h[T]
    S = 2 * (w[T] + h[T]) * dz[T]
    np.add.at(out, cls, A)
    np.add.at(out, nc + cls, S)
    typ = np.where(c[:, Wn.NET_TYPE] > 0.6, 1, np.where(names == net, 2, 0))     # 0 other signal, 1 power, 2 same net
    nb = ~own

    # target x neighbour geometry
    dx = np.abs(x[T, None] - x[None, :]) - (w[T, None] + w[None, :]) / 2
    dy = np.abs(y[T, None] - y[None, :]) - (h[T, None] + h[None, :]) / 2
    zov = (np.minimum(z[T, None] + dz[T, None] / 2, z[None, :] + dz[None, :] / 2)
           - np.maximum(z[T, None] - dz[T, None] / 2, z[None, :] - dz[None, :] / 2))
    ox = np.clip(-dx, 0, np.minimum(w[T, None], w[None, :]))
    oy = np.clip(-dy, 0, np.minimum(h[T, None], h[None, :]))
    lat = (zov > 0) & nb[None, :]
    fx = lat & (dy < 0) & (dx > 0)                                               # facing along x
    fy = lat & (dx < 0) & (dy > 0)                                               # facing along y
    gap = np.where(fx, dx, np.where(fy, dy, np.inf))
    length = np.where(fx, oy, np.where(fy, ox, 0.0))
    val = np.where(np.isfinite(gap), eps[T, None] * np.clip(zov, 0, None) * length / np.maximum(gap, GAP_FLOOR), 0.0)
    for ti, tname in enumerate(TYPES):
        mt = typ[None, :] == ti
        for bi, (lo, hi) in enumerate(bins[:3]):
            sel = mt & (gap >= lo) & (gap < hi)
            out[idx[f"LAT_{tname}_b{bi}"]] = val[sel].sum()
            if ti < 2 and bi < 2:
                np.add.at(out, idx[f"X_LAT_{tname}_b{bi}_{cls_names[0]}"] + cls, np.where(sel, val, 0.0).sum(1))

    # nearest lateral neighbour of each owned row, and the row's free side length
    gn = np.where(lat & (fx | fy), gap, np.inf)
    jn = gn.argmin(1)
    gmin = gn[np.arange(len(T)), jn]
    has = np.isfinite(gmin)
    np.add.at(out, idx[f"X_NN1_{cls_names[0]}"] + cls, np.where(has, val[np.arange(len(T)), jn], 0.0))
    np.add.at(out, idx[f"X_FREE_{cls_names[0]}"] + cls, np.where(has & (gmin < corr.gap_bins[-1]), 0.0, np.maximum(w[T], h[T])))

    # per-aggressor-net proxies (other signal nets only)
    sig = (typ == 0) & nb
    pair = None
    if sig.any():
        un, inv = np.unique(names[sig], return_inverse=True)
        cols = np.where(sig)[0]
        pv = np.zeros((len(un), 7))
        pv[:, 5] = np.inf
        for bi, (lo, hi) in enumerate(bins[:3]):
            np.add.at(pv[:, bi], inv, np.where((gap[:, cols] >= lo) & (gap[:, cols] < hi), val[:, cols], 0.0).sum(0))
        np.minimum.at(pv[:, 5], inv, gap[:, cols].min(0))
        np.add.at(pv[:, 6], inv, 1.0)
        pair = (un, pv, cols, inv)
    near = gap < corr.near_gap
    out[idx["NNB_sig"]] = np.unique(np.where(near & (typ[None, :] == 0))[1]).size
    out[idx["NNB_pwr"]] = np.unique(np.where(near & (typ[None, :] == 1))[1]).size

    # vertical: plan overlap area, gap in z
    gz_up = (z[None, :] - dz[None, :] / 2) - (z[T, None] + dz[T, None] / 2)
    gz_dn = (z[T, None] - dz[T, None] / 2) - (z[None, :] + dz[None, :] / 2)
    a_ov = ox * oy * nb[None, :] * (typ[None, :] != 2)
    for k, (gz, vname) in enumerate(((gz_up, "VUP"), (gz_dn, "VDN"))):
        m = (gz > 1e-4) & (gz < corr.vertical_window) & (a_ov > 0)
        v = np.where(m, eps[T, None] * a_ov / np.maximum(gz, GAP_FLOOR), 0.0)
        out[idx[f"{vname}_sig"]] = v[:, typ == 0].sum()
        out[idx[f"{vname}_pwr"]] = v[:, typ == 1].sum()
        np.add.at(out, idx[f"X_{vname}_sig_{cls_names[0]}"] + cls, v[:, typ == 0].sum(1))
        np.add.at(out, idx[f"X_{vname}_pwr_{cls_names[0]}"] + cls, v[:, typ == 1].sum(1))
        out[idx["COVUP" if k == 0 else "COVDN"]] = np.minimum(np.where(m, a_ov, 0.0).sum(1), A).sum()
        if pair is not None:
            np.add.at(pair[1][:, 3 + k], pair[3], v[:, pair[2]].sum(0))
    out[idx["n_own"]] = T.size
    out[idx["n_own_pin"]] = int((cls == nc - 1).sum())
    bbox = (float((x[T] - w[T] / 2).min()), float((x[T] + w[T] / 2).max()), float((y[T] - h[T] / 2).min()),
            float((y[T] + h[T] / 2).max()), float(z[T].max()))
    pairs = {}
    if pair is not None:
        pairs = {str(n): r for n, r in zip(pair[0], pair[1]) if r[:5].sum() > 0}
    return out, pairs, bbox


def net_tables(tiles_by_net, tech, corr):
    """tiles_by_net: {net: [tile, ...]} -> tables for the correction models.

    Returns dict:
        nets                 sorted net names
        tile_X, tile_net     per-tile features (n_tiles, F) and the net index of each row   (tile-additive net total model)
        ptile_X, ptile_v, ptile_a   per-(victim, aggressor, tile) features (n_rows, 15), victim index, aggressor name
        pairs                {victim: {aggressor: 7 pair proxies summed over the victim's tiles (closest gap: minimum)}}
    """
    names = tile_names(tech)
    ix = {n: i for i, n in enumerate(names)}
    nc = len(classes(tech))
    nets = sorted(tiles_by_net)
    tile_X, tile_net, ptile_X, ptile_v, ptile_a, pairs = [], [], [], [], [], {}
    for vi, net in enumerate(nets):
        acc = {}
        for tile in tiles_by_net[net]:
            r = tile_features(tile, net, tech, corr)
            if r is None:
                continue
            o, pd_, _ = r
            tile_X.append(o.astype(np.float32))
            tile_net.append(vi)
            ctx = np.array([o[ix["LAT_sig_b0"]], o[ix["LAT_sig_b1"]], o[ix["LAT_sig_b2"]], o[ix["VUP_sig"]], o[ix["VDN_sig"]], o[ix["n_own"]],
                            o[:nc].sum(), o[nc:2 * nc].sum()], dtype=np.float32)       # tile context of the victim side
            for a, v in pd_.items():
                ptile_X.append(np.concatenate([v.astype(np.float32), ctx]))
                ptile_v.append(vi)
                ptile_a.append(a)
                q = acc.get(a)
                if q is None:
                    acc[a] = v.copy()
                else:
                    gmin = min(q[5], v[5])
                    q += v
                    q[5] = gmin
        pairs[net] = acc
    return dict(nets=nets, names=names,
                tile_X=np.array(tile_X, dtype=np.float32).reshape(-1, len(names)), tile_net=np.array(tile_net, dtype=np.int64),
                ptile_X=np.array(ptile_X, dtype=np.float32).reshape(-1, len(PAIR_TILE_NAMES)), ptile_v=np.array(ptile_v, dtype=np.int64),
                ptile_a=np.array(ptile_a, dtype=object), pairs=pairs)
