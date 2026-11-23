"""SPEF writer, and a SPEF reader for the capacitance and resistance sections (e.g. field-solver labels).

``write_spef`` streams one *D_NET block per net: the RC network of ``rcnet.RCNetwork`` and the capacitances of an
extraction result (``correction.apply`` format: net -> dict(ground=[[node, c]], coupling=[[node, other net, c]])).
A coupling row names the first node of the other net, so the other net's block can carry the mirrored row.
"""
import datetime

MIN_CAP = 1e-5     # fF; smaller rows are not written


def write_header(fh, design, ports, program="BasisCap"):
    """ports: iterable of (name, direction)."""
    now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    fh.write("\n".join(['*SPEF "IEEE 1481-1999"', f'*DESIGN "{design}"', f'*DATE "{now}"', '*VENDOR "none"', f'*PROGRAM "{program}"',
                        '*VERSION "1.0"', '*DESIGN_FLOW "PIN_CAP NONE" "NAME_SCOPE LOCAL"', "*DIVIDER /", "*DELIMITER :", "*BUS_DELIMITER []",
                        "*T_UNIT 1.0 NS", "*C_UNIT 1.0 FF", "*R_UNIT 1.0 OHM", "*L_UNIT 1.0 HENRY", "", "*PORTS"]) + "\n")
    for name, direction in sorted(ports):
        fh.write(f"{name} {direction}\n")
    fh.write("\n")


def write_net(fh, network, caps):
    """network: RCNetwork; caps: dict(ground, coupling) of this net."""
    gnd = [(n, c) for n, c in caps["ground"] if c > MIN_CAP]
    cpl = [(n, a, c) for n, a, c in caps["coupling"] if c > MIN_CAP]
    total = sum(c for _, c in gnd) + sum(c for _, _, c in cpl)
    fh.write(f"*D_NET {network.net} {total:.6f}\n\n*CONN\n")
    for kind in ("*P", "*I"):
        for name, p in network.pins.items():
            if p["kind"] == kind:
                cell = f" *D {p['cell']}" if p.get("cell") else ""
                fh.write(f"{kind} {name} {p['direction']} *C {p['xy'][0]:.4f} {p['xy'][1]:.4f}{cell}\n")
    for name, (x, y, _) in network.nodes.items():
        fh.write(f"*N {name} *C {x:.4f} {y:.4f}\n")
    fh.write("\n*CAP\n")
    k = 0
    for n, c in gnd:
        k += 1
        fh.write(f"{k} {n} {c:.6f}\n")
    for n, a, c in cpl:
        k += 1
        fh.write(f"{k} {n} {a}:1 {c:.6f}\n")
    fh.write("\n*RES\n")
    for i, (a, b, r) in enumerate(network.resistors, 1):
        fh.write(f"{i} {a} {b} {r:.6f}\n")
    fh.write("*END\n\n")


def write_spef(path, design, ports, networks, caps):
    """networks: {net: RCNetwork}; caps: extraction result for the same nets."""
    with open(path, "w") as fh:
        write_header(fh, design, ports)
        for net, network in networks.items():
            write_net(fh, network, caps.get(net, dict(ground=[], coupling=[])))


def read_spef(path):
    """-> {net: dict(total, ground=[[node, c]], coupling=[[node, other net, c]], resistors=[(a, b, r)])}.

    The other net of a coupling row is the owner of the row's second node: the net whose *CONN section declares it,
    else the name before the last delimiter."""
    nets, owner, pending = {}, {}, []
    cur = sec = None
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.split("//", 1)[0].split()
            if not s:
                continue
            if s[0].startswith("*"):
                if s[0] == "*D_NET":
                    cur = s[1]
                    nets[cur] = dict(total=float(s[2]), ground=[], coupling=[], resistors=[])
                    sec = None
                elif s[0] in ("*CONN", "*CAP", "*RES"):
                    sec = s[0]
                elif s[0] == "*END":
                    cur = sec = None
                elif sec == "*CONN" and s[0] in ("*P", "*I", "*N") and cur is not None:
                    owner[s[1]] = cur
                continue
            if cur is None:
                continue
            if sec == "*CAP" and len(s) == 3:
                nets[cur]["ground"].append([s[1], float(s[2])])
            elif sec == "*CAP" and len(s) == 4:
                pending.append((cur, s[1], s[2], float(s[3])))
            elif sec == "*RES" and len(s) >= 4:
                nets[cur]["resistors"].append((s[1], s[2], float(s[3])))
    for net, n1, n2, c in pending:
        def own(node):
            return owner.get(node) or (node.rsplit(":", 1)[0] if ":" in node else node)
        a1, a2 = own(n1), own(n2)
        other, node = (a2, n1) if a1 == net else (a1, n2)
        nets[net]["coupling"].append([node, other, c])
    return nets
