"""Applying the final correction to an extraction result, and the post-processing of couplings.

An extraction result is a dict  net -> dict(ground=[[node, c], ...], coupling=[[node, other_net, c], ...])  (fF); the RC
network (nodes, resistors) is not touched by any function here.

    correct()             pair rescaling, then ground = net total - couplings
    coupling_threshold()  couplings below a threshold move to ground at their node (net totals unchanged)
    symmetrize()          every coupling gets one value in both nets' blocks
"""
from collections import defaultdict

R_MIN, R_MAX = 0.2, 5.0        # clip of a pair's rescaling factor
G_MIN, G_MAX = 0.25, 4.0       # clip of a net's ground rescaling factor


def pair_sums(caps):
    """{v: {a: sum of v's coupling rows towards net a}} (directed)."""
    out = {}
    for v, blk in caps.items():
        d = defaultdict(float)
        for _, a, c in blk["coupling"]:
            if a != v:
                d[a] += c
        out[v] = dict(d)
    return out


def net_totals(caps):
    return {v: sum(c for _, c in blk["ground"]) + sum(c for _, _, c in blk["coupling"]) for v, blk in caps.items()}


def correct(caps, net_total, pair_coupling):
    """net_total {net: T}; pair_coupling {(a, b) sorted: c}.  Returns (corrected caps, stats).

    Pair step: one factor per unordered pair (mirrored rows stay mirrors) = target / extracted sum, taken from the
    block of the net whose name sorts first, clipped to [R_MIN, R_MAX]; a pair without a target keeps its value.
    Ground step: every ground row of a net is multiplied by (T - coupling sum) / ground sum, clipped to [G_MIN, G_MAX]."""
    ps = pair_sums(caps)
    ratio, n_clip = {}, 0
    for v, d in ps.items():
        for a, x in d.items():
            if v < a and x > 0 and (v, a) in pair_coupling:
                r = pair_coupling[(v, a)] / x
                if not (R_MIN <= r <= R_MAX):
                    n_clip += 1
                    r = min(max(r, R_MIN), R_MAX)
                ratio[(v, a)] = r
    out, g_clip = {}, 0
    for v, blk in caps.items():
        cpl = [[n, a, c * ratio.get((v, a) if v < a else (a, v), 1.0)] for n, a, c in blk["coupling"]]
        c_sum = sum(c for _, _, c in cpl)
        g_sum = sum(c for _, c in blk["ground"])
        gnd = [list(r) for r in blk["ground"]]
        if v in net_total and g_sum > 0:
            f = (net_total[v] - c_sum) / g_sum
            if not (G_MIN <= f <= G_MAX):
                g_clip += 1
                f = min(max(f, G_MIN), G_MAX)
            gnd = [[n, c * f] for n, c in gnd]
        out[v] = dict(ground=gnd, coupling=cpl)
    return out, dict(pairs_rescaled=len(ratio), pairs_clipped=n_clip, nets_ground_clipped=g_clip)


def coupling_threshold(caps, threshold):
    """Coupling rows below the threshold move to ground at their own node; ground rows of one node are merged."""
    out = {}
    for v, blk in caps.items():
        gnd, order, cpl = {}, [], []
        for n, c in blk["ground"]:
            if n not in gnd:
                gnd[n] = 0.0
                order.append(n)
            gnd[n] += c
        for n, a, c in blk["coupling"]:
            if c < threshold:
                if n not in gnd:
                    gnd[n] = 0.0
                    order.append(n)
                gnd[n] += c
            else:
                cpl.append([n, a, c])
        out[v] = dict(ground=[[n, gnd[n]] for n in order], coupling=cpl)
    return out


def symmetrize(caps):
    """If both nets' blocks carry the pair {v, a}, both groups of rows are rescaled to the mean of the two sums;
    a pair listed on one side only is left as is."""
    ps = pair_sums(caps)
    out = {}
    for v, blk in caps.items():
        cpl = []
        for n, a, c in blk["coupling"]:
            mine, theirs = ps[v].get(a, 0.0), ps.get(a, {}).get(v, 0.0)
            f = 0.5 * (mine + theirs) / mine if (mine > 0 and theirs > 0) else 1.0
            cpl.append([n, a, c * f])
        out[v] = dict(ground=[list(r) for r in blk["ground"]], coupling=cpl)
    return out
