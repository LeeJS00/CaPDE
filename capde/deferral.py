"""Deferral policy (paper Section III-C).

A deferred endpoint has a fallback.  By default it fails.  Alternatively, one partial run field-solves a share beta of
the candidate nets (the coupled victims on the worst paths of the deferred endpoints) and extracts the rest with the
fast extractor; Eq. (5) then re-decides the deferred endpoints on the slack of that run with a half-width b_beta fit
per share beta (``capde.interval.redecide``).

Inputs the user provides
    paths     for every deferred endpoint, (its slack, its worst timing paths as lists of net names, worst path first),
              from a timing-path report of the reference back-annotation run; no field-solver run on L is needed
    coupled   optional set of net names that carry coupling capacitance in the extracted parasitics of L
"""
import numpy as np

from .interval import labels


def default_fail(decision):
    """Default policy: every deferral becomes a fail."""
    decision = np.asarray(decision, dtype=object).copy()
    decision[decision == "defer"] = "fail"
    return decision


def victims_per_endpoint(paths, coupled=None, n_paths=10):
    """{endpoint: (worst slack, [paths as net lists])} -> {endpoint: (worst slack, [candidate nets in path order, unique])}.

    Only the first ``n_paths`` paths of an endpoint are used; a net not in ``coupled`` (when given) is skipped."""
    out = {}
    for e, (s0, plist) in paths.items():
        seq = []
        for nets in plist[:n_paths]:
            for n in nets:
                if (coupled is None or n in coupled) and n not in seq:
                    seq.append(n)
        out[e] = (float(s0), seq)
    return out


def round_robin(seq, worst_slack, n):
    """Order the candidate pool: endpoints by ascending worst slack (ties by name); in rounds, each endpoint contributes
    its next net that is not yet taken.  Returns the first n nets of that order."""
    order = sorted(seq, key=lambda e: (worst_slack.get(e, float("inf")), e))
    ptr = {e: 0 for e in order}
    take, got = [], set()
    while len(take) < n:
        moved = False
        for e in order:
            q = seq[e]
            while ptr[e] < len(q) and q[ptr[e]] in got:
                ptr[e] += 1
            if ptr[e] < len(q) and len(take) < n:
                take.append(q[ptr[e]])
                got.add(q[ptr[e]])
                ptr[e] += 1
                moved = True
        if not moved:
            break
    return take


def candidate_nets(deferred, paths, beta, coupled=None, n_paths=10):
    """Nets to field-solve for a share beta in (0, 1] of the candidate nets.

    Returns (selected nets, pool size N); the selection holds floor(N * beta) nets of the round-robin order."""
    per = victims_per_endpoint({e: paths[e] for e in deferred}, coupled, n_paths)
    seq = {e: v[1] for e, v in per.items()}
    s0 = {e: v[0] for e, v in per.items()}
    pool = sorted({n for q in seq.values() for n in q})
    order = round_robin(seq, s0, len(pool))
    return order[: int(np.floor(len(pool) * beta))], len(pool)


def deferred_endpoints(endpoints, x, b, theta=0.0):
    """Endpoints that Eq. (5) defers."""
    lab = labels(x, b, theta)
    return [e for e, d in zip(endpoints, lab) if d == "defer"]


# Generic template of the partial re-extraction run.  Every {placeholder} is supplied by the user: the command of an
# extractor that field-solves a listed subset of nets and extracts the rest with a fast extractor, and a signoff-timer
# command that writes 'endpoint<TAB>slack'.  No technology file is part of this repository.
REEXTRACT_TEMPLATE = """\
# 1. partial field-solver re-extraction of layout L
{extractor_cmd} \\
    --layout {layout_def} \\
    --technology {technology_file} \\
    --field-solve-nets {net_list_file} \\
    --other-nets fast \\
    --output {partial_spef}

# 2. signoff timing of N_L on the re-extracted parasitics (crosstalk on)
{timer_cmd} \\
    --netlist {netlist_L} \\
    --constraints {constraints} \\
    --parasitics {partial_spef} \\
    --report-endpoint-slack {slack_table}
"""


def reextract_script(net_list_file, **fields):
    """Fill the template; missing fields stay as visible {placeholders}."""
    class _Keep(dict):
        def __missing__(self, k):
            return "{" + k + "}"
    return REEXTRACT_TEMPLATE.format_map(_Keep(net_list_file=net_list_file, **fields))
