"""Slack estimators from per-endpoint signoff-timer slack tables (paper Section III-A, Eq. (2)).

Notation
    B, L        reference layout and optimized layout of one iteration
    N_B, N_L    their netlists
    G(X)        field-solver parasitics of layout X      (G(B) is computed once per reference layout)
    S(X)        extractor parasitics of layout X         (S(B) once, S(L) per iteration)
    s(N, P)     endpoint slack of the signoff timer on netlist N annotated with parasitics P

Every function takes arrays that are aligned on the same endpoint order (see ``align``).  The timer runs
themselves are outside this package: any signoff timer that writes ``endpoint<TAB>slack`` lines can feed it.

The 2 x 2 family (is the reference back-annotated on N_L?  is an extracted difference added?):

    estimator                  reference term      extracted difference
    CaPDE (Eq. (2))            s(N_L, G(B))        s(N_L, S(L)) - s(N_L, S(B))
    back-annotation alone      s(N_L, G(B))        none
    carried offset             s(N_B, G(B))        s(N_L, S(L)) - s(N_B, S(B))
    learned extraction alone   none                s(N_L, S(L))
"""
import numpy as np

ESTIMATORS = ("capde", "back_annotation_alone", "carried_offset", "learned_extraction_alone")


def read_slack_table(path):
    """Read ``endpoint<TAB>slack`` lines into a dict; lines whose second field is not a number are skipped."""
    out = {}
    with open(path) as fh:
        for line in fh:
            tok = line.rstrip("\n").split("\t")
            if len(tok) >= 2:
                try:
                    out[tok[0]] = float(tok[1])
                except ValueError:
                    pass
    return out


def align(*tables):
    """Endpoints present in every table (sorted) and one array per table in that order."""
    endpoints = sorted(set.intersection(*[set(t) for t in tables]))
    if not endpoints:
        raise ValueError("the slack tables share no endpoint")
    return endpoints, [np.array([t[e] for e in endpoints], dtype=float) for t in tables]


def layout_delta(s_nl_sl, s_nl_sb):
    """Bracket of Eq. (2): how far the extracted parasitics of L move the slack from those of B on the same netlist N_L."""
    return np.asarray(s_nl_sl, float) - np.asarray(s_nl_sb, float)


def capde_estimate(s_nl_gb, s_nl_sl, s_nl_sb):
    """Eq. (2): x = s(N_L, G(B)) + [s(N_L, S(L)) - s(N_L, S(B))]."""
    return np.asarray(s_nl_gb, float) + layout_delta(s_nl_sl, s_nl_sb)


def back_annotation_alone(s_nl_gb):
    """x_rt = s(N_L, G(B)): keeps the back-annotated reference and ignores the wire part."""
    return np.asarray(s_nl_gb, float).copy()


def carried_offset(s_nb_gb, s_nl_sl, s_nb_sb):
    """x_off = s(N_B, G(B)) + [s(N_L, S(L)) - s(N_B, S(B))]: the reference is timed on N_B, not on N_L."""
    return np.asarray(s_nb_gb, float) + (np.asarray(s_nl_sl, float) - np.asarray(s_nb_sb, float))


def learned_extraction_alone(s_nl_sl):
    """x_alone = s(N_L, S(L)): no reference term."""
    return np.asarray(s_nl_sl, float).copy()


def estimator_family(s_nl_gb, s_nl_sl, s_nl_sb, s_nb_gb, s_nb_sb):
    """All four estimators as a dict name -> array."""
    return {"capde": capde_estimate(s_nl_gb, s_nl_sl, s_nl_sb),
            "back_annotation_alone": back_annotation_alone(s_nl_gb),
            "carried_offset": carried_offset(s_nb_gb, s_nl_sl, s_nb_sb),
            "learned_extraction_alone": learned_extraction_alone(s_nl_sl)}


def reference_slack(g, g_repeat=None):
    """Field-solver reference slack s* (Eq. (1)): the more pessimistic of two field-solver runs on the optimized layout.

    g = s(N_L, G(L)); g_repeat = s(N_L, G'(L)) from an independent field-solver realization.  With one run, s* = g."""
    g = np.asarray(g, float)
    return g.copy() if g_repeat is None else np.minimum(g, np.asarray(g_repeat, float))


def cell_and_wire_part(s_nb_gb, s_nl_gb, g):
    """Split of the slack shift g - s(N_B, G(B)) of a sizing iteration: cell part (exact in the back-annotation) and wire part."""
    s_nb_gb, s_nl_gb, g = (np.asarray(a, float) for a in (s_nb_gb, s_nl_gb, g))
    return s_nl_gb - s_nb_gb, g - s_nl_gb


def error_stats(x, sstar):
    """Bias, variance, MSE and absolute-error percentiles of an estimate against the reference slack."""
    e = np.asarray(x, float) - np.asarray(sstar, float)
    a = np.abs(e)
    return dict(n=int(len(e)), bias=float(e.mean()), var=float(e.var()), mse=float((e ** 2).mean()), rms=float(np.sqrt((e ** 2).mean())),
                p50_abs=float(np.percentile(a, 50)), p95_abs=float(np.percentile(a, 95)), p99_abs=float(np.percentile(a, 99)))


def error_correlation(s_nl_sl, s_nl_sb, g_l, g_b_on_nl):
    """rho of Eq. (4): correlation of the extractor slack errors eps_L = s(N_L, S(L)) - s(N_L, G(L)) and
    eps_B = s(N_L, S(B)) - s(N_L, G(B)).  CaPDE has the lower MSE exactly when 2 E[eps_L eps_B] > E[eps_B^2]."""
    eps_l = np.asarray(s_nl_sl, float) - np.asarray(g_l, float)
    eps_b = np.asarray(s_nl_sb, float) - np.asarray(g_b_on_nl, float)
    rho = float(np.corrcoef(eps_l, eps_b)[0, 1]) if eps_l.std() > 0 and eps_b.std() > 0 else float("nan")
    return dict(rho=rho, mse_alone=float((eps_l ** 2).mean()), mse_capde=float(((eps_l - eps_b) ** 2).mean()),
                condition_holds=bool(2 * float((eps_l * eps_b).mean()) > float((eps_b ** 2).mean())))
