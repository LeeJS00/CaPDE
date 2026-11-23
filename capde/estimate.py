"""Slack estimates from per-endpoint signoff-timer slack tables (paper Sections II and III-A, Eqs. (1)-(4)).

Notation
    B, L        reference layout and optimized layout of one closure iteration
    N_B, N_L    their netlists
    G(X)        field-solver parasitics of layout X      (G(B) is computed once per reference layout)
    S(X)        extractor parasitics of layout X         (S(B) once per reference layout, S(L) per iteration)
    s(N, P)     crosstalk-aware endpoint slack of the signoff timer on netlist N annotated with parasitics P

Every function takes arrays aligned on the same endpoint order (see ``align``).  The timer runs are outside this
package: any signoff timer that writes ``endpoint<TAB>slack`` lines can feed it.

The CaPDE estimate and the three comparison estimators of Section III-A:

    estimate                   reference term      extracted difference
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
    """x_ba = s(N_L, G(B)): keeps the back-annotated reference and ignores the wire part."""
    return np.asarray(s_nl_gb, float).copy()


def carried_offset(s_nb_gb, s_nl_sl, s_nb_sb):
    """x_off = s(N_B, G(B)) + [s(N_L, S(L)) - s(N_B, S(B))]: keeps a delta but times the reference on N_B."""
    return np.asarray(s_nb_gb, float) + (np.asarray(s_nl_sl, float) - np.asarray(s_nb_sb, float))


def learned_extraction_alone(s_nl_sl):
    """x_alone = s(N_L, S(L)): keeps only the extracted values of L."""
    return np.asarray(s_nl_sl, float).copy()


def estimator_family(s_nl_gb, s_nl_sl, s_nl_sb, s_nb_gb, s_nb_sb):
    """The CaPDE estimate and the three comparison estimators as a dict name -> array."""
    return {"capde": capde_estimate(s_nl_gb, s_nl_sl, s_nl_sb),
            "back_annotation_alone": back_annotation_alone(s_nl_gb),
            "carried_offset": carried_offset(s_nb_gb, s_nl_sl, s_nb_sb),
            "learned_extraction_alone": learned_extraction_alone(s_nl_sl)}


def reference_slack(g, g_repeat=None):
    """Field-solver slack s* (Eq. (1)): the more pessimistic of two field-solver runs on the optimized layout.

    g = s(N_L, G(L)); g_repeat = s(N_L, G'(L)) from a second field-solver run.  With one run, s* = g.
    The same minimum of s(N_L, G(B)) and s(N_L, G'(B)) gives the reference at its more pessimistic run (Section III-B)."""
    g = np.asarray(g, float)
    return g.copy() if g_repeat is None else np.minimum(g, np.asarray(g_repeat, float))


def cell_and_wire_part(s_nb_gb, s_nl_gb, g):
    """Split of the slack shift g - s(N_B, G(B)) of a sizing iteration (Section III-A).

    Returns (cell part s(N_L, G(B)) - s(N_B, G(B)), given exactly by the reference back-annotation;
    wire part g - s(N_L, G(B)), estimated by the layout delta)."""
    s_nb_gb, s_nl_gb, g = (np.asarray(a, float) for a in (s_nb_gb, s_nl_gb, g))
    return s_nl_gb - s_nb_gb, g - s_nl_gb


def error_stats(x, sstar):
    """Error of an estimate against the field-solver slack (Section IV-A): mean error (positive = optimistic),
    slack MSE and the 95th-percentile absolute slack error."""
    e = np.asarray(x, float) - np.asarray(sstar, float)
    return dict(n=int(len(e)), bias=float(e.mean()), mse=float((e ** 2).mean()), p95_abs=float(np.percentile(np.abs(e), 95)))


def error_correlation(s_nl_sl, s_nl_sb, g_l, g_b_on_nl):
    """Eqs. (3)-(4): correlation rho of the extractor slack errors eps_L = s(N_L, S(L)) - s(N_L, G(L)) and
    eps_B = s(N_L, S(B)) - s(N_L, G(B)), the MSE of extraction alone E[eps_L^2], the MSE of the CaPDE estimate
    E[(eps_L - eps_B)^2], and whether the condition 2 E[eps_L eps_B] > E[eps_B^2] for a lower CaPDE MSE holds."""
    eps_l = np.asarray(s_nl_sl, float) - np.asarray(g_l, float)
    eps_b = np.asarray(s_nl_sb, float) - np.asarray(g_b_on_nl, float)
    rho = float(np.corrcoef(eps_l, eps_b)[0, 1]) if eps_l.std() > 0 and eps_b.std() > 0 else float("nan")
    return dict(rho=rho, mse_alone=float((eps_l ** 2).mean()), mse_capde=float(((eps_l - eps_b) ** 2).mean()),
                condition_holds=bool(2 * float((eps_l * eps_b).mean()) > float((eps_b ** 2).mean())))
