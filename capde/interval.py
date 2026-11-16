"""Decision intervals and the three-way decision (paper Section III-B, Eq. (5)).

The fixed-width interval uses one half-width b for every endpoint: the finite-sample (1 - alpha) quantile of the absolute
residuals |x - s*| of training layouts (split-conformal rule).  Decision at required slack theta:

    pass  if x - b > theta        fail  if x + b < theta        defer otherwise

Outcome names (Section II): a violator has s* < theta; a pass of a violator is a missed violation; a fail with
s* >= theta is a false violation.  With the default policy a deferral fails, so 'false violations (deferrals failed)'
adds the deferred non-violators; 'deferrals open' counts neither.
"""
import math

import numpy as np


def fit_half_width(abs_residuals, alpha=0.05):
    """Split-conformal half-width: the value at rank ceil((n + 1)(1 - alpha)) of the sorted |x - s*| (capped at n)."""
    r = np.sort(np.abs(np.asarray(abs_residuals, float)))
    n = len(r)
    if n == 0:
        raise ValueError("no residual to fit on")
    rank = min(int(math.ceil((n + 1) * (1.0 - alpha))), n)
    return float(r[rank - 1])


def decide(x, b, theta=0.0):
    """Eq. (5).  x: estimates; b: scalar or per-endpoint half-width.  Returns boolean arrays (passed, failed, deferred)."""
    x = np.asarray(x, float)
    passed = (x - b) > theta
    failed = (x + b) < theta
    return passed, failed, ~(passed | failed)


def coverage(x, b, sstar):
    """Share of endpoints whose reference slack lies inside [x - b, x + b]."""
    x, sstar = np.asarray(x, float), np.asarray(sstar, float)
    return float(np.mean(np.abs(x - sstar) <= b)) if len(x) else float("nan")


def counts(x, b, sstar, theta=0.0):
    """Six-cell table (decision x violator) and the derived counts of one layout."""
    x, sstar = np.asarray(x, float), np.asarray(sstar, float)
    p, f, d = decide(x, b, theta)
    v = sstar < theta
    pv, pn = int((p & v).sum()), int((p & ~v).sum())
    fv, fn = int((f & v).sum()), int((f & ~v).sum())
    dv, dn = int((d & v).sum()), int((d & ~v).sum())
    return dict(n=len(x), violators=int(v.sum()), passed=pv + pn, failed=fv + fn, deferred=dv + dn, decided=pv + pn + fv + fn,
                missed=pv,                              # pass of a violator
                false_violations_open=fn,               # deferrals left open
                false_violations_failed=fn + dn,        # deferrals fail by default
                deferred_violators=dv, deferred_non_violators=dn, coverage=coverage(x, b, sstar))


def rates(c):
    """Ratios of a ``counts`` dict (None where the denominator is 0)."""
    div = lambda a, b: (a / b) if b else None            # noqa: E731
    return dict(decision_rate=div(c["decided"], c["n"]), deferral_rate=div(c["deferred"], c["n"]),
                missed_per_decided=div(c["missed"], c["decided"]), missed_per_passed=div(c["missed"], c["passed"]),
                missed_per_violator=div(c["missed"], c["violators"]),
                false_violations_open_per_decided=div(c["false_violations_open"], c["decided"]))


def redecide(decision, s_reextracted, b_budget, theta=0.0):
    """Re-decision after a partial re-extraction (Section III-C): only the deferred endpoints are re-decided, on their
    slack from the re-extracted parasitics and a half-width fit for that re-extraction budget.

    decision: array of 'pass' | 'fail' | 'defer'; s_reextracted: slack per endpoint (NaN where not available)."""
    decision = np.asarray(decision, dtype=object).copy()
    s = np.asarray(s_reextracted, float)
    todo = (decision == "defer") & np.isfinite(s)
    decision[todo & (s - b_budget > theta)] = "pass"
    decision[todo & (s + b_budget < theta)] = "fail"
    return decision


def labels(x, b, theta=0.0):
    """Decision as an array of 'pass' | 'fail' | 'defer'."""
    p, f, _ = decide(x, b, theta)
    return np.where(p, "pass", np.where(f, "fail", "defer")).astype(object)
