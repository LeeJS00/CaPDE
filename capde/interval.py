"""Decision intervals and the three-way decision (paper Sections II and III-B, Eq. (5)).

The fixed-width interval uses one half-width b for every endpoint: the finite-sample (1 - alpha) quantile of the
absolute training residuals by the split-conformal rule.  The residuals are those of the layout delta against the
field-solver shift s* - min(s(N_L, G(B)), s(N_L, G'(B))) on the optimized layouts of the interval-fit designs, with
both layouts at their more pessimistic field-solver run.  Decision at required slack theta:

    pass  if x - b > theta        fail  if x + b < theta        defer otherwise

Outcomes (Section II): a violator has s* < theta; a pass of a violator is a missed violation; a fail with s* >= theta
is a false violation.  A deferral left open is neither; a deferral that fails by default counts as a fail.
"""
import math

import numpy as np


def fit_half_width(abs_residuals, alpha=0.05):
    """Split-conformal half-width: the value at rank ceil((n + 1)(1 - alpha)) of the sorted absolute residuals
    (capped at n); 1 - alpha is the target coverage."""
    r = np.sort(np.abs(np.asarray(abs_residuals, float)))
    n = len(r)
    if n == 0:
        raise ValueError("no residual to fit on")
    rank = min(int(math.ceil((n + 1) * (1.0 - alpha))), n)
    return float(r[rank - 1])


def decide(x, b, theta=0.0):
    """Eq. (5).  x: estimates; b: half-width (the fixed width b).  Returns boolean arrays (passed, failed, deferred)."""
    x = np.asarray(x, float)
    passed = (x - b) > theta
    failed = (x + b) < theta
    return passed, failed, ~(passed | failed)


def coverage(x, b, sstar):
    """Coverage (Section II): share of endpoints whose field-solver slack s* lies inside [x - b, x + b]."""
    x, sstar = np.asarray(x, float), np.asarray(sstar, float)
    return float(np.mean(np.abs(x - sstar) <= b)) if len(x) else float("nan")


def counts(x, b, sstar, theta=0.0):
    """Decision counts of one layout: violators, passes, fails, deferrals, missed and false violations, coverage."""
    x, sstar = np.asarray(x, float), np.asarray(sstar, float)
    p, f, d = decide(x, b, theta)
    v = sstar < theta
    pv, pn = int((p & v).sum()), int((p & ~v).sum())
    fv, fn = int((f & v).sum()), int((f & ~v).sum())
    dn = int((d & ~v).sum())
    return dict(n=len(x), violators=int(v.sum()), passed=pv + pn, failed=fv + fn, deferred=int(d.sum()), decided=pv + pn + fv + fn,
                missed=pv,                              # pass of a violator
                false_violations_open=fn,               # deferrals left open
                false_violations_failed=fn + dn,        # deferrals fail by default
                coverage=coverage(x, b, sstar))


def rates(c):
    """Shares of a ``counts`` dict (None where the denominator is 0): decided and deferred endpoints, missed
    violations among decided endpoints (the risk of the risk-coverage curve) and among violators."""
    div = lambda a, b: (a / b) if b else None            # noqa: E731
    return dict(decision_rate=div(c["decided"], c["n"]), deferral_rate=div(c["deferred"], c["n"]),
                missed_per_decided=div(c["missed"], c["decided"]), missed_per_violator=div(c["missed"], c["violators"]))


def redecide(decision, s_reextracted, b_beta, theta=0.0):
    """Re-decision after a partial field-solver re-extraction (Section III-C): Eq. (5) re-decides only the deferred
    endpoints, on their slack from the partial run and the half-width b_beta fit for the field-solved share beta.

    decision: array of 'pass' | 'fail' | 'defer'; s_reextracted: slack per endpoint (NaN where not available)."""
    decision = np.asarray(decision, dtype=object).copy()
    s = np.asarray(s_reextracted, float)
    open_ = (decision == "defer") & np.isfinite(s)
    decision[open_ & (s - b_beta > theta)] = "pass"
    decision[open_ & (s + b_beta < theta)] = "fail"
    return decision


def labels(x, b, theta=0.0):
    """Eq. (5) as an array of 'pass' | 'fail' | 'defer'."""
    p, f, _ = decide(x, b, theta)
    return np.where(p, "pass", np.where(f, "fail", "defer")).astype(object)
