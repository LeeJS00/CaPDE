"""Design-level statistics of the evaluation (paper Section IV-A): design-cluster bootstrap 95% confidence intervals
with 2,000 resamples and the exact two-sided sign-flip test, uncorrected for multiple tests.

Endpoints of one design are not independent, so every interval and test resamples or permutes whole designs."""
import itertools

import numpy as np


def bootstrap_weights(n_designs, n_boot=2000, seed=0):
    """(n_boot, n_designs) matrix of design multiplicities: each row resamples the designs with replacement."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_boot, n_designs))
    for i in range(n_boot):
        w[i] = np.bincount(rng.integers(0, n_designs, n_designs), minlength=n_designs)
    return w


def bootstrap_ratio(num, den, weights):
    """Per resample: pooled ratio sum_d w_d num_d / sum_d w_d den_d.  num, den: per-design sums (n_designs,) or
    (n_designs, n_seeds); with extractor seeds the pooled ratio is formed per seed and averaged over the seeds."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if num.ndim == 1:
        num, den = num[:, None], den[:, None]
    a, b = weights @ num, weights @ den
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan), axis=1)


def bootstrap_mean_ci(per_design, n_boot=2000, seed=0, level=0.95):
    """Percentile interval of the mean of per-design values under design resampling."""
    v = np.asarray(per_design, float)
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    lo = 100 * (1 - level) / 2
    return [float(np.percentile(m, lo)), float(np.percentile(m, 100 - lo))]


def percentile_ci(samples, level=0.95):
    """Percentile interval of bootstrap samples (non-finite samples are ignored)."""
    v = np.asarray(samples, float)
    v = v[np.isfinite(v)]
    lo = 100 * (1 - level) / 2
    return [float(np.percentile(v, lo)), float(np.percentile(v, 100 - lo))] if len(v) else [None, None]


def signflip(per_design):
    """Exact two-sided sign-flip test on per-design paired differences; statistic |mean|.

    Returns (p, smallest attainable p = 2 / 2^N for N nonzero differences; 0.0156 for seven designs).  All 2^N sign
    patterns are enumerated."""
    v = np.asarray(per_design, float)
    n = len(v)
    obs = abs(v.mean())
    stat = np.array([abs((v * np.array(sg)).mean()) for sg in itertools.product((1, -1), repeat=n)])
    nz = int(np.sum(v != 0))
    return float(np.mean(stat >= obs - 1e-12)), (2.0 / 2 ** nz if nz else 1.0)


def paired_test(per_design, alpha=0.05, n_boot=2000, seed=0):
    """Contrast of per-design differences: mean, design-cluster bootstrap 95% interval and sign-flip p.

    status: 'negative' / 'positive' when the interval excludes 0 on that side and p < alpha (a direction is stated);
    otherwise 'not_distinguished'."""
    v = np.asarray(per_design, float)
    p, floor = signflip(v)
    ci = bootstrap_mean_ci(v, n_boot, seed)
    out = dict(mean=float(v.mean()), ci95=ci, p_signflip=p, floor_p=floor, n_designs=len(v))
    if ci[1] < 0 and p < alpha:
        out["status"] = "negative"
    elif ci[0] > 0 and p < alpha:
        out["status"] = "positive"
    else:
        out["status"] = "not_distinguished"
    return out
