"""Models of the final correction (paper Section III-D): gradient-boosted trees fit on training designs.

1. Net total  T(net) = sum over the net's tiles of g(x_tile): a per-tile model of geometry features alone.  Only net
   totals are labelled, so g is fit by aggregate-label boosting.  Tiles away from a reroute keep their prediction.
2. Pair model: rescales each coupling from the extractor's pair values, pair geometry proxies and a pair tile-additive
   geometry estimate.
Ground capacitance takes the rest of the net total (``apply.py``).
"""
import lightgbm as lgb
import numpy as np
from scipy.optimize import nnls

TREE = dict(learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
            num_threads=8, seed=0, deterministic=True, force_row_wise=True)
PAIR_STACK = dict(objective="l1", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.5,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0, deterministic=True, force_row_wise=True)
D_PAIR = 1e-4            # fF, offset of the pair log-ratio target
GAP_CAP_COLUMN = 5       # closest-gap column of the pair rows


def _additive_fit(X, group, y, weight, scale, hess_scale, params, rounds):
    """Aggregate-label boosting: minimize sum_groups w (scale * sum_rows f - y)^2 with f = linear base + trees.

    grad_row = scale * w (S - y); hess_row = hess_scale * w * rows(group) (majorizes the rank-one block of the Hessian)."""
    X = np.asarray(X, dtype=np.float64)
    n_rows = np.bincount(group, minlength=len(y)).astype(np.float64)
    Xn = np.zeros((len(y), X.shape[1]))
    np.add.at(Xn, group, X)
    Xn *= scale
    sc = Xn.std(0) + 1e-9
    beta, _ = nnls((Xn / sc) * np.sqrt(weight)[:, None], y * np.sqrt(weight), maxiter=10000)     # non-negative linear base
    beta = beta / sc
    base = X @ beta

    def objective(pred, _):
        S = scale * np.bincount(group, weights=pred, minlength=len(y))
        r = weight * (S - y)
        return scale * r[group], (hess_scale * weight * n_rows)[group]

    booster = lgb.train(dict(params, objective=objective), lgb.Dataset(X, label=np.zeros(len(X)), init_score=base, free_raw_data=False),
                        num_boost_round=rounds)
    return dict(beta=beta, booster=booster)


def _additive_rows(model, X):
    X = np.asarray(X, dtype=np.float64)
    return X @ model["beta"] + model["booster"].predict(X)


# ---- 1. net total: tile-additive ----------------------------------------------------------------------------------
def fit_tile_total(tile_X, tile_net, y_net, rounds=600, num_leaves=63, min_data_in_leaf=200):
    """tile_X (n_tiles, F), tile_net (n_tiles,) net index, y_net (n_nets,) field-solver net totals (fF, > 0).
    Relative least squares: w = 1 / max(y, 0.05)^2."""
    y = np.asarray(y_net, dtype=np.float64)
    w = 1.0 / np.maximum(y, 0.05) ** 2
    return _additive_fit(tile_X, np.asarray(tile_net), y, w, 1.0, 1.0, dict(TREE, num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf), rounds)


def predict_tile_total(model, tile_X, tile_net, n_nets):
    return np.bincount(np.asarray(tile_net), weights=_additive_rows(model, tile_X), minlength=n_nets)


# ---- 2a. pair tile-additive geometry estimate -----------------------------------------------------------------------
def pair_rows(tab, gap_cap):
    """Rows of ``features.net_tables`` -> (X, pair index per row, pair keys).  A pair is the unordered {victim, aggressor};
    rows whose aggressor is not an extracted net are dropped; the closest gap is capped."""
    nets = tab["nets"]
    known = set(nets)
    keys, pid, ok = {}, [], []
    for v, a in zip(tab["ptile_v"], tab["ptile_a"]):
        v = nets[v]
        if a not in known or a == v:
            ok.append(False)
            continue
        ok.append(True)
        pid.append(keys.setdefault((v, a) if v < a else (a, v), len(keys)))
    X = tab["ptile_X"][np.array(ok, dtype=bool)].astype(np.float64)
    X[:, GAP_CAP_COLUMN] = np.minimum(X[:, GAP_CAP_COLUMN], gap_cap)
    return X, np.array(pid, dtype=np.int64), list(keys)


def fit_pair_additive(X, pid, y_pair, rounds=600, w0=0.01, min_data_in_leaf=200):
    """c{v, a} = 1/2 (sum over v's tiles of g(x_{t, a}) + sum over a's tiles of g(x_{t, v})).
    y_pair: field-solver pair coupling (mean of the two sides; 0 for a candidate pair the reference does not list)."""
    y = np.asarray(y_pair, dtype=np.float64)
    return _additive_fit(X, pid, y, 1.0 / np.maximum(y, w0), 0.5, 0.25, dict(TREE, num_leaves=127, min_data_in_leaf=min_data_in_leaf), rounds)


def predict_pair_additive(model, X, pid, keys):
    S = 0.5 * np.bincount(pid, weights=_additive_rows(model, X), minlength=len(keys))
    return dict(zip(keys, np.maximum(S, 0.0)))


# ---- 2b. pair stack -------------------------------------------------------------------------------------------------
PAIR_FEATURES = (["c1", "c2", "csym", "listed", "T1", "T2", "Cs1", "Cs2", "nagg1", "nagg2"]
                 + [f"{p}_{k}" for k in (1, 2) for p in ("LATb0", "LATb1", "LATb2", "VUP", "VDN", "mingap", "rows")]
                 + ["share1", "share2", "rank1", "rank2"])
_LOG_COLUMNS = (0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21)


def pair_table(pair_sums, totals, proxies, gap_cap, reference_pairs=None):
    """Candidate pairs and their features.

    pair_sums   {v: {a: c}} directed coupling sums of the extractor output (block of v towards a)
    totals      {v: net total capacitance of the extractor output}
    proxies     {v: {a: 7 pair proxies}} (``features.net_tables``['pairs'])
    Candidates = pairs listed by the extractor (either side) or by the proxies (either direction).  Side 1 is the net
    with the larger extracted total.  Returns dict(keys, X, [y]) with y = reference pair coupling (mean of listed sides)."""
    nets = set(totals)
    cand = set()
    for src in (pair_sums, proxies):
        for v, d in src.items():
            for a in d:
                if a in nets and v in nets and a != v:
                    cand.add((v, a) if v < a else (a, v))
    cs_net = {v: sum(d.values()) for v, d in pair_sums.items()}
    n_agg = {v: len(d) for v, d in pair_sums.items()}
    rank = {v: {a: i for i, (a, _) in enumerate(sorted(d.items(), key=lambda z: -z[1]))} for v, d in pair_sums.items()}
    zero = np.zeros(7)
    capped = lambda p: [float(min(x, gap_cap)) if i == 5 else float(x) for i, x in enumerate(p)]      # noqa: E731
    rows, keys, y = [], [], []
    for v, a in sorted(cand):
        if totals[v] < totals[a]:
            v, a = a, v
        c1, c2 = pair_sums.get(v, {}).get(a, 0.0), pair_sums.get(a, {}).get(v, 0.0)
        listed = (c1 > 0) + (c2 > 0)
        csym = (c1 + c2) / listed if listed else 0.0
        rows.append([c1, c2, csym, listed, totals[v], totals[a], cs_net.get(v, 0.0), cs_net.get(a, 0.0), n_agg.get(v, 0), n_agg.get(a, 0)]
                    + capped(proxies.get(v, {}).get(a, zero)) + capped(proxies.get(a, {}).get(v, zero))
                    + [c1 / (cs_net.get(v, 0.0) + 1e-9), c2 / (cs_net.get(a, 0.0) + 1e-9), rank.get(v, {}).get(a, 999), rank.get(a, {}).get(v, 999)])
        keys.append((v, a))
        if reference_pairs is not None:
            g1, g2 = reference_pairs.get(v, {}).get(a, 0.0), reference_pairs.get(a, {}).get(v, 0.0)
            y.append((g1 + g2) / ((g1 > 0) + (g2 > 0)) if (g1 > 0 or g2 > 0) else 0.0)
    out = dict(keys=keys, X=np.array(rows, dtype=np.float64).reshape(-1, len(PAIR_FEATURES)))
    if reference_pairs is not None:
        out["y"] = np.array(y)
    return out


def _stack_X(tab, c_additive):
    X = tab["X"].copy()
    for j in _LOG_COLUMNS:
        X[:, j] = np.log(X[:, j] + 1e-5)
    ct = np.array([c_additive.get((a, b) if a < b else (b, a), 0.0) for a, b in tab["keys"]])
    cs = tab["X"][:, 2]
    return np.column_stack([X, np.log(ct + 1e-5), np.log((ct + 1e-4) / (cs + 1e-4))]), cs


def fit_pair_stack(tables, additive, rounds=800, share_weight=10.0, min_data_in_leaf=100):
    """tables: list of ``pair_table`` dicts with 'y' (one per training design); additive: list of pair-additive estimates
    of the same designs (out-of-fold: from a model fit without that design).
    Target ln((y + d) / (c_sym + d)); weight sqrt(max(y, c)) x (1 + share_weight x min(c_sym / min(T1, T2), 1))."""
    Xs, ys, cs, rs = [], [], [], []
    for t, add in zip(tables, additive):
        X, c = _stack_X(t, add)
        Xs.append(X)
        ys.append(t["y"])
        cs.append(c)
        rs.append(t["X"][:, 2] / (np.minimum(t["X"][:, 4], t["X"][:, 5]) + 1e-9))
    X, y, c, r = np.vstack(Xs), np.concatenate(ys), np.concatenate(cs), np.concatenate(rs)
    w = np.sqrt(np.maximum(y, c) + 1e-4) * (1.0 + share_weight * np.minimum(r, 1.0))
    return lgb.train(dict(PAIR_STACK, min_data_in_leaf=min_data_in_leaf), lgb.Dataset(X, np.log((y + D_PAIR) / (c + D_PAIR)), weight=w),
                     num_boost_round=rounds)


def predict_pair_stack(booster, table, additive):
    """-> {unordered pair (sorted names): corrected coupling}."""
    X, cs = _stack_X(table, additive)
    c = np.maximum((cs + D_PAIR) * np.exp(booster.predict(X)) - D_PAIR, 0.0)
    return {((a, b) if a < b else (b, a)): float(x) for (a, b), x in zip(table["keys"], c)}
