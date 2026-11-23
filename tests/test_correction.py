"""Final correction on synthetic data: features, per-tile net capacitance, pair model, application."""
import os

import numpy as np

from basiscap import load_config, windows
from basiscap.correction import apply, features, models

CFG = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "example.yaml"))


def _tiles(n_nets=40, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_nets):
        tiles = []
        for _ in range(int(rng.integers(1, 4))):
            t = windows.synthetic_window(CFG.technology, rng, n_target=int(rng.integers(1, 4)))
            t["net_names"] = [f"n{i}" if n == "net_t" else f"n{(i + 1 + int(n[-1])) % n_nets}" if n.startswith("net_a") else n for n in t["net_names"]]
            tiles.append(t)
        out[f"n{i}"] = tiles
    return out


def test_features():
    tab = features.net_tables(_tiles(), CFG.technology, CFG.correction)
    names = features.tile_names(CFG.technology)
    assert tab["tile_X"].shape[1] == len(names) and len(tab["tile_X"]) == len(tab["tile_net"]) > 0
    assert np.all(np.isfinite(tab["tile_X"])) and np.all(tab["tile_X"][:, names.index("n_own")] >= 1)
    area = tab["tile_X"][:, :len(features.classes(CFG.technology))].sum(axis=1)
    assert np.all(area > 0), "every tile owns metal area"
    assert tab["ptile_X"].shape[1] == len(features.PAIR_TILE_NAMES)


def test_tile_total_fit_apply():
    tab = features.net_tables(_tiles(120, seed=1), CFG.technology, CFG.correction)
    n = len(tab["nets"])
    names = features.tile_names(CFG.technology)
    nc = len(features.classes(CFG.technology))
    side = np.bincount(tab["tile_net"], weights=tab["tile_X"][:, nc:2 * nc].sum(axis=1), minlength=n)
    y = 0.05 + 3.0 * side                                         # a synthetic 'net total' that is additive over tiles
    m = models.fit_tile_total(tab["tile_X"], tab["tile_net"], y, rounds=30, min_data_in_leaf=5)
    pred = models.predict_tile_total(m, tab["tile_X"], tab["tile_net"], n)
    assert pred.shape == (n,) and np.all(np.isfinite(pred))
    assert np.median(np.abs(pred - y) / y) < 0.25, "the tile-additive model recovers an additive target"
    assert len(names) == tab["tile_X"].shape[1]


def test_pair_model_and_apply():
    tab = features.net_tables(_tiles(120, seed=2), CFG.technology, CFG.correction)
    gap_cap = float(CFG.correction.gap_cap)
    X, pid, keys = models.pair_rows(tab, gap_cap)
    assert len(keys) > 0 and len(X) == len(pid)
    rng = np.random.default_rng(0)
    lat = 0.5 * np.bincount(pid, weights=X[:, :3].sum(axis=1), minlength=len(keys))
    truth = dict(zip(keys, 0.002 + 0.01 * lat))                    # synthetic field-solver pair couplings
    add = models.fit_pair_additive(X, pid, np.array([truth[k] for k in keys]), rounds=20, min_data_in_leaf=5)
    c_add = models.predict_pair_additive(add, X, pid, keys)
    assert all(v >= 0 for v in c_add.values())

    # a synthetic extraction result: the field-solver couplings with multiplicative error, one node per net
    caps = {v: dict(ground=[[f"{v}:1", 0.5]], coupling=[]) for v in tab["nets"]}
    ref_pairs = {v: {} for v in tab["nets"]}
    for (a, b), c in truth.items():
        for v, o in ((a, b), (b, a)):
            caps[v]["coupling"].append([f"{v}:1", o, c * float(rng.uniform(0.5, 2.0))])
            ref_pairs[v][o] = c
    table = models.pair_table(apply.pair_sums(caps), apply.net_totals(caps), tab["pairs"], gap_cap, label_pairs=ref_pairs)
    assert table["X"].shape == (len(table["keys"]), len(models.PAIR_FEATURES)) and len(table["y"]) == len(table["keys"])
    pair_model = models.fit_pair_model([table], [c_add], rounds=20, min_data_in_leaf=5)
    c_pair = models.predict_pair_model(pair_model, table, c_add)
    assert all(v >= 0 for v in c_pair.values())

    target = {v: 1.0 for v in tab["nets"]}
    out = apply.correct(caps, target, c_pair)
    before, after = apply.pair_sums(caps), apply.pair_sums(out)
    assert any(abs(after[v][a] - x) > 1e-12 for v, d in before.items() for a, x in d.items()), "couplings are rescaled"
    tot = apply.net_totals(out)
    for v in tab["nets"]:
        f = sum(c for _, c in out[v]["ground"]) / sum(c for _, c in caps[v]["ground"])
        assert abs(tot[v] - 1.0) < 1e-9 or min(abs(f - apply.G_MIN), abs(f - apply.G_MAX)) < 1e-9, \
            "ground takes the rest of the net total unless its factor is clipped"
    assert all(c > 0 for blk in out.values() for _, c in blk["ground"])


def test_threshold_and_symmetry():
    caps = {"a": dict(ground=[["a:1", 1.0]], coupling=[["a:1", "b", 0.30], ["a:2", "b", 0.00005], ["a:1", "c", 0.2]]),
            "b": dict(ground=[["b:1", 2.0]], coupling=[["b:1", "a", 0.10]]),
            "c": dict(ground=[["c:1", 1.0]], coupling=[])}
    before = apply.net_totals(caps)
    th = apply.coupling_threshold(caps, 1e-4)
    assert th["a"]["ground"] == [["a:1", 1.0], ["a:2", 0.00005]] and len(th["a"]["coupling"]) == 2
    assert all(abs(apply.net_totals(th)[v] - before[v]) < 1e-12 for v in caps), "net totals are unchanged by the threshold"
    sym = apply.symmetrize(th)
    ps = apply.pair_sums(sym)
    assert abs(ps["a"]["b"] - 0.2) < 1e-12 and abs(ps["b"]["a"] - 0.2) < 1e-12, "both sides carry the mean"
    assert abs(ps["a"]["c"] - 0.2) < 1e-12, "a pair listed on one side only is left as is"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
