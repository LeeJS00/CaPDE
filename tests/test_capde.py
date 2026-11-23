"""Estimators, interval, decision, deferral and statistics on a synthetic table with hand-computed answers."""
import numpy as np

from capde import deferral, estimate, interval, stats

# six endpoints; slacks of the five timer runs (arbitrary units)
S_NL_GB = [10.0, 2.0, -1.0, 0.5, -5.0, 3.0]
S_NL_SL = [9.0, 1.5, -2.5, 1.5, -6.0, 2.0]
S_NL_SB = [9.5, 2.5, -1.5, 0.5, -5.5, 2.5]
S_NB_GB = [11.0, 3.0, 0.0, 1.0, -4.0, 4.0]
S_NB_SB = [10.0, 3.5, -0.5, 1.5, -4.5, 3.0]
G, G_REPEAT = [9.4, 0.8, -1.6, -0.2, -5.2, 1.2], [9.6, 1.1, -1.9, 0.1, -5.6, 1.4]


def test_estimators():
    fam = estimate.estimator_family(S_NL_GB, S_NL_SL, S_NL_SB, S_NB_GB, S_NB_SB)
    assert np.allclose(fam["capde"], [9.5, 1.0, -2.0, 1.5, -5.5, 2.5])
    assert np.allclose(fam["back_annotation_alone"], S_NL_GB)
    assert np.allclose(fam["carried_offset"], [10.0, 1.0, -2.0, 1.0, -5.5, 3.0])
    assert np.allclose(fam["learned_extraction_alone"], S_NL_SL)
    assert np.allclose(estimate.reference_slack(G, G_REPEAT), [9.4, 0.8, -1.9, -0.2, -5.6, 1.2])


def test_identity():
    """Eq. (3): x - s(N_L, G(L)) = eps_L - eps_B for one field-solver run."""
    x = estimate.capde_estimate(S_NL_GB, S_NL_SL, S_NL_SB)
    eps_l = np.array(S_NL_SL) - np.array(G)
    eps_b = np.array(S_NL_SB) - np.array(S_NL_GB)
    assert np.allclose(x - np.array(G), eps_l - eps_b)


def test_half_width():
    r = [0.1, 0.2, 0.1, 1.7, 0.1, 1.3]
    assert interval.fit_half_width(r, alpha=0.05) == 1.7          # rank ceil(7 * 0.95) = 7, capped at n = 6
    assert interval.fit_half_width(r, alpha=0.5) == 0.2           # rank ceil(7 * 0.5) = 4
    assert interval.fit_half_width(np.arange(1, 40), alpha=0.05) == 38.0     # n = 39: rank ceil(40 * 0.95) = 38


def test_decision_counts():
    x = estimate.capde_estimate(S_NL_GB, S_NL_SL, S_NL_SB)
    sstar = estimate.reference_slack(G, G_REPEAT)
    p, f, d = interval.decide(x, 1.0, 0.0)
    assert p.tolist() == [True, False, False, True, False, True]
    assert f.tolist() == [False, False, True, False, True, False]
    assert d.tolist() == [False, True, False, False, False, False]
    c = interval.counts(x, 1.0, sstar, 0.0)
    assert (c["n"], c["violators"], c["passed"], c["failed"], c["deferred"], c["decided"]) == (6, 3, 3, 2, 1, 5)
    assert (c["missed"], c["false_violations_open"], c["false_violations_failed"]) == (1, 0, 1)
    assert abs(c["coverage"] - 4 / 6) < 1e-12
    assert abs(interval.rates(c)["missed_per_decided"] - 0.2) < 1e-12


def test_deferral():
    lab = interval.labels([5.0, 0.5, -0.5, -5.0], 1.0)
    assert lab.tolist() == ["pass", "defer", "defer", "fail"]
    assert deferral.default_fail(lab).tolist() == ["pass", "fail", "fail", "fail"]
    assert interval.redecide(lab, [np.nan, 3.0, 0.5, np.nan], 1.0).tolist() == ["pass", "pass", "defer", "fail"]
    seq = {"e1": ["n1", "n2", "n3"], "e2": ["n2", "n4"]}
    assert deferral.round_robin(seq, {"e1": -1.0, "e2": -2.0}, 4) == ["n2", "n1", "n4", "n3"]
    paths = {"e1": (-1.0, [["n1", "n2"], ["n3"]]), "e2": (-2.0, [["n2", "n4"]])}
    nets, pool = deferral.candidate_nets(["e1", "e2"], paths, beta=0.5)
    assert (nets, pool) == (["n2", "n1"], 4)
    nets, pool = deferral.candidate_nets(["e1", "e2"], paths, beta=1.0, coupled={"n1", "n4"})
    assert (nets, pool) == (["n4", "n1"], 2)
    assert "{extractor_cmd}" in deferral.reextract_script("nets.txt") and "nets.txt" in deferral.reextract_script("nets.txt")


def test_stats():
    p, floor = stats.signflip([1.0, 2.0, 3.0])
    assert (p, floor) == (0.25, 0.25)
    w = stats.bootstrap_weights(5, n_boot=200, seed=1)
    assert w.shape == (200, 5) and np.all(w.sum(axis=1) == 5)
    r = stats.bootstrap_ratio([1, 0, 2, 0, 1], [10, 10, 10, 10, 10], w)
    lo, hi = stats.percentile_ci(r)
    assert 0.0 <= lo <= 0.08 <= hi <= 0.2
    t = stats.paired_test([-1.0, -2.0, -1.5, -0.5, -1.2, -0.8, -1.1])
    assert t["status"] == "negative" and t["n_designs"] == 7 and abs(t["p_signflip"] - 2 / 128) < 1e-12
    assert stats.paired_test([-1.0, -2.0, -1.5])["status"] == "not_distinguished"     # smallest p with 3 designs is 0.25


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
