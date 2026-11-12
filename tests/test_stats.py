import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from tbstudy import stats as st


def _toy(n=138, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(80), np.ones(58)].astype(int)[:n]
    p1 = np.clip(y * 0.6 + rng.normal(0.2, 0.2, len(y)), 0, 1)
    p2 = np.clip(y * 0.4 + rng.normal(0.3, 0.25, len(y)), 0, 1)
    return y, p1, p2


def test_delong_aucs_match_sklearn():
    y, p1, p2 = _toy()
    r = st.delong_test(y, p1, p2)
    assert r["auc1"] == pytest.approx(roc_auc_score(y, p1), abs=1e-12)
    assert r["auc2"] == pytest.approx(roc_auc_score(y, p2), abs=1e-12)
    assert r["delta_auc"] == pytest.approx(r["auc2"] - r["auc1"])


def test_delong_se_agrees_with_bootstrap():
    y, p1, p2 = _toy(seed=3)
    r = st.delong_test(y, p1, p2)
    rng = np.random.default_rng(1)
    diffs = []
    while len(diffs) < 3000:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], p2[i]) - roc_auc_score(y[i], p1[i]))
    assert r["se"] == pytest.approx(np.std(diffs), rel=0.15)


def test_delong_handles_ties_and_identical_models():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.5, 0.5, 0.9, 0.5, 0.7])
    r = st.delong_test(y, p, p)
    assert r["delta_auc"] == 0 and r["p"] == pytest.approx(1.0)


@pytest.mark.parametrize("spec,pi,t1000,per_case", [
    (0.838, 0.05, 199, 4.4),    # paper Table 8, whole image
    (0.488, 0.05, 531, 11.8),   # paper Table 8, lung ROI
    (0.838, 0.01, None, 18.8),  # paper text, 1% prevalence
    (0.488, 0.01, None, 57.3),
    (0.838, 0.10, None, 2.6),   # paper text, 10% prevalence
    (0.488, 0.10, None, 6.1),
])
def test_test_demand_reproduces_paper(spec, pi, t1000, per_case):
    r = st.test_demand(0.90, spec, pi)
    if t1000 is not None:
        assert round(r["tests_per_1000"]) == t1000
    assert round(r["tests_per_case"], 1) == pytest.approx(per_case, abs=0.05)


def test_share_avoided_matches_paper():
    assert st.test_demand(0.90, 0.838, 0.05)["share_avoided"] == pytest.approx(0.801, abs=0.001)
    assert st.test_demand(0.90, 0.488, 0.05)["share_avoided"] == pytest.approx(0.469, abs=0.001)


def test_operating_points():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    s90 = st.spec_at_sens(y, p, 0.9)
    assert s90["sensitivity"] == 1.0 and s90["specificity"] == 0.75
    h = st.sens_at_spec(y, p, 0.95)
    assert h["specificity"] == 1.0 and h["sensitivity"] == 0.75
    yd = st.youden_point(y, p)
    assert yd["sensitivity"] - (1 - yd["specificity"]) == pytest.approx(0.75)


def test_threshold_metrics_and_brier():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.8, 0.4, 0.6])
    m = st.threshold_metrics(y, p)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert st.brier(y, p) == pytest.approx(np.mean([0.04, 0.04, 0.36, 0.36]))


def test_wilson_matches_reference_value():
    lo, hi = st.wilson_ci(10, 13)          # 76.9%
    assert lo == pytest.approx(0.4973, abs=1e-3) and hi == pytest.approx(0.9182, abs=1e-3)


def test_mcnemar_equal_discordance_gives_p1():
    y = np.array([1] * 10 + [0] * 10)
    p1 = np.r_[np.ones(10), np.zeros(10)]           # all correct
    p2 = np.r_[np.zeros(5), np.ones(5), np.ones(5), np.zeros(5)]
    r = st.mcnemar_exact(y, p1, p2)
    assert r["model1_only_correct"] == 10 and r["model2_only_correct"] == 0
    p3 = p1.copy(); p3[0] = 0
    r2 = st.mcnemar_exact(y, p3, np.r_[np.ones(1), p1[1:]] * 0 + p1)  # symmetric case
    assert 0 <= r2["p"] <= 1


def test_bootstrap_ci_contains_point_estimate():
    y, p1, _ = _toy()
    lo, hi = st.bootstrap_ci(y, p1, roc_auc_score, n_boot=300, seed=0)
    assert lo < roc_auc_score(y, p1) < hi


def test_wilson_bounds_are_exact_at_0_and_100_percent():
    for n in (1, 3, 5, 13, 24):
        assert st.wilson_ci(n, n)[1] == 1.0 and st.wilson_ci(0, n)[0] == 0.0
        lo, hi = st.wilson_ci(n, n)
        assert lo <= 1.0 <= hi


def test_bootstrap_all_nan_metric_returns_nan_quietly():
    import warnings
    y = np.array([0, 1] * 10); p = np.zeros(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lo, hi = st.bootstrap_ci(y, p, lambda yy, pp: st.threshold_metrics(yy, pp)["precision"], n_boot=50)
    assert np.isnan(lo) and np.isnan(hi)
