"""Statistics for the classifier comparison [paper, 'Classification metrics and statistical analysis']."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------------------------------- DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Paired DeLong test for two correlated AUCs (fast algorithm of Sun & Xu, 2014).

    Returns AUCs, their difference (auc2 - auc1), its SE, z and two-sided p.
    """
    y = np.asarray(y).astype(int)
    preds = np.vstack([p1, p2]).astype(float)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[1] - aucs[0]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 1e-300)))
    z = diff / se
    p = 2 * stats.norm.sf(abs(z))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta_auc": float(diff),
            "se": se, "z": float(z), "p": float(p)}


# --------------------------------------------------------------------------- threshold metrics
def threshold_metrics(y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and not np.isnan(prec) else np.nan
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": (tp + tn) / len(y),
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def all_metrics(y, p, thr: float = 0.5) -> dict:
    out = threshold_metrics(y, p, thr)
    out["auc"] = float(roc_auc_score(y, p))
    out["brier"] = brier(y, p)
    return out


# --------------------------------------------------------------------------- operating points
def youden_point(y, p) -> dict:
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    return {"threshold": float(min(thr[i], 1.0)), "sensitivity": float(tpr[i]), "specificity": float(1 - fpr[i])}


def spec_at_sens(y, p, min_sens: float = 0.90) -> dict:
    """Highest specificity among thresholds whose sensitivity is >= min_sens."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= min_sens - 1e-12)[0]
    i = ok[np.argmin(fpr[ok])]
    return {"threshold": float(min(thr[i], 1.0)), "sensitivity": float(tpr[i]), "specificity": float(1 - fpr[i])}


def sens_at_spec(y, p, min_spec: float = 0.95) -> dict:
    """Highest sensitivity among thresholds whose specificity is >= min_spec."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(1 - fpr >= min_spec - 1e-12)[0]
    i = ok[np.argmax(tpr[ok])]
    return {"threshold": float(min(thr[i], 1.0)), "sensitivity": float(tpr[i]), "specificity": float(1 - fpr[i])}


def test_demand(se: float, sp: float, prevalence: float) -> dict:
    """[paper, Eqs. 5-6] confirmatory tests per 1,000 screened and per TB case detected."""
    pos_rate = prevalence * se + (1 - prevalence) * (1 - sp)
    return {"tests_per_1000": 1000 * pos_rate,
            "tests_per_case": pos_rate / (prevalence * se),
            "share_avoided": 1 - pos_rate}


# --------------------------------------------------------------------------- uncertainty
def bootstrap_ci(y, p, fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of fn(y, p) over resamples of the paired predictions (skips one-class resamples)."""
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    while len(vals) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], p[idx]))
    vals = np.asarray(vals, float)
    if np.isnan(vals).all():  # metric undefined in every resample (e.g. precision with no positive calls)
        return float("nan"), float("nan")
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    den = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return float(lo), float(hi)


def mcnemar_exact(y, p1, p2, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    c1 = (np.asarray(p1) >= thr).astype(int) == y
    c2 = (np.asarray(p2) >= thr).astype(int) == y
    b = int((c1 & ~c2).sum())   # model 1 right, model 2 wrong
    c = int((~c1 & c2).sum())   # model 2 right, model 1 wrong
    pval = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"model1_only_correct": b, "model2_only_correct": c, "p": pval}


def bland_altman(a, b) -> dict:
    d = np.asarray(b, float) - np.asarray(a, float)
    bias = float(d.mean()); sd = float(d.std(ddof=1))
    return {"bias": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}
