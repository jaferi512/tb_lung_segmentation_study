"""Stage 3: every table and figure of the paper, computed from the saved out-of-fold outputs.

This stage needs no GPU and no PyTorch; it reads outputs/cache, outputs/segmentation and
outputs/classification and writes outputs/results/{results.json, RESULTS.md, figures/*.png}.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from . import stats as st
from .config import Config
from .data import keyword_flags
from .seg_metrics import seg_scores
from .utils import load_json, log, save_json

# Headline values reported in the paper, shown next to this run for comparison.
PAPER = {
    "unet_dsc": 0.976, "unet_iou": 0.954, "unet_hd95": 3.33, "otsu_dsc": 0.607,
    "area_bias": -0.48, "auc_whole": 0.946, "auc_roi": 0.882, "delta_auc": -0.064, "delong_p": 0.017,
    "spec_at_90_whole": 0.838, "spec_at_90_roi": 0.488, "tests_1000_whole": 199, "tests_1000_roi": 531,
    "spearman_rho": 0.36, "brier_whole": 0.098, "brier_roi": 0.133,
}

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
LABELS = {"whole": "Whole image", "roi": "U-Net lung ROI"}
SHORT = {"whole": "whole image", "roi": "lung ROI"}
COLORS = {"whole": BLUE, "roi": ORANGE}


def _ms(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)), float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0


# =========================================================================== computations
def cohort_table(records, cfg: Config) -> dict:
    lab = np.array([r.label for r in records])
    age = np.array([r.age for r in records])
    sex = np.array([r.sex for r in records])
    land = np.array([r.landscape for r in records])
    read = [r.reading for r in records]
    pleural = keyword_flags(read, cfg.eval.pleural_keywords)
    inactive = keyword_flags(read, cfg.eval.inactive_keywords)
    out = {}
    for name, sel in (("normal", lab == 0), ("tb", lab == 1), ("all", np.ones_like(lab, bool))):
        a = age[sel]
        out[name] = {"n": int(sel.sum()), "male": int((sex[sel] == "M").sum()), "female": int((sex[sel] == "F").sum()),
                     "sex_missing": int((sex[sel] == "").sum()),
                     "age_mean": float(np.nanmean(a)) if np.isfinite(a).any() else np.nan,
                     "age_sd": float(np.nanstd(a, ddof=1)) if np.isfinite(a).sum() > 1 else np.nan,
                     "age_min": float(np.nanmin(a)) if np.isfinite(a).any() else np.nan,
                     "age_max": float(np.nanmax(a)) if np.isfinite(a).any() else np.nan,
                     "landscape": int(land[sel].sum())}
    out["tb"]["pleural_readings"] = int((pleural & (lab == 1)).sum())
    out["tb"]["inactive_readings"] = int((inactive & (lab == 1)).sum())
    a0, a1 = age[(lab == 0) & np.isfinite(age)], age[(lab == 1) & np.isfinite(age)]
    out["age_mannwhitney_p"] = float(stats.mannwhitneyu(a0, a1).pvalue) if len(a0) and len(a1) else np.nan
    known = sex != ""
    tbl = [[int(((sex == "M") & (lab == g) & known).sum()), int(((sex == "F") & (lab == g) & known).sum())] for g in (0, 1)]
    out["sex_fisher_p"] = float(stats.fisher_exact(tbl).pvalue) if np.array(tbl).sum() else np.nan
    return out


def segmentation_results(masks_ref, unet, otsu, labels, folds) -> dict:
    per = {"unet": [seg_scores(p, g) for p, g in zip(unet, masks_ref)],
           "otsu": [seg_scores(p, g) for p, g in zip(otsu, masks_ref)]}
    keys = ["dsc", "iou", "sensitivity", "precision", "hd95"]
    table = {}
    for method in ("otsu", "unet"):
        subsets = {"all": np.ones_like(labels, bool)}
        if method == "unet":
            subsets.update({"normal": labels == 0, "tb": labels == 1})
        for sub, sel in subsets.items():
            rows = [per[method][i] for i in np.where(sel)[0]]
            table[f"{method}_{sub}"] = {"n": len(rows), **{k: _ms([r[k] for r in rows]) for k in keys}}
    d_unet = np.array([r["dsc"] for r in per["unet"]])
    d_otsu = np.array([r["dsc"] for r in per["otsu"]])
    a_ref = np.array([r["area_ref_pct"] for r in per["unet"]])
    a_pred = np.array([r["area_pred_pct"] for r in per["unet"]])
    tests = {
        "wilcoxon_unet_vs_otsu_p": float(stats.wilcoxon(d_unet, d_otsu).pvalue) if np.any(d_unet != d_otsu) else np.nan,
        "unet_better_on_n": int((d_unet > d_otsu).sum()),
        "mannwhitney_normal_vs_tb_p": float(stats.mannwhitneyu(d_unet[labels == 0], d_unet[labels == 1]).pvalue),
    }
    area = {"pearson_r": float(stats.pearsonr(a_ref, a_pred)[0]), "ref_mean": float(a_ref.mean()),
            "pred_mean": float(a_pred.mean()), **st.bland_altman(a_ref, a_pred),
            "bias_tb": float((a_pred - a_ref)[labels == 1].mean()),
            "bias_normal": float((a_pred - a_ref)[labels == 0].mean())}
    folds_out = []
    for k in np.unique(folds):
        sel = folds == k
        folds_out.append({"fold": int(k) + 1, "dsc": _ms(d_unet[sel]),
                          "hd95": float(np.mean([per["unet"][i]["hd95"] for i in np.where(sel)[0]]))})
    return {"table": table, "tests": tests, "area": area, "folds": folds_out,
            "per_image_dsc_unet": d_unet, "per_image_dsc_otsu": d_otsu,
            "area_ref": a_ref, "area_pred": a_pred}


def classification_results(oof, cfg: Config) -> dict:
    y, pw, pr = oof["label"], oof["p_whole"], oof["p_roi"]
    E = cfg.eval
    boot = lambda p, fn: st.bootstrap_ci(y, p, fn, E.n_bootstrap, E.bootstrap_seed)
    metric_fns = {
        "auc": lambda yy, pp: roc_auc_score(yy, pp),
        "accuracy": lambda yy, pp: st.threshold_metrics(yy, pp)["accuracy"],
        "sensitivity": lambda yy, pp: st.threshold_metrics(yy, pp)["sensitivity"],
        "specificity": lambda yy, pp: st.threshold_metrics(yy, pp)["specificity"],
        "precision": lambda yy, pp: st.threshold_metrics(yy, pp)["precision"],
        "f1": lambda yy, pp: st.threshold_metrics(yy, pp)["f1"],
    }
    models = {}
    for name, p in (("whole", pw), ("roi", pr)):
        m = st.all_metrics(y, p)
        m["ci"] = {k: boot(p, fn) for k, fn in metric_fns.items()}
        s90 = st.spec_at_sens(y, p, E.target_sensitivity)
        s95 = st.sens_at_spec(y, p, E.high_specificity)
        m["operating_points"] = {
            "youden": st.youden_point(y, p),
            "spec_at_sens90": {**s90, "ci": boot(p, lambda yy, pp: st.spec_at_sens(yy, pp, E.target_sensitivity)["specificity"])},
            "sens_at_spec95": {**s95, "ci": boot(p, lambda yy, pp: st.sens_at_spec(yy, pp, E.high_specificity)["sensitivity"])},
        }
        m["meets_who_triage"] = bool(s90["specificity"] >= E.target_specificity_who)
        # [paper] projections use Se = 0.90 and the specificity reached at sensitivity >= 0.90
        m["test_demand"] = {str(pi): st.test_demand(E.target_sensitivity, s90["specificity"], pi) for pi in E.prevalences}
        models[name] = m
    folds = []
    for k in np.unique(oof["fold"]):
        sel = oof["fold"] == k
        ok = y[sel].min() != y[sel].max()
        folds.append({"fold": int(k), "n": int(sel.sum()), "n_tb": int(y[sel].sum()),
                      "auc_whole": float(roc_auc_score(y[sel], pw[sel])) if ok else np.nan,
                      "auc_roi": float(roc_auc_score(y[sel], pr[sel])) if ok else np.nan,
                      "acc_whole": st.threshold_metrics(y[sel], pw[sel])["accuracy"],
                      "acc_roi": st.threshold_metrics(y[sel], pr[sel])["accuracy"]})
    fa_w = np.array([f["auc_whole"] for f in folds]); fa_r = np.array([f["auc_roi"] for f in folds])
    valid = np.isfinite(fa_w) & np.isfinite(fa_r)
    fold_wilcoxon = float(stats.wilcoxon(fa_w[valid], fa_r[valid]).pvalue) if valid.sum() >= 2 and np.any(fa_w[valid] != fa_r[valid]) else np.nan
    return {"models": models, "delong": st.delong_test(y, pw, pr), "mcnemar": st.mcnemar_exact(y, pw, pr),
            "folds": folds, "fold_auc_wilcoxon_p": fold_wilcoxon,
            "fold_auc_mean_sd": {"whole": _ms(fa_w), "roi": _ms(fa_r)}}


def mechanism_results(oof, dsc_unet, records, cfg: Config) -> dict:
    y = oof["label"]
    tb = y == 1
    dp = oof["p_whole"] - oof["p_roi"]
    rho, p = stats.spearmanr(dp[tb], dsc_unet[tb])
    readings = [r.reading for r in records]
    pleural = keyword_flags(readings, cfg.eval.pleural_keywords)
    inactive = keyword_flags(readings, cfg.eval.inactive_keywords)
    groups = {"all TB": tb, "pleural disease": tb & pleural, "no pleural disease": tb & ~pleural,
              "inactive / scarred / calcified": tb & inactive}
    sub = {}
    for name, sel in groups.items():
        n = int(sel.sum())
        entry = {"n": n}
        for model, key in (("whole", "p_whole"), ("roi", "p_roi")):
            k = int((oof[key][sel] >= 0.5).sum())
            entry[model] = {"detected": k, "sensitivity": k / n if n else np.nan, "wilson": st.wilson_ci(k, n)}
        sub[name] = entry
    return {"spearman_rho": float(rho), "spearman_p": float(p), "delta_p": dp, "subgroups": sub}


# =========================================================================== figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "lines.linewidth": 2,
    })
    return plt


def figures(cfg: Config, images, masks_ref, unet, labels, seg, cls, mech, oof):
    plt = _style()
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    y = oof["label"]

    # --- Fig A: segmentation examples (best normal, best TB, median, worst)
    d = seg["per_image_dsc_unet"]
    order = np.argsort(d)
    picks = [("Best normal", np.where(labels == 0)[0][np.argmax(d[labels == 0])]),
             ("Best TB", np.where(labels == 1)[0][np.argmax(d[labels == 1])]),
             ("Median", order[len(order) // 2]), ("Worst", order[0])]
    fig, ax = plt.subplots(2, 4, figsize=(10, 5.6), gridspec_kw={"hspace": 0.12, "wspace": 0.05})
    for j, (title, i) in enumerate(picks):
        ax[0, j].imshow(images[i], cmap="gray"); ax[0, j].set_title(f"{title}\nDSC {d[i]:.3f}")
        ax[1, j].imshow(images[i], cmap="gray")
        ax[1, j].contour(masks_ref[i], levels=[0.5], colors=["#1baf7a"], linewidths=1.2)
        ax[1, j].contour(unet[i], levels=[0.5], colors=[ORANGE], linewidths=1.2)
        for a in ax[:, j]:
            a.axis("off")
    fig.text(0.5, 0.06, "Bottom row: reference contour (green) and U-Net contour (orange)", ha="center", color=MUTED)
    fig.savefig(fig_dir / "fig_segmentation_examples.png"); plt.close(fig)

    # --- Fig B: U-Net learning curve (last fold) + DSC distributions
    curve_files = sorted(cfg.seg_dir.glob("curve_fold*.json"))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    if curve_files:
        c = load_json(curve_files[-1])
        fold_name = curve_files[-1].stem.replace("curve_fold", "fold ")
        ax[0].plot(c["epoch"], c["train_loss"], color=GREY)
        ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Training loss (BCE + Dice)")
        ax[0].set_title(f"(a) U-Net training loss, {fold_name}")
        ax[1].plot(c["epoch"], c["heldout_dsc"], color=BLUE)
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Held-out DSC")
        ax[1].set_title(f"(b) Held-out DSC, {fold_name} (monitoring only)")
    groups = [seg["per_image_dsc_otsu"][labels == 0], seg["per_image_dsc_otsu"][labels == 1],
              d[labels == 0], d[labels == 1]]
    bp = ax[2].boxplot(groups, widths=0.5, patch_artist=True, medianprops={"color": INK})
    for patch, col in zip(bp["boxes"], [GREY, GREY, BLUE, BLUE]):
        patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
    ax[2].set_xticks([1, 2, 3, 4], ["Otsu\nnormal", "Otsu\nTB", "U-Net\nnormal", "U-Net\nTB"])
    ax[2].set_ylabel("Per-image DSC"); ax[2].set_title("(c) Overlap by method and class")
    fig.savefig(fig_dir / "fig_segmentation_curves.png"); plt.close(fig)

    # --- Fig C: area agreement, Bland-Altman, delta-p vs DSC
    ar, ap, A = seg["area_ref"], seg["area_pred"], seg["area"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"wspace": 0.35})
    for lab, col, name in ((0, BLUE, "Normal"), (1, ORANGE, "TB")):
        ax[0].scatter(ar[labels == lab], ap[labels == lab], s=18, color=col, label=name, edgecolor="white", linewidth=0.5)
    lim = [min(ar.min(), ap.min()) - 1, max(ar.max(), ap.max()) + 1]
    ax[0].plot(lim, lim, color=GREY, lw=1, ls="--")
    ax[0].set_xlabel("Reference lung area (% of image)"); ax[0].set_ylabel("U-Net lung area (%)")
    ax[0].set_title(f"(a) Area agreement, r = {A['pearson_r']:.3f}"); ax[0].legend()
    mean_a, diff = (ar + ap) / 2, ap - ar
    ax[1].scatter(mean_a, diff, s=18, color=GREY, edgecolor="white", linewidth=0.5)
    ax[1].axhline(A["bias"], color=INK, lw=1.2)
    for v in (A["loa_low"], A["loa_high"]):
        ax[1].axhline(v, color=INK, lw=1, ls="--")
    ax[1].set_xlabel("Mean area (%)"); ax[1].set_ylabel("U-Net − reference (pp)")
    ax[1].set_title(f"(b) Bland–Altman, bias {A['bias']:.2f} pp")
    tb = y == 1
    ax[2].scatter(d[tb], mech["delta_p"][tb], s=20, color=ORANGE, edgecolor="white", linewidth=0.5)
    ax[2].axhline(0, color=GREY, lw=1)
    ax[2].set_xlabel("U-Net DSC (TB images)"); ax[2].set_ylabel("Δp = p(whole) − p(ROI)")
    ax[2].set_title(f"(c) Score drop vs mask quality, ρ = {mech['spearman_rho']:.2f}")
    fig.savefig(fig_dir / "fig_area_and_mask_quality.png"); plt.close(fig)

    # --- Fig D: ROC curves + classifier training loss
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, key in (("whole", "p_whole"), ("roi", "p_roi")):
        fpr, tpr, _ = roc_curve(y, oof[key])
        m = cls["models"][name]
        ax[0].plot(fpr, tpr, color=COLORS[name], label=f"{LABELS[name]} (AUC {m['auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], color=GREY, lw=1, ls=":")
    ax[0].set_xlabel("1 − specificity"); ax[0].set_ylabel("Sensitivity")
    ax[0].set_title(f"(a) Out-of-fold ROC, DeLong p = {cls['delong']['p']:.3f}"); ax[0].legend(loc="lower right")
    tc_file = cfg.cls_dir / "training_curves.json"
    if tc_file.exists():
        tc = load_json(tc_file)["train_loss"]
        for name in ("whole", "roi"):
            arr = np.array(tc[name]); ep = np.arange(1, arr.shape[1] + 1)
            mu, sd = arr.mean(0), arr.std(0)
            ax[1].plot(ep, mu, color=COLORS[name], label=LABELS[name])
            ax[1].fill_between(ep, mu - sd, mu + sd, color=COLORS[name], alpha=0.15, linewidth=0)
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Training loss (mean ± SD over folds)")
        ax[1].set_title("(b) Classifier training loss"); ax[1].legend()
        from matplotlib.ticker import MaxNLocator
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(fig_dir / "fig_roc.png"); plt.close(fig)

    # --- Fig E: score distributions + sensitivity/specificity vs threshold
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    rng = np.random.default_rng(0)
    for j, (name, key) in enumerate((("whole", "p_whole"), ("roi", "p_roi"))):
        for lab, off in ((0, -0.18), (1, 0.18)):
            v = oof[key][y == lab]
            ax[0].scatter(np.full(len(v), j + off) + rng.uniform(-0.06, 0.06, len(v)), v, s=12,
                          color=COLORS[name], alpha=0.75 if lab else 0.35, edgecolor="none")
            ax[0].boxplot([v], positions=[j + off], widths=0.12, showfliers=False,
                          medianprops={"color": INK}, boxprops={"color": INK}, whiskerprops={"color": INK}, capprops={"color": INK})
    ax[0].set_xticks([-0.18, 0.18, 0.82, 1.18], ["Whole\nnormal", "Whole\nTB", "ROI\nnormal", "ROI\nTB"])
    ax[0].set_ylabel("Predicted probability of TB"); ax[0].set_title("(a) Out-of-fold scores")
    thr = np.linspace(0, 1, 201)
    for name, key, ls in (("whole", "p_whole", "--"), ("roi", "p_roi", "-")):
        se = [((oof[key] >= t) & (y == 1)).sum() / (y == 1).sum() for t in thr]
        sp = [((oof[key] < t) & (y == 0)).sum() / (y == 0).sum() for t in thr]
        ax[1].plot(thr, se, color=ORANGE, ls=ls, lw=1.6, label=f"Sensitivity, {SHORT[name]}")
        ax[1].plot(thr, sp, color=BLUE, ls=ls, lw=1.6, label=f"Specificity, {SHORT[name]}")
    ax[1].axhline(0.9, color=GREY, ls=":", lw=1)
    ax[1].set_xlabel("Decision threshold"); ax[1].set_ylabel("Rate"); ax[1].set_title("(b) Sensitivity / specificity vs threshold")
    ax[1].legend(fontsize=7, loc="lower center")
    fig.savefig(fig_dir / "fig_scores_thresholds.png"); plt.close(fig)

    # --- Fig F: confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for j, name in enumerate(("whole", "roi")):
        m = cls["models"][name]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        rown = cm / cm.sum(1, keepdims=True)
        ax[j].imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax[j].text(c, r, f"{cm[r, c]}\n({100 * rown[r, c]:.0f}%)", ha="center", va="center",
                           color="white" if rown[r, c] > 0.6 else INK)
        ax[j].set_xticks([0, 1], ["Pred. normal", "Pred. TB"]); ax[j].set_yticks([0, 1], ["Normal", "TB"])
        ax[j].set_title(f"({'ab'[j]}) {LABELS[name]}"); ax[j].grid(False)
    fig.savefig(fig_dir / "fig_confusion.png"); plt.close(fig)

    # --- Fig G: fold-wise AUC + subgroup sensitivity
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"wspace": 0.75})
    f = cls["folds"]; ks = [x["fold"] for x in f]
    ax[0].plot(ks, [x["auc_whole"] for x in f], "s-", color=BLUE, label=LABELS["whole"], markersize=7)
    ax[0].plot(ks, [x["auc_roi"] for x in f], "o-", color=ORANGE, label=LABELS["roi"], markersize=7)
    ax[0].set_xticks(ks); ax[0].set_xlabel("Fold"); ax[0].set_ylabel("AUC"); ax[0].set_title("(a) Fold-wise AUC"); ax[0].legend()
    sub = mech["subgroups"]; names = list(sub)
    ypos = np.arange(len(names))
    for dy, model in ((-0.15, "whole"), (0.15, "roi")):
        s = np.array([sub[n][model]["sensitivity"] for n in names])
        lo = np.array([sub[n][model]["wilson"][0] for n in names]); hi = np.array([sub[n][model]["wilson"][1] for n in names])
        # clip: floating-point rounding can put s a hair outside its own interval (e.g. 1.0 vs 0.9999999999999999)
        err = np.nan_to_num(np.clip(np.vstack([s - lo, hi - s]), 0, None))
        ax[1].errorbar(s, ypos + dy, xerr=err, fmt="s" if model == "whole" else "o",
                       color=COLORS[model], label=LABELS[model], markersize=6, capsize=2)
    ax[1].set_yticks(ypos, [f"{n} (n={sub[n]['n']})" for n in names]); ax[1].invert_yaxis()
    ax[1].axvline(0.9, color=GREY, ls=":", lw=1)
    ax[1].set_xlabel("Sensitivity at 0.5 (Wilson 95% CI)"); ax[1].set_title("(b) Sensitivity by lesion subgroup")
    ax[1].set_xlim(0, 1.02); ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.savefig(fig_dir / "fig_folds_subgroups.png"); plt.close(fig)


def gradcam_figure(cfg: Config, images_u8, labels, folds, oof, n_cases: int = 4):
    """Grad-CAM for the TB images with the largest drop in probability (whole -> ROI), using the fold
    models for which each image was in the test set [paper]. Requires PyTorch and the saved models."""
    import torch

    from .gradcam import GradCAM
    from .models import resnet18_binary
    from .train_cls import build_inputs, to_tensor
    from .outputs import load_roi_masks
    from .utils import get_device

    plt = _style()
    device = get_device(cfg.device)
    tb = np.where(labels == 1)[0]
    dp = oof["p_whole"][tb] - oof["p_roi"][tb]
    cases = tb[np.argsort(dp)[::-1][:n_cases]]
    fig, ax = plt.subplots(2, len(cases), figsize=(2.6 * len(cases), 5.6), squeeze=False)
    for j, i in enumerate(cases):
        k = int(folds[i])
        roi_masks = load_roi_masks(cfg, k, images_u8.shape)
        for r, cond in enumerate(("whole", "roi")):
            model = resnet18_binary(False)
            model.load_state_dict(torch.load(cfg.cls_dir / f"resnet18_{cond}_fold{k + 1}.pt", map_location="cpu"))
            model.to(device)
            inp = build_inputs(images_u8[i:i + 1], cond, roi_masks[i:i + 1], cfg.cls.input_size)[0]
            cam_obj = GradCAM(model, model.layer4)
            cam, prob = cam_obj(to_tensor(inp)[None].to(device))
            cam_obj.close()
            ax[r, j].imshow(inp, cmap="gray"); ax[r, j].imshow(cam, cmap="jet", alpha=0.35)
            ax[r, j].set_title(f"{LABELS[cond]}\np(TB) = {prob:.2f}", fontsize=8); ax[r, j].axis("off")
    fig.suptitle("Grad-CAM, TB images with the largest score drop (qualitative only)", fontsize=10)
    out = cfg.results_dir / "figures" / "fig_gradcam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    log(f"Wrote {out}")


# =========================================================================== report
def _f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(v, ci, nd=3):
    return f"{_f(v, nd)} ({_f(ci[0], nd)}–{_f(ci[1], nd)})"


def markdown_report(cohort, seg, cls, mech, cfg: Config) -> str:
    W, R = cls["models"]["whole"], cls["models"]["roi"]
    T = seg["table"]
    L = ["# Results of this run", "",
         "All numbers are out-of-fold estimates (5-fold stratified CV, seed 42) produced by this "
         "reimplementation. The *Paper* column gives the values published in Haider et al. (2026) for comparison.", ""]
    L += ["## Headline comparison", "", "| Quantity | This run | Paper |", "|---|---|---|"]
    rows = [
        ("U-Net DSC", T["unet_all"]["dsc"][0], PAPER["unet_dsc"]),
        ("U-Net HD95 (px)", T["unet_all"]["hd95"][0], PAPER["unet_hd95"]),
        ("Otsu baseline DSC", T["otsu_all"]["dsc"][0], PAPER["otsu_dsc"]),
        ("Lung-area bias (pp)", seg["area"]["bias"], PAPER["area_bias"]),
        ("AUC, whole image", W["auc"], PAPER["auc_whole"]),
        ("AUC, lung ROI", R["auc"], PAPER["auc_roi"]),
        ("ΔAUC (ROI − whole)", cls["delong"]["delta_auc"], PAPER["delta_auc"]),
        ("DeLong p", cls["delong"]["p"], PAPER["delong_p"]),
        ("Specificity at ≥90% sensitivity, whole", W["operating_points"]["spec_at_sens90"]["specificity"], PAPER["spec_at_90_whole"]),
        ("Specificity at ≥90% sensitivity, ROI", R["operating_points"]["spec_at_sens90"]["specificity"], PAPER["spec_at_90_roi"]),
        ("Brier score, whole", W["brier"], PAPER["brier_whole"]),
        ("Brier score, ROI", R["brier"], PAPER["brier_roi"]),
        ("Spearman ρ (Δp vs DSC, TB)", mech["spearman_rho"], PAPER["spearman_rho"]),
    ]
    L += [f"| {a} | {_f(b)} | {_f(c)} |" for a, b, c in rows] + [""]

    c = cohort
    L += ["## Cohort (paper Table 3)", "", "| | Normal | TB | All |", "|---|---|---|---|",
          f"| Radiographs | {c['normal']['n']} | {c['tb']['n']} | {c['all']['n']} |",
          "| Male / female / not recorded | " + " | ".join(f"{c[g]['male']} / {c[g]['female']} / {c[g]['sex_missing']}" for g in ("normal", "tb", "all")) + " |",
          "| Age, mean ± SD | " + " | ".join(f"{_f(c[g]['age_mean'], 1)} ± {_f(c[g]['age_sd'], 1)}" for g in ("normal", "tb", "all")) + " |",
          "| Landscape acquisitions | " + " | ".join(str(c[g]["landscape"]) for g in ("normal", "tb", "all")) + " |",
          f"| Readings mentioning pleural disease | – | {c['tb']['pleural_readings']} | – |",
          f"| Readings describing inactive / scarred / calcified disease | – | {c['tb']['inactive_readings']} | – |", "",
          f"Age Mann–Whitney p = {_f(c['age_mannwhitney_p'], 4)}; sex Fisher p = {_f(c['sex_fisher_p'], 4)}.", ""]

    L += ["## Lung segmentation (paper Table 6)", "", "| Method / subset | n | DSC | IoU | Sensitivity | Precision | HD95 (px) |",
          "|---|---|---|---|---|---|---|"]
    for key, name in (("otsu_all", "Otsu + CLAHE baseline"), ("unet_all", "U-Net, all"), ("unet_normal", "U-Net, normal"), ("unet_tb", "U-Net, TB")):
        r = T[key]
        L.append(f"| {name} | {r['n']} | " + " | ".join(f"{_f(r[m][0])} ± {_f(r[m][1])}" for m in ("dsc", "iou", "sensitivity", "precision")) +
                 f" | {_f(r['hd95'][0], 2)} ± {_f(r['hd95'][1], 2)} |")
    t, A = seg["tests"], seg["area"]
    L += ["", f"U-Net better than baseline on {t['unet_better_on_n']} images (Wilcoxon p = {t['wilcoxon_unet_vs_otsu_p']:.2e}); "
              f"normal vs TB DSC Mann–Whitney p = {_f(t['mannwhitney_normal_vs_tb_p'], 4)}.",
          f"Lung area: r = {_f(A['pearson_r'])}, bias {_f(A['bias'], 2)} pp (LoA {_f(A['loa_low'], 2)} to {_f(A['loa_high'], 2)}); "
          f"bias on TB {_f(A['bias_tb'], 2)}, on normal {_f(A['bias_normal'], 2)}.", ""]

    L += ["## TB classification (paper Table 7)", "", "| Metric | Whole image | U-Net lung ROI |", "|---|---|---|"]
    for m in ("auc", "accuracy", "sensitivity", "specificity", "precision", "f1"):
        L.append(f"| {m.upper() if m == 'auc' else m.capitalize()} | {_ci(W[m], W['ci'][m])} | {_ci(R[m], R['ci'][m])} |")
    L.append(f"| Brier score | {_f(W['brier'])} | {_f(R['brier'])} |")
    dl, mc = cls["delong"], cls["mcnemar"]
    L += ["", f"ΔAUC (ROI − whole) = {dl['delta_auc']:+.3f}, DeLong z = {dl['z']:.2f}, p = {dl['p']:.4f}. "
              f"McNemar at 0.5: {mc['model1_only_correct']} vs {mc['model2_only_correct']} discordant, exact p = {_f(mc['p'])}.", ""]

    L += ["## Operating points and test demand (paper Table 8)", "", "| Operating point | Whole image | U-Net lung ROI |", "|---|---|---|"]
    yw, yr = W["operating_points"]["youden"], R["operating_points"]["youden"]
    sw, sr = W["operating_points"]["spec_at_sens90"], R["operating_points"]["spec_at_sens90"]
    hw, hr = W["operating_points"]["sens_at_spec95"], R["operating_points"]["sens_at_spec95"]
    L += [f"| Youden-optimal threshold | {_f(yw['threshold'])} | {_f(yr['threshold'])} |",
          f"| Sensitivity / specificity at Youden point | {_f(yw['sensitivity'])} / {_f(yw['specificity'])} | {_f(yr['sensitivity'])} / {_f(yr['specificity'])} |",
          f"| Specificity at sensitivity ≥ 0.90 | {_ci(sw['specificity'], sw['ci'])} | {_ci(sr['specificity'], sr['ci'])} |",
          f"| Sensitivity at specificity ≥ 0.95 | {_ci(hw['sensitivity'], hw['ci'])} | {_ci(hr['sensitivity'], hr['ci'])} |",
          f"| Meets WHO triage minimum (sens ≥ 0.90, spec ≥ 0.70) | {'Yes' if W['meets_who_triage'] else 'No'} | {'Yes' if R['meets_who_triage'] else 'No'} |"]
    for pi in cfg.eval.prevalences:
        tw, tr = W["test_demand"][str(pi)], R["test_demand"][str(pi)]
        L += [f"| Confirmatory tests per 1,000 screened, π = {pi:.0%} | {tw['tests_per_1000']:.0f} | {tr['tests_per_1000']:.0f} |",
              f"| Tests per TB case detected, π = {pi:.0%} | {tw['tests_per_case']:.1f} | {tr['tests_per_case']:.1f} |"]
    L.append("")

    L += ["## Fold-wise results (paper Table 9)", "", "| Fold | n (TB) | U-Net DSC | U-Net HD95 | AUC whole | AUC ROI | Accuracy whole / ROI |",
          "|---|---|---|---|---|---|---|"]
    for fs, fc in zip(seg["folds"], cls["folds"]):
        L.append(f"| {fc['fold']} | {fc['n']} ({fc['n_tb']}) | {_f(fs['dsc'][0])} ± {_f(fs['dsc'][1])} | {_f(fs['hd95'], 2)} | "
                 f"{_f(fc['auc_whole'])} | {_f(fc['auc_roi'])} | {_f(fc['acc_whole'])} / {_f(fc['acc_roi'])} |")
    fm = cls["fold_auc_mean_sd"]
    L += ["", f"Mean fold AUC: whole {_f(fm['whole'][0])} ± {_f(fm['whole'][1])}, ROI {_f(fm['roi'][0])} ± {_f(fm['roi'][1])}; "
              f"Wilcoxon on paired fold AUCs p = {_f(cls['fold_auc_wilcoxon_p'])}.", ""]

    L += ["## Is the loss explained by mask errors? (paper Section 'Mechanism')", "",
          f"Spearman ρ between Δp = p(whole) − p(ROI) and U-Net DSC on TB images: {mech['spearman_rho']:.2f} (p = {mech['spearman_p']:.4f}).", "",
          "| TB subgroup | n | Detected, whole (Wilson CI) | Detected, ROI (Wilson CI) |", "|---|---|---|---|"]
    for name, e in mech["subgroups"].items():
        cell = lambda m: f"{e[m]['detected']}/{e['n']} = {_f(e[m]['sensitivity'])} ({_f(e[m]['wilson'][0])}–{_f(e[m]['wilson'][1])})"
        L.append(f"| {name} | {e['n']} | {cell('whole')} | {cell('roi')} |")
    L += ["", "Subgroups are keyword-based and overlapping; treat them as hypothesis-generating.", ""]
    return "\n".join(L)


def run(cfg: Config, images, masks_ref, records, folds, with_gradcam: bool = True):
    from .outputs import load_oof, load_oof_masks

    labels = np.array([r.label for r in records])
    unet, otsu = load_oof_masks(cfg)
    oof = load_oof(cfg)
    assert list(oof["image_id"]) == [r.image_id for r in records], "prediction file does not match the cache order"

    cohort = cohort_table(records, cfg)
    seg = segmentation_results(masks_ref, unet, otsu, labels, folds)
    cls = classification_results(oof, cfg)
    mech = mechanism_results(oof, seg["per_image_dsc_unet"], records, cfg)

    figures(cfg, images, masks_ref, unet, labels, seg, cls, mech, oof)
    if with_gradcam:
        try:
            gradcam_figure(cfg, images, labels, folds, oof)
        except Exception as e:  # models missing or torch unavailable
            log(f"Grad-CAM skipped: {e}")

    report = markdown_report(cohort, seg, cls, mech, cfg)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    strip = lambda d: {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
    save_json({"config": cfg.to_dict(), "cohort": cohort, "segmentation": strip(seg), "classification": cls,
               "mechanism": strip(mech), "paper": PAPER}, cfg.results_dir / "results.json")
    log(f"Wrote {cfg.results_dir / 'RESULTS.md'} and figures in {cfg.results_dir / 'figures'}")
    return report
