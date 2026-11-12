# Results of this run

All numbers are out-of-fold estimates (5-fold stratified CV, seed 42) produced by this reimplementation. The *Paper* column gives the values published in Haider et al. (2026) for comparison.

## Headline comparison

| Quantity | This run | Paper |
|---|---|---|
| U-Net DSC | 0.975 | 0.976 |
| U-Net HD95 (px) | 3.479 | 3.330 |
| Otsu baseline DSC | 0.773 | 0.607 |
| Lung-area bias (pp) | -0.334 | -0.480 |
| AUC, whole image | 0.933 | 0.946 |
| AUC, lung ROI | 0.877 | 0.882 |
| ΔAUC (ROI − whole) | -0.056 | -0.064 |
| DeLong p | 0.061 | 0.017 |
| Specificity at ≥90% sensitivity, whole | 0.600 | 0.838 |
| Specificity at ≥90% sensitivity, ROI | 0.475 | 0.488 |
| Brier score, whole | 0.122 | 0.098 |
| Brier score, ROI | 0.138 | 0.133 |
| Spearman ρ (Δp vs DSC, TB) | 0.372 | 0.360 |

## Cohort (paper Table 3)

| | Normal | TB | All |
|---|---|---|---|
| Radiographs | 80 | 58 | 138 |
| Male / female / not recorded | 26 / 53 / 1 | 37 / 21 / 0 | 63 / 74 / 1 |
| Age, mean ± SD | 34.3 ± 15.4 | 48.1 ± 20.1 | 40.1 ± 18.8 |
| Landscape acquisitions | 29 | 12 | 41 |
| Readings mentioning pleural disease | – | 13 | – |
| Readings describing inactive / scarred / calcified disease | – | 23 | – |

Age Mann–Whitney p = 0.0002; sex Fisher p = 0.0005.

## Lung segmentation (paper Table 6)

| Method / subset | n | DSC | IoU | Sensitivity | Precision | HD95 (px) |
|---|---|---|---|---|---|---|
| Otsu + CLAHE baseline | 138 | 0.773 ± 0.238 | 0.673 ± 0.231 | 0.729 ± 0.249 | 0.842 ± 0.228 | 32.63 ± 39.27 |
| U-Net, all | 138 | 0.975 ± 0.017 | 0.952 ± 0.031 | 0.969 ± 0.026 | 0.982 ± 0.018 | 3.48 ± 4.28 |
| U-Net, normal | 80 | 0.980 ± 0.005 | 0.961 ± 0.010 | 0.975 ± 0.010 | 0.986 ± 0.009 | 2.26 ± 1.31 |
| U-Net, TB | 58 | 0.968 ± 0.024 | 0.938 ± 0.043 | 0.960 ± 0.036 | 0.976 ± 0.025 | 5.17 ± 6.06 |

U-Net better than baseline on 137 images (Wilcoxon p = 2.20e-24); normal vs TB DSC Mann–Whitney p = 0.0021.
Lung area: r = 0.992, bias -0.33 pp (LoA -1.87 to 1.20); bias on TB -0.45, on normal -0.25.

## TB classification (paper Table 7)

| Metric | Whole image | U-Net lung ROI |
|---|---|---|
| AUC | 0.933 (0.885–0.969) | 0.877 (0.811–0.934) |
| Accuracy | 0.848 (0.783–0.906) | 0.833 (0.768–0.891) |
| Sensitivity | 0.897 (0.804–0.966) | 0.741 (0.625–0.846) |
| Specificity | 0.812 (0.722–0.895) | 0.900 (0.829–0.962) |
| Precision | 0.776 (0.672–0.873) | 0.843 (0.738–0.938) |
| F1 | 0.832 (0.754–0.898) | 0.789 (0.696–0.864) |
| Brier score | 0.122 | 0.138 |

ΔAUC (ROI − whole) = -0.056, DeLong z = -1.88, p = 0.0608. McNemar at 0.5: 14 vs 12 discordant, exact p = 0.845.

## Operating points and test demand (paper Table 8)

| Operating point | Whole image | U-Net lung ROI |
|---|---|---|
| Youden-optimal threshold | 0.815 | 0.713 |
| Sensitivity / specificity at Youden point | 0.828 / 0.925 | 0.741 / 0.938 |
| Specificity at sensitivity ≥ 0.90 | 0.600 (0.474–0.952) | 0.475 (0.362–0.816) |
| Sensitivity at specificity ≥ 0.95 | 0.741 (0.567–0.902) | 0.724 (0.327–0.833) |
| Meets WHO triage minimum (sens ≥ 0.90, spec ≥ 0.70) | No | No |
| Confirmatory tests per 1,000 screened, π = 1% | 405 | 529 |
| Tests per TB case detected, π = 1% | 45.0 | 58.8 |
| Confirmatory tests per 1,000 screened, π = 5% | 425 | 544 |
| Tests per TB case detected, π = 5% | 9.4 | 12.1 |
| Confirmatory tests per 1,000 screened, π = 10% | 450 | 562 |
| Tests per TB case detected, π = 10% | 5.0 | 6.2 |

## Fold-wise results (paper Table 9)

| Fold | n (TB) | U-Net DSC | U-Net HD95 | AUC whole | AUC ROI | Accuracy whole / ROI |
|---|---|---|---|---|---|---|
| 1 | 28 (12) | 0.972 ± 0.021 | 3.96 | 0.943 | 0.927 | 0.821 / 0.786 |
| 2 | 28 (12) | 0.977 ± 0.012 | 2.77 | 0.938 | 0.885 | 0.857 / 0.821 |
| 3 | 28 (12) | 0.971 ± 0.023 | 4.48 | 0.927 | 0.906 | 0.857 / 0.821 |
| 4 | 27 (11) | 0.973 ± 0.017 | 4.03 | 0.955 | 0.903 | 0.889 / 0.852 |
| 5 | 27 (11) | 0.981 ± 0.006 | 2.13 | 0.892 | 0.864 | 0.815 / 0.889 |

Mean fold AUC: whole 0.931 ± 0.024, ROI 0.897 ± 0.024; Wilcoxon on paired fold AUCs p = 0.062.

## Is the loss explained by mask errors? (paper Section 'Mechanism')

Spearman ρ between Δp = p(whole) − p(ROI) and U-Net DSC on TB images: 0.37 (p = 0.0040).

| TB subgroup | n | Detected, whole (Wilson CI) | Detected, ROI (Wilson CI) |
|---|---|---|---|
| all TB | 58 | 52/58 = 0.897 (0.792–0.952) | 43/58 = 0.741 (0.616–0.837) |
| pleural disease | 13 | 11/13 = 0.846 (0.578–0.957) | 9/13 = 0.692 (0.424–0.873) |
| no pleural disease | 45 | 41/45 = 0.911 (0.793–0.965) | 34/45 = 0.756 (0.613–0.858) |
| inactive / scarred / calcified | 23 | 20/23 = 0.870 (0.679–0.955) | 16/23 = 0.696 (0.491–0.844) |

Subgroups are keyword-based and overlapping; treat them as hypothesis-generating.
