# Self-touch FCN: 14-model analysis

## Scope and method

This report compares the newest completed 500-epoch run for each requested FCN input variant. Ranking is by the minimum aggregate validation raw-taxel MAE. Accuracy is the exported raw closeness score and is strongly related to MAE; correlation and R² provide shape/variance checks.

Overfitting risk is a validation-only diagnostic: **high** means final degradation or the normalized late rising trend exceeds 5%; **watch** means it exceeds 2%; **low** means neither threshold is crossed. This can detect late regression but cannot prove train/test overfitting without a matching train-metric series.

## Executive findings

- Best overall: **selftouch_fcn_postrq** at epoch 475 (MAE 381.90, accuracy 93.31%).
- Fastest to within 5% of its own best: **selftouch_fcn_posvel** at epoch 100.
- Largest improvement from first evaluation: **selftouch_fcn_posvel** (39.50%).
- Most stable late MAE: **selftouch_fcn_velcmd** (CV 0.06%).
- Best finger-specific checkpoints: index = **selftouch_fcn_posveltrq** (epoch 500, MAE 505.35), thumb = **selftouch_fcn_postrqcmd** (epoch 275, MAE 472.06), middle = **selftouch_fcn_postrq** (epoch 500, MAE 323.27), ring = **selftouch_fcn_postrq** (epoch 125, MAE 199.01).
- Validation regression flags: none.
- Multi-input variants average MAE 395.06 versus 441.36 for single-input variants (10.49% lower), although this is descriptive rather than a seeded significance test.
- Average finger difficulty by MAE: index 547.24, thumb 489.27, middle 368.88, ring 227.76.

## Overall ranking

| Rank | Variant | Run | Best epoch | Best MAE ↓ | Accuracy ↑ | RMSE ↓ | Corr ↑ | R² ↑ | p95 ↓ | Abs bias ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | selftouch_fcn_postrq | olive-galaxy-1 | 475 | 381.90 | 93.31 | 659.83 | 0.897 | 0.808 | 1352.39 | 37.62 |
| 2 | selftouch_fcn_posveltrq | balmy-serenity-1 | 400 | 382.71 | 93.30 | 672.14 | 0.893 | 0.801 | 1370.58 | 39.59 |
| 3 | selftouch_fcn_poscmd | swift-dust-1 | 425 | 385.91 | 93.24 | 671.91 | 0.893 | 0.802 | 1362.16 | 24.15 |
| 4 | selftouch_fcn_postrqcmd | vocal-dragon-1 | 325 | 389.47 | 93.18 | 681.03 | 0.890 | 0.797 | 1394.82 | 26.19 |
| 5 | selftouch_fcn_poscmdvel | crisp-haze-1 | 375 | 389.91 | 93.17 | 676.29 | 0.891 | 0.799 | 1381.91 | 27.78 |
| 6 | selftouch_fcn_trqcmd | legendary-disco-1 | 400 | 392.46 | 93.12 | 687.18 | 0.890 | 0.796 | 1398.51 | 29.72 |
| 7 | selftouch_fcn_veltrq | fluent-plant-1 | 400 | 394.23 | 93.09 | 681.81 | 0.890 | 0.797 | 1398.96 | 34.47 |
| 8 | selftouch_fcn_posvel | solar-frost-1 | 425 | 395.05 | 93.08 | 691.19 | 0.888 | 0.793 | 1412.29 | 29.16 |
| 9 | selftouch_fcn_velcmd | fresh-galaxy-1 | 425 | 397.53 | 93.04 | 692.17 | 0.888 | 0.792 | 1406.09 | 25.51 |
| 10 | selftouch_fcn_vel | proud-snow-1 | 450 | 436.79 | 92.35 | 757.32 | 0.864 | 0.753 | 1603.24 | 22.29 |
| 11 | selftouch_fcn_pos | winter-donkey-1 | 450 | 438.99 | 92.31 | 762.67 | 0.864 | 0.751 | 1620.98 | 25.98 |
| 12 | selftouch_fcn_posveltrqcmd | rare-feather-1 | 450 | 441.38 | 92.27 | 764.38 | 0.864 | 0.751 | 1634.23 | 29.71 |
| 13 | selftouch_fcn_trq | swift-forest-1 | 450 | 441.91 | 92.26 | 763.24 | 0.863 | 0.750 | 1620.10 | 23.71 |
| 14 | selftouch_fcn_cmd | serene-snow-1 | 450 | 447.77 | 92.16 | 774.95 | 0.860 | 0.743 | 1664.38 | 33.40 |

## Convergence and late-regression diagnostics

| Variant | Initial MAE | Improvement | ≤10% best | ≤5% best | ≤2% best | Final MAE | Final Δ | Late trend/100 ep | Late CV | Risk |
|---|---|---|---|---|---|---|---|---|---|---|
| selftouch_fcn_postrq | 596.22 | 35.95% | 75 | 125 | 250 | 382.11 | 0.06% | -0.28% | 0.12% | low |
| selftouch_fcn_posveltrq | 600.19 | 36.24% | 100 | 175 | 250 | 383.18 | 0.12% | 0.06% | 0.11% | low |
| selftouch_fcn_poscmd | 587.74 | 34.34% | 75 | 150 | 200 | 386.78 | 0.23% | -0.16% | 0.20% | low |
| selftouch_fcn_postrqcmd | 587.54 | 33.71% | 75 | 125 | 200 | 390.49 | 0.26% | -0.06% | 0.08% | low |
| selftouch_fcn_poscmdvel | 586.00 | 33.46% | 75 | 125 | 175 | 391.88 | 0.51% | 0.28% | 0.13% | low |
| selftouch_fcn_trqcmd | 643.92 | 39.05% | 75 | 125 | 250 | 394.36 | 0.48% | 0.26% | 0.18% | low |
| selftouch_fcn_veltrq | 616.24 | 36.03% | 75 | 125 | 225 | 394.91 | 0.17% | 0.17% | 0.09% | low |
| selftouch_fcn_posvel | 653.01 | 39.50% | 75 | 100 | 175 | 395.79 | 0.19% | -0.08% | 0.11% | low |
| selftouch_fcn_velcmd | 607.95 | 34.61% | 75 | 150 | 200 | 398.17 | 0.16% | 0.11% | 0.06% | low |
| selftouch_fcn_vel | 689.42 | 36.64% | 150 | 300 | 350 | 438.44 | 0.38% | -1.06% | 0.80% | low |
| selftouch_fcn_pos | 685.94 | 36.00% | 150 | 250 | 300 | 439.02 | 0.01% | -1.01% | 0.75% | low |
| selftouch_fcn_posveltrqcmd | 693.79 | 36.38% | 150 | 250 | 300 | 442.12 | 0.17% | -0.88% | 0.68% | low |
| selftouch_fcn_trq | 689.97 | 35.95% | 150 | 200 | 300 | 441.92 | 0.00% | -1.13% | 0.89% | low |
| selftouch_fcn_cmd | 698.81 | 35.92% | 150 | 250 | 300 | 448.56 | 0.18% | -0.84% | 0.63% | low |

## Recommended checkpoint files

| Variant | Best evaluated epoch | Exact checkpoint? | Recommended saved epoch | Checkpoint |
|---|---|---|---|---|
| selftouch_fcn_postrq | 475 | yes | 475 | `model_weight/selftouch_fcn_postrq/olive-galaxy-1/epoch474.pth` |
| selftouch_fcn_posveltrq | 400 | yes | 400 | `model_weight/selftouch_fcn_posveltrq/balmy-serenity-1/epoch399.pth` |
| selftouch_fcn_poscmd | 425 | yes | 425 | `model_weight/selftouch_fcn_poscmd/swift-dust-1/epoch424.pth` |
| selftouch_fcn_postrqcmd | 325 | yes | 325 | `model_weight/selftouch_fcn_postrqcmd/vocal-dragon-1/epoch324.pth` |
| selftouch_fcn_poscmdvel | 375 | yes | 375 | `model_weight/selftouch_fcn_poscmdvel/crisp-haze-1/epoch374.pth` |
| selftouch_fcn_trqcmd | 400 | yes | 400 | `model_weight/selftouch_fcn_trqcmd/legendary-disco-1/epoch399.pth` |
| selftouch_fcn_veltrq | 400 | yes | 400 | `model_weight/selftouch_fcn_veltrq/fluent-plant-1/epoch399.pth` |
| selftouch_fcn_posvel | 425 | yes | 425 | `model_weight/selftouch_fcn_posvel/solar-frost-1/epoch424.pth` |
| selftouch_fcn_velcmd | 425 | yes | 425 | `model_weight/selftouch_fcn_velcmd/fresh-galaxy-1/epoch424.pth` |
| selftouch_fcn_vel | 450 | no | 500 | `model_weight/selftouch_fcn_vel/proud-snow-1/epoch499.pth` |
| selftouch_fcn_pos | 450 | no | 500 | `model_weight/selftouch_fcn_pos/winter-donkey-1/epoch499.pth` |
| selftouch_fcn_posveltrqcmd | 450 | no | 500 | `model_weight/selftouch_fcn_posveltrqcmd/rare-feather-1/epoch499.pth` |
| selftouch_fcn_trq | 450 | no | 500 | `model_weight/selftouch_fcn_trq/swift-forest-1/epoch499.pth` |
| selftouch_fcn_cmd | 450 | no | 500 | `model_weight/selftouch_fcn_cmd/serene-snow-1/epoch499.pth` |

## Per-finger performance at each model's overall-best epoch

| Variant | Index MAE / Acc | Thumb MAE / Acc | Middle MAE / Acc | Ring MAE / Acc |
|---|---|---|---|---|
| selftouch_fcn_postrq | 506.12 / 91.13% | 475.21 / 91.67% | 323.60 / 94.33% | 222.65 / 96.10% |
| selftouch_fcn_posveltrq | 508.70 / 91.09% | 477.99 / 91.63% | 327.69 / 94.26% | 216.46 / 96.21% |
| selftouch_fcn_poscmd | 512.72 / 91.02% | 476.88 / 91.65% | 325.57 / 94.30% | 228.45 / 96.00% |
| selftouch_fcn_postrqcmd | 532.28 / 90.67% | 472.72 / 91.72% | 333.19 / 94.16% | 219.68 / 96.15% |
| selftouch_fcn_poscmdvel | 534.73 / 90.63% | 474.77 / 91.68% | 337.42 / 94.09% | 212.72 / 96.27% |
| selftouch_fcn_trqcmd | 520.46 / 90.88% | 474.29 / 91.69% | 337.64 / 94.08% | 237.45 / 95.84% |
| selftouch_fcn_veltrq | 519.32 / 90.90% | 483.80 / 91.52% | 338.83 / 94.06% | 234.99 / 95.88% |
| selftouch_fcn_posvel | 525.16 / 90.80% | 483.27 / 91.53% | 337.96 / 94.08% | 233.83 / 95.90% |
| selftouch_fcn_velcmd | 534.97 / 90.63% | 482.68 / 91.54% | 339.47 / 94.05% | 233.02 / 95.92% |
| selftouch_fcn_vel | 591.87 / 89.63% | 508.47 / 91.09% | 427.36 / 92.51% | 219.46 / 96.16% |
| selftouch_fcn_pos | 586.38 / 89.73% | 508.02 / 91.10% | 431.48 / 92.44% | 230.08 / 95.97% |
| selftouch_fcn_posveltrqcmd | 592.69 / 89.62% | 505.38 / 91.15% | 427.71 / 92.51% | 239.73 / 95.80% |
| selftouch_fcn_trq | 596.10 / 89.56% | 513.74 / 91.00% | 433.23 / 92.41% | 224.55 / 96.07% |
| selftouch_fcn_cmd | 599.86 / 89.49% | 512.52 / 91.02% | 443.19 / 92.24% | 235.53 / 95.87% |

## Output files

- `model_summary.csv`: overall/best/final/convergence diagnostics.
- `per_finger_metrics.csv`: per-finger metrics at model-best, finger-best, and final epochs.
- `mae_convergence.png` and `accuracy_convergence.png`: all histories.
- `best_vs_final_mae.png`: late degradation comparison.
- `per_finger_mae_heatmap.png` and `per_finger_accuracy_heatmap.png`: cross-model finger comparison.

## Selection guidance

Use the recommended saved checkpoint rather than assuming every best evaluation epoch has a corresponding weight file. For deployment, repeat the top candidates with multiple random seeds: these results represent one run per input variant, so small ranking differences are not evidence of a reproducible advantage.

## Compatibility caveat

These results describe the checkpoint files listed above. They were trained before the latest shared FCN backbone change that ported the `selftouch_fcn_pos` temporal-difference/depthwise architecture to the other variants. New runs from the current code are a new experiment and should not be mixed into this ranking without regenerating the report.
