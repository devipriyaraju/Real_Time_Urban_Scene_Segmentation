# Experiment A results

The Experiment A evaluation produced:

| Metric | Value |
|---|---:|
| Cityscapes validation mIoU | 66.24% |
| Pixel accuracy | 94.24% |

See `results/metrics/metrics.csv` for all 19 per-class IoUs and `results/figures/`
for the confusion matrix, per-class IoU chart, and six qualitative examples.

## Boundary-F1 caution

The exploratory notebook stored `boundary_f1 = 1.0000`. I found an issue in the notebook Boundary-F1 calculation, so I do **not** report it as a valid result. The raw historical value
is retained in `metrics.csv` for provenance only.
