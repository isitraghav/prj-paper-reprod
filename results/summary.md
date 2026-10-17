# P2 reproduction summary (ELDDS1400C5 test split, 190 images, 640 px)

Machine: Laptop B (RTX 3050 Laptop 6 GB, Win 11). Batch: 16 x 4 accum (paper 32 x 2). Settings and deviations: `init.md` §3, §11.

| Model | P (paper / ours) | R (paper / ours) | mAP@.5 (paper T1 / ours / Δ) | mAP@.5:.95 (paper / ours / Δ) | mAP@.5 no TTA (paper T2 / ours / Δ) | mAP@.5 TTA @640 (paper T2 / ours / Δ) | TTA @832 (sensitivity) |
|---|---|---|---|---|---|---|---|
| YOLOv5s_only | 0.726 / 0.748 | 0.723 / 0.707 | 0.738 / 0.754 / +0.016 | 0.502 / 0.516 / +0.014 | 0.739 / 0.754 / +0.015 | 0.766 / 0.766 / +0.000 | 0.774 |
| YOLOv5s+GAM+ASFF+DIoU | 0.735 / 0.754 | 0.760 / 0.727 | 0.763 / 0.757 / -0.006 | 0.516 / 0.514 / -0.002 | 0.762 / 0.757 / -0.005 | 0.777 / 0.758 / -0.019 | 0.773 |
| YOLOv8s | — / 0.764 | — / 0.746 | — / 0.764 / — | — / 0.550 / — | 0.765 / 0.764 / -0.001 | 0.775 / 0.753 / -0.022 | 0.761 |
