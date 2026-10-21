### Detection, class-agnostic (defect vs background)

| Model | Dataset | home? | P | R | mAP@.5 [95% CI] | mAP@.5:.95 |
|---|---|---|---|---|---|---|
| P1 YOLOv5s | PV-Multi-Defect (P1) | home | 0.781 | 0.827 | 0.823 [0.771, 0.868] | 0.476 |
| P1 YOLOv5s | ELDDS1400C5 (P2) | cross | 0.147 | 0.046 | 0.035 [0.024, 0.049] | 0.010 |
| P1 GBH-YOLOv5 | PV-Multi-Defect (P1) | home | 0.728 | 0.835 | 0.768 [0.715, 0.821] | 0.412 |
| P1 GBH-YOLOv5 | ELDDS1400C5 (P2) | cross | 0.213 | 0.071 | 0.053 [0.040, 0.069] | 0.012 |
| P2 YOLOv5s | PV-Multi-Defect (P1) | cross | 0.035 | 0.035 | 0.008 [0.004, 0.021] | 0.003 |
| P2 YOLOv5s | ELDDS1400C5 (P2) | home | 0.509 | 0.556 | 0.466 [0.378, 0.593] | 0.190 |
| P2 YOLOv5s+GAM+ASFF+DIoU | PV-Multi-Defect (P1) | cross | 0.014 | 0.053 | 0.004 [0.002, 0.007] | 0.001 |
| P2 YOLOv5s+GAM+ASFF+DIoU | ELDDS1400C5 (P2) | home | 0.521 | 0.549 | 0.464 [0.372, 0.590] | 0.193 |
| P2 YOLOv8s | PV-Multi-Defect (P1) | cross | 0.091 | 0.031 | 0.015 [0.007, 0.032] | 0.006 |
| P2 YOLOv8s | ELDDS1400C5 (P2) | home | 0.553 | 0.550 | 0.479 [0.377, 0.611] | 0.209 |

### Detection, class-aware on the home dataset (our evaluator; paper metric)

| Model | Dataset | P | R | mAP@.5 | mAP@.5 defect classes only | mAP@.5:.95 | AP@.5 per class |
|---|---|---|---|---|---|---|---|
| P1 YOLOv5s | PV-Multi-Defect (P1) | 0.839 | 0.825 | 0.856 | 0.856 | 0.527 | broken 0.861, hot_spot 0.900, black_border 0.936, scratch 0.649, no_electricity 0.933 |
| P1 GBH-YOLOv5 | PV-Multi-Defect (P1) | 0.761 | 0.835 | 0.823 | 0.823 | 0.521 | broken 0.797, hot_spot 0.833, black_border 0.920, scratch 0.649, no_electricity 0.914 |
| P2 YOLOv5s | ELDDS1400C5 (P2) | 0.738 | 0.715 | 0.748 | 0.698 | 0.506 | Examined 0.995, ShortCircuitString 0.931, ShortCircuitCell 0.933, Crack 0.759, MicroCrack 0.362, OtherError 0.506 |
| P2 YOLOv5s+GAM+ASFF+DIoU | ELDDS1400C5 (P2) | 0.748 | 0.730 | 0.750 | 0.701 | 0.504 | Examined 0.995, ShortCircuitString 0.917, ShortCircuitCell 0.954, Crack 0.773, MicroCrack 0.339, OtherError 0.522 |
| P2 YOLOv8s | ELDDS1400C5 (P2) | 0.744 | 0.749 | 0.762 | 0.716 | 0.545 | Examined 0.995, ShortCircuitString 0.954, ShortCircuitCell 0.965, Crack 0.793, MicroCrack 0.328, OtherError 0.540 |

### Tile classification (defective vs functional)

| Model | Dataset | home? | n tiles (defective) | AUC [95% CI] | Acc | macro-F1 [95% CI] | P_def | R_def | F1_def |
|---|---|---|---|---|---|---|---|---|---|
| P1 YOLOv5s | PV-Multi-Defect (P1) | home | 3552 (513) | 0.977 [0.969, 0.985] | 0.961 | 0.921 [0.904, 0.937] | 0.851 | 0.881 | 0.866 |
| P1 YOLOv5s | ELDDS1400C5 (P2) | cross | 6840 (1716) | 0.577 [0.547, 0.606] | 0.353 | 0.348 [0.312, 0.380] | 0.263 | 0.877 | 0.405 |
| P1 YOLOv5s | ELPV (P3) | cross | 656 (269) | 0.600 [0.555, 0.646] | 0.500 | 0.483 [0.443, 0.521] | 0.442 | 0.829 | 0.576 |
| P1 GBH-YOLOv5 | PV-Multi-Defect (P1) | home | 3552 (513) | 0.977 [0.970, 0.984] | 0.958 | 0.917 [0.903, 0.931] | 0.842 | 0.875 | 0.859 |
| P1 GBH-YOLOv5 | ELDDS1400C5 (P2) | cross | 6840 (1716) | 0.608 [0.582, 0.634] | 0.440 | 0.438 [0.410, 0.465] | 0.272 | 0.736 | 0.398 |
| P1 GBH-YOLOv5 | ELPV (P3) | cross | 656 (269) | 0.606 [0.563, 0.650] | 0.541 | 0.541 [0.502, 0.579] | 0.459 | 0.662 | 0.542 |
| P2 YOLOv5s | PV-Multi-Defect (P1) | cross | 3552 (513) | 0.479 [0.439, 0.521] | 0.588 | 0.458 [0.438, 0.477] | 0.135 | 0.341 | 0.193 |
| P2 YOLOv5s | ELDDS1400C5 (P2) | home | 6840 (1716) | 0.949 [0.938, 0.959] | 0.892 | 0.862 [0.847, 0.875] | 0.752 | 0.848 | 0.797 |
| P2 YOLOv5s | ELPV (P3) | cross | 656 (269) | 0.457 [0.412, 0.504] | 0.593 | 0.416 [0.388, 0.447] | 0.538 | 0.052 | 0.095 |
| P2 YOLOv5s+GAM+ASFF+DIoU | PV-Multi-Defect (P1) | cross | 3552 (513) | 0.499 [0.459, 0.539] | 0.620 | 0.471 [0.450, 0.490] | 0.137 | 0.308 | 0.190 |
| P2 YOLOv5s+GAM+ASFF+DIoU | ELDDS1400C5 (P2) | home | 6840 (1716) | 0.951 [0.939, 0.962] | 0.895 | 0.865 [0.851, 0.879] | 0.760 | 0.847 | 0.802 |
| P2 YOLOv5s+GAM+ASFF+DIoU | ELPV (P3) | cross | 656 (269) | 0.524 [0.479, 0.566] | 0.512 | 0.487 [0.448, 0.527] | 0.395 | 0.357 | 0.375 |
| P2 YOLOv8s | PV-Multi-Defect (P1) | cross | 3552 (513) | 0.484 [0.446, 0.530] | 0.548 | 0.442 [0.424, 0.462] | 0.134 | 0.390 | 0.199 |
| P2 YOLOv8s | ELDDS1400C5 (P2) | home | 6840 (1716) | 0.948 [0.935, 0.959] | 0.901 | 0.868 [0.853, 0.881] | 0.798 | 0.808 | 0.803 |
| P2 YOLOv8s | ELPV (P3) | cross | 656 (269) | 0.495 [0.451, 0.542] | 0.508 | 0.503 [0.465, 0.540] | 0.417 | 0.502 | 0.455 |
| P3 VGG-19 CNN | PV-Multi-Defect (P1) | cross | 3552 (513) | 0.497 [0.444, 0.554] | 0.393 | 0.363 [0.340, 0.385] | 0.137 | 0.606 | 0.224 |
| P3 VGG-19 CNN | ELDDS1400C5 (P2) | cross | 6840 (1716) | 0.591 [0.560, 0.624] | 0.383 | 0.383 [0.352, 0.412] | 0.259 | 0.781 | 0.389 |
| P3 VGG-19 CNN | ELPV (P3) | home | 656 (269) | 0.877 [0.845, 0.906] | 0.809 | 0.787 [0.754, 0.820] | 0.909 | 0.595 | 0.719 |

### ELPV with P3's confidence weights (P3 protocol)

| Model | acc_w | macro-F1_w | AUC_w |
|---|---|---|---|
| P1 YOLOv5s | 47.65 | 46.83 | 63.40 |
| P1 GBH-YOLOv5 | 53.91 | 53.77 | 63.49 |
| P2 YOLOv5s | 64.89 | 44.11 | 43.32 |
| P2 YOLOv5s+GAM+ASFF+DIoU | 51.73 | 47.36 | 51.64 |
| P2 YOLOv8s | 50.28 | 48.94 | 48.05 |
| P3 VGG-19 CNN | 86.49 | 84.20 | 90.18 |

### Inference speed on this machine (RTX 4050 Laptop, batch 1 for detectors, fp16)

| Model | input px | ms/img PV | ms/img ELDDS | ms/img ELPV | NMS ms/img (ELDDS) |
|---|---|---|---|---|---|
| P1 YOLOv5s | 960 | 17.9 | 11.7 | 13.9 | 5.8 |
| P1 GBH-YOLOv5 | 960 | 19.1 | 16.5 | 16.5 | 5.2 |
| P2 YOLOv5s | 640 | 11.8 | 12.4 | 15.0 | 1.7 |
| P2 YOLOv5s+GAM+ASFF+DIoU | 640 | 27.9 | 24.0 | 22.8 | 129.9 |
| P2 YOLOv8s | 640 | 13.1 | 12.6 | 11.9 | 1.5 |
| P3 VGG-19 CNN | 300 | 11.7 per tile (batch 16, fp32) | | | |
