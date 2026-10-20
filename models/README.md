# Final trained models

Stored with **Git LFS** (`.gitattributes`). Clone with Git LFS installed, or skip the binaries with
`GIT_LFS_SKIP_SMUDGE=1 git clone ...` and fetch one later with `git lfs pull --include models/p3/<file>`.
Checksums: `models/SHA256SUMS` (`cd models && sha256sum -c SHA256SUMS`). Settings, deviations and logs: `init.md`,
`CHECKLIST.md`, `results/`.

## P1: GBH-YOLOv5 (PV-Multi-Defect)
Trained on the RTX 4070 Laptop (8 GB, `4070_JOBS.md`). Val split, 222 images, evaluated at 960 px, batch 1.

| File | Model | mAP@.5 (val) | Paper mAP@.5 | Load with |
|---|---|---|---|---|
| `p1/yolov5s.pt` | YOLOv5s v5.0 (baseline; batch 16 = paper) | **86.5** | 78.1 | `code/gbh-yolov5` @51ee042 + `code/repro/p1/gbh_51ee042_fix.patch`, env `p1` |
| `p1/gbh_yolov5.pt` | GBH-YOLOv5 released yaml (`yolov5s_pv.yaml`; batch 8 × 8 accum) | **82.1** | 97.8 | same |
| `p1/yolov5_1.pt` | YOLOv5-1 ablation (BottleneckCSP backbone; `code/repro/p1/yolov5_1.yaml`; batch 16; stopped ep 202) | **85.2** | 94.2 | same, cfg `code/repro/p1/yolov5_1.yaml` |

Evaluation command: `(cd code/gbh-yolov5 && ../../envs/p1/Scripts/python.exe test.py --weights ../../models/p1/<file>.pt --data data/pv_multi_defect.yaml --img 960 --batch 1 --task val)`
Per-class details and deviation notes: `CHECKLIST.md` §7, `init.md` §13.

## P2: ELDDS1400C5 (EL defect detection, 5 classes)
`best.pt` of each final run (stripped, fp16), as evaluated in `CHECKLIST.md` §6. Test split, 190 images.

| File | Model | mAP@.5 (no TTA / TTA 640) | Paper (no TTA / TTA) | Load with |
|---|---|---|---|---|
| `p2/yolov5s.pt` | YOLOv5s v7.0 | 0.754 / 0.766 | 0.739 / 0.766 | `code/yolov5-v7.0` @ v7.0 + `code/repro/p2/yolov5_v7.0_p2.patch`, env `p2` |
| `p2/yolov5s_gam_asff.pt` | YOLOv5s + grouped GAM + ASFF head (evaluate with DIoU-NMS: `--diou-nms`) | 0.757 / 0.758 | 0.762 / 0.777 | same (needs the patch: GAM/ASFF classes) |
| `p2/yolov8s.pt` | YOLOv8s (ultralytics 8.0.99) | 0.764 / 0.753 | 0.765 / 0.775 | env `p2`, `code/repro/p2/v8_val.py` for TTA |

Evaluation commands: `code/repro/p2/eval_p2.sh`.

## P3: ELPV (Deitsch et al. 2019, functional vs defective cells)
Test split `datasets/splits/elpv/test.txt` (656 cells). Weighted = confidence weights of the paper's Table 2.

| File | Model | Test acc_w / F1_w / AUC_w | Paper acc / F1 |
|---|---|---|---|
| `p3/cnn_vgg19_seed0.h5` | VGG-19 regression, final weights after 100 + 100 epochs (Keras `save_weights`); primary run | 86.49 / 84.20 / 90.18 | 88.42 / 88.39 |
| `p3/cnn_vgg19_seed1.h5` | same settings, seed 1 | 85.87 / 83.29 / 89.43 | |
| `p3/cnn_vgg19_seed2.h5` | same settings, seed 2 | 84.98 / 82.66 / 90.86 | |
| | CNN mean ± sd over seeds 0–2 | 85.78 ± 0.76 / 83.38 ± 0.77 / 90.16 ± 0.72 | 88.42 / 88.39 |
| `p3/svm_kaze_vgg_vlad_strict.npz` | KAZE (OpenCV defaults) / VGG + 5×VLAD (K 100) + PCA-whitening 32 + linear SVM, C 0.1 | 69.70 / 66.32 / 74.88 | 82.44 / 82.52 |
| `p3/svm_kaze_vgg_vlad_thr1e-4_nopca.npz` | same, KAZE threshold 1e-4, no PCA, C 1 (bounded check, init.md P3-12) | 79.65 / 77.09 / 84.08 | 82.44 / 82.52 |

- CNN: `envs/p3cnn/Scripts/python.exe -I code/repro/p3/eval_cnn.py models/p3/cnn_vgg19_seed0.h5` (re-checked: identical
  metrics). Input: grayscale cell replicated to 3 channels, `vgg19.preprocess_input`; output = defect probability,
  defective iff > 0.5.
- SVM `.npz` (numpy arrays only, no pickles): `codebooks` (5, 100, 120), `pca_mean` / `pca_components` / `pca_scale`
  (empty if no PCA), `svm_coef`, `svm_intercept`, `meta` (JSON). Reference scorer: `predict_npz()` in
  `code/repro/p3/export_svm.py`, which refits, saves, reloads and asserts the logged test accuracy.
  Descriptors: OpenCV 3.4.18 contrib (`env p3svm`) `KAZE_create(threshold=meta.kaze_threshold)` + `xfeatures2d.VGG_create()`.
