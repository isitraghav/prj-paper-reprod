# Reproduction checklist

Live progress tracker. Details, reasons and the full deviation list are in `init.md` (§11 Deviations, §13 Progress log).
Machine for everything below: **Laptop B**, RTX 3050 Laptop 6 GB, Ryzen 7 7840HS, 15 GB RAM, Windows 11, driver 595.97.

Legend: `[x]` done · `[~]` running · `[ ]` to do · `[-]` deferred (by user decision)

## Scope decided on 2026-10-17
- **P2: train now** (YOLOv5s, YOLOv5s+GAM+ASFF+DIoU, YOLOv8s; 100 epochs each).
- **P1: on this machine too** (user decision 2026-10-17, after Kaggle attempts were dropped: no GPU/internet before phone verification, then P100 unavailable/T4 queue). GBH uses the **released** `yolov5s_pv.yaml` (user decision). Starts automatically after P2 via `code/repro/run_queue_local.sh` (P2 eval → P1 1-epoch smoke → P1 full). Kaggle tooling kept in `code/repro/kaggle/` (unused).

## 1. Environments (uv, locked)
| | Paper states | We use | Match |
|---|---|---|---|
| P1 Python | 3.8 (GBH README) | 3.8.20 | ✅ |
| P1 PyTorch / CUDA | 1.8 (README 1.8.1) / 11.1 | 1.8.1+cu111 | ✅ |
| P2 Python | 3.8.8 | 3.8.8 | ✅ |
| P2 PyTorch | 1.11.0 | 1.11.0 | ✅ |
| P2 CUDA | 11.1 | **11.3** | ❌ no cu111 wheel exists for torch 1.11.0 (only cu113/cu115) |
| P2 ultralytics | not stated | 8.0.99 (last release before submission) | — |
| Other packages | not stated | pinned `==` in `code/repro/env/p{1,2}/pyproject.toml`, locked in `uv.lock` | — |

- [x] `envs/p1` synced from `code/repro/env/p1/uv.lock` *(currently built with `uv pip`, same versions; re-sync from lock after P1 is un-deferred)*
- [x] `envs/p2` synced from `code/repro/env/p2/uv.lock` (Python 3.8.8)
- [x] Both envs see the GPU (CUDA ok)

Rebuild on any machine:
```
UV_PROJECT_ENVIRONMENT="$PWD/envs/p1" uv sync --project code/repro/env/p1 --frozen
UV_PROJECT_ENVIRONMENT="$PWD/envs/p2" uv sync --project code/repro/env/p2 --frozen
```

## 2. Code, weights, data
- [x] GBH-YOLOv5 @ `51ee042` (+ whitespace fix for `train.py` line 1, `code/repro/p1/gbh_51ee042_fix.patch`)
- [x] ultralytics/yolov5 v7.0 @ `915bbf2` (+ P2 modules, `code/repro/p2/yolov5_v7.0_p2.patch`)
- [x] Weights, the versions each codebase fetches itself: `yolov5s.pt` v5.0 (P1), `yolov5s.pt` v7.0 (P2), `yolov8s.pt` assets v0.0.0 (8.0.0, the default for ultralytics 8.0.99). SHA256 in `weights/SHA256SUMS`
- [x] P1 data: 1108 images (= paper), frozen split 886 / 222 (= paper counts; membership unpublishable → re-created, seed 0)
- [x] P2 data: ELDDS1400C5 v1, 6 classes, **1186** / 190 (paper 1187 / 190; 2 stray-label images excluded)
- [x] Sanity box drawings checked (`results/sanity/`)

## 3. Models
- [x] P1 YOLOv5s: 224 layers / 7.06 M fused = Table 6 exactly
- [x] P1 GBH (released yaml): 290 / 7.04 M (paper 270 / 7.24; nothing reproduces 270)
- [x] P2 YOLOv5s 16.0 GFLOPs (paper 16) · GAM+ASFF **30.0** (paper 30.0) · YOLOv8s 28.7 at nc=6 (paper 28.7)
- [x] P2 DIoU-NMS implemented and unit-tested (inference only)

## 4. Training parameters: paper vs ours
| Param | P2 paper | P2 ours (all 3 models) | Match |
|---|---|---|---|
| Optimizer | Adam | Adam (`--optimizer Adam` / `optimizer=Adam`) | ✅ |
| Initial LR | 0.01 | 0.01 (`hyp.scratch-low.yaml` lr0 / `lr0=0.01`) | ✅ |
| Epochs | 100 | 100 | ✅ |
| Image size | 640 × 640 | 640 | ✅ |
| Init weights | yolov5s.pt / yolov8s.pt | same (see §2) | ✅ |
| YOLOv8 mosaic | off in last 10 epochs | `close_mosaic=10` | ✅ |
| Batch | 32 | see below | ❌ VRAM |
| Other hyps | not stated | codebase defaults | ✅ |

**Batch:** no P2 model fits physical batch 32 in 6 GB (measured: the driver spills into system RAM, 15–32 s/iter). Used: largest **divisor of 64** that fits, so that with the codebases' built-in accumulation (`nbs=64`) every optimizer step sees **64 images, as in the paper** (paper: 32 × 2 accumulations), with identical weight-decay scaling. Only BatchNorm batch statistics differ. Max-fit per model (probe): YOLOv5s 24, GAM+ASFF 16, YOLOv8s 24.

| Param | P1 paper | P1 ours (deferred) | Match |
|---|---|---|---|
| Optimizer / LR | Adam / 0.001 | `--adam`, `hyp.gbh.yaml` lr0 0.001 | ✅ |
| Epochs / img | 500 / 960 | 500 / 960 | ✅ |
| Batch | 16 | **GBH 4 × 16 accum, YOLOv5s 8 × 8** (largest divisors of 64 that fit; max-fit GBH 6, YOLOv5s 12) → 64 images/step as paper's 16 × 4 | ❌ VRAM |
| Warm-up | max(3 ep, 1000 it) ≈ 18 ep @16 | same code → ≈ 4.5 ep (GBH) / 9 ep (YOLOv5s) | ❌ consequence of batch |

## 5. P2 runs
| Run | Batch (phys × accum) | Status | Time | Notes |
|---|---|---|---|---|
| smoke p2_v5s @ 32 | 32 | [x] ❌ | | spills to 6.8 GB, crashed after 8 iters (`CUDA error: resource already mapped`) → 32 impossible |
| smoke p2_v5s | 16 × 4 | [x] | 45 s/ep | mAP@.5 0.051 → 0.19 → 0.376; Adam lr 0.01, wd 0.0005 logged |
| smoke p2_v5_improved | 16 × 4 | [x] | 1–2 min/ep | mAP@.5 0.103 → 0.295; 30.0 GFLOPs; slight VRAM spill (5.7 GB) |
| smoke p2_v8s | 16 × 4 | [x] | 55 s/ep | mAP@.5 0.049 → 0.406 → 0.522 |
| eval pipeline (smoke weights) | | [x] | | DIoU-NMS ≈ NMS (0.326 vs 0.328) after 2 fixes (fp16, time limit); v8 TTA wrapper works (0.522 → 0.560) |
| **p2_v5s** | 16 × 4 | [x] 15:12–15:52, exit 0 | 0.65 h | best.pt (ep 96) on test: P 0.746, R 0.711, mAP@.5 **0.755** (paper 0.738), mAP@.5:.95 0.516 (paper 0.502) |
| **p2_v5_improved** | 16 × 4 | [x] 16:12–19:38, exit 0 | 3.42 h | best.pt (ep 94) on test, standard NMS: P 0.749, R 0.727, mAP@.5 **0.753** (paper 0.763), mAP@.5:.95 0.513 (paper 0.516). First attempt aborted (alloc setting, see init.md §11.16) |
| **p2_v8s** | 16 × 4 | [x] 20:03–21:26, exit 0 | 1.38 h | best.pt: P 0.757, R 0.749, mAP@.5 **0.764** (paper 0.765), mAP@.5:.95 0.549. First attempt aborted at ep 3 (Windows commit exhausted, workers=2 → restarted with workers=1) |

## 6. P2 evaluation (test split, 190 images) — done 2026-10-17 21:33
`bash code/repro/p2/eval_p2.sh && envs/p2/Scripts/python.exe -I code/repro/summarize.py` → `results/summary.md`, `results/summary.csv`

| Model | mAP@.5 no TTA (paper / ours) | mAP@.5:.95 (paper / ours) | P / R ours (paper) | TTA @640 (paper / ours) | TTA @832 (sensitivity) |
|---|---|---|---|---|---|
| YOLOv5s | 0.739 / **0.754** (+0.015) | 0.502 / 0.516 | 0.748 / 0.707 (0.726 / 0.723) | 0.766 / **0.766** (±0) | 0.774 |
| GAM+ASFF+DIoU-NMS | 0.762 / **0.757** (−0.005) | 0.516 / 0.514 | 0.754 / 0.727 (0.735 / 0.760) | 0.777 / 0.758 (−0.019) | 0.773 |
| YOLOv8s | 0.765 / **0.764** (−0.001) | — / 0.550 | 0.764 / 0.746 (—) | 0.775 / 0.753 (−0.022) | 0.761 |

Findings:
- **Without TTA all three models are within ±0.016 mAP@.5 of the paper** (YOLOv8s −0.001, improved −0.005, YOLOv5s +0.015).
- **The paper's main claim does not reproduce:** GAM+ASFF+DIoU beats YOLOv5s by only +0.003 here (paper +0.023 without TTA, +0.011 with TTA). Our GAM/ASFF placement is inferred (no released code), so this is not conclusive.
- TTA: YOLOv5s gains exactly as in the paper (+0.012 → 0.766). The improved model gains almost nothing (+0.001), and YOLOv8s *loses* (−0.011) with 8.0.99's TTA (forced on via `v8_val.py`, since 8.0.99's validator ignores `augment`).
- DIoU-NMS vs standard NMS on the improved model: 0.757 vs 0.753 (+0.004).
- Discarded: an earlier v5s TTA number (0.727) from a stale eval that ran concurrently with training under memory exhaustion.
- [x] Confusion matrix, PR and F1 curves: saved per run in `results/p2/eval/<run>_<tta>/` (stock codebase thresholds; not the paper's conf 0.2 / IoU 0.5 for Fig. 16)

## 7. P1 → runs on the RTX 4070 laptop (8 GB)
Decision 2026-10-17: P1 moves to the RTX 4070 laptop (faster, more VRAM). Laptop B only finishes P2.

| Run | Batch on 4070 | Status | Time |
|---|---|---|---|
| mem probe | — | [x] | YOLOv5s 8: 6.01 GiB 0.18 s/it · 16: 6.46 GiB 0.58 s/it OK · GBH 4: 5.77 GiB · 8: 5.57 GiB 0.34 s/it OK · 16: 11.4 GiB 34 s/it SPILL |
| smoke (1 epoch each) | 16 / 8, workers 4 | [x] exit 0 / 0 | v5s 34 s/ep, mAP@.5 0.042 · GBH 54 s/ep, mAP@.5 0.008 |
| p1_yolov5s (500 ep @ 960) | **16 × 4 accum = paper** | [x] 2026-10-17 21:18 → 2026-10-18 07:52, exit 0 (user pause after ep 350, 01:08–06:56, resumed) | ~20–25 s/ep (ep 20–48 slowed by background GPU apps); best.pt = ep 297 |
| p1_gbh (released yaml, 500 ep @ 960) | 8 × 8 accum | [x] 2026-10-18 07:52 → 13:22, exit 0 | 5.49 h, ~39 s/ep; best.pt = ep 457 |
- [x] P1 evaluation (val 222, 960, best.pt, batch 1; `results/p1/eval_p1_*.log`)

| Model | mAP@.5 | P | R | broken | hot_spot | black_border | scratch | no_electricity | mAP@.5:.95 | layers / params (fused) | ms/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv5s paper | 78.1 | 83.2 | 73.4 | 78.5 | 87.8 | 85.4 | 69.3 | 88.0 | — | 224 / 7.06 M | 484 |
| **YOLOv5s ours** | **86.5** | 87.1 | 80.9 | 85.0 | 90.2 | 97.4 | 65.5 | 94.3 | 52.9 | 224 / 7.06 M | 17.1 |
| GBH paper | 97.8 | 96.4 | 93.3 | 99.5 | 97.5 | 97.2 | 97.4 | 98.0 | — | 270 / 7.24 M | 587 |
| **GBH ours (released yaml)** | **82.1** | 81.1 | 81.2 | 81.0 | 82.8 | 92.3 | 63.4 | 91.1 | 51.3 | 290 / 7.04 M | 20.3 |

→ YOLOv5s baseline is 8.4 above the paper; GBH is 15.7 below the paper and **4.4 below our own YOLOv5s**: the paper's GBH gain does not reproduce (released yaml, re-created split, GBH batch 8).

**Follow-up: YOLOv5-1** (paper ablation, backbone C3 → BottleneckCSP; `code/repro/p1/yolov5_1.yaml`, 228 layers / 7.15 M = paper exactly; batch 16 = paper)
- [x] Stopped at epoch 202 by the pre-set epoch-200 rule: best mAP@.5 0.861 vs YOLOv5s 0.862 by the same epoch. Resumable from `results/p1/p1_yolov5_1/weights/last.pt`.
- [x] best.pt (ep 176), val @960: **mAP@.5 85.2** (paper 94.2), P 85.3, R 79.2; broken 77.9 · hot_spot 89.4 · black_border 91.7 · scratch 73.0 · no_electricity 94.1. → The paper's +16.1 from BottleneckCSP does not reproduce, even with an exactly specified model (ours −1.3 vs our YOLOv5s).

**Steps on the 4070 laptop** (Git Bash on Windows, or bash on Linux; repo root):
```bash
git clone https://github.com/isitraghav/prj-paper-reprod && cd prj-paper-reprod
# 1. exact env from the lock (needs uv)
UV_PROJECT_ENVIRONMENT="$PWD/envs/p1" uv sync --project code/repro/env/p1 --frozen
PY=envs/p1/Scripts/python.exe   # Linux: envs/p1/bin/python
# 2. pinned code + patch, data, weights
git clone -q https://github.com/CCNUZFW/GBH-YOLOv5 code/gbh-yolov5
git -C code/gbh-yolov5 checkout -q 51ee042edf913bc1a6c3f25b6f296d6f8d6ef079
git -C code/gbh-yolov5 apply --whitespace=nowarn ../../code/repro/p1/gbh_51ee042_fix.patch
git clone -q https://github.com/CCNUZFW/PV-Multi-Defect datasets/raw/PV-Multi-Defect
git -C datasets/raw/PV-Multi-Defect checkout -q 2c4df0c1477032d6b0dd9372a517cdbfd6a09e09
mkdir -p weights && curl -L -o weights/yolov5s_v5.0.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
(cd weights && sha256sum -c --ignore-missing SHA256SUMS)        # must say yolov5s_v5.0.pt: OK
# 3. data from the FROZEN split (reuses datasets/splits/pv_multi_defect; must print train 886, val 222)
$PY -I code/repro/voc2yolo_pv.py --src datasets/raw/PV-Multi-Defect --dst datasets/pv_multi_defect_yolo --splits datasets/splits/pv_multi_defect
# 4. check real VRAM limits (pick the largest divisor of 64 marked OK; expect YOLOv5s 16, GBH 8)
(cd code/gbh-yolov5 && ../../$PY ../repro/mem_probe.py --cfg ../repro/p1/yolov5s.yaml --img 960 --nc 5 --batches 8 16)
(cd code/gbh-yolov5 && ../../$PY ../repro/mem_probe.py --cfg ../repro/p1/gbh_released.yaml --img 960 --nc 5 --batches 4 8 16)
# 5. smoke, then full (sleep off, plugged in, performance mode)
BATCH_YOLOV5S=16 BATCH_GBH=8 bash code/repro/p1/run_p1.sh smoke yolov5s gbh
BATCH_YOLOV5S=16 BATCH_GBH=8 bash code/repro/p1/run_p1.sh full yolov5s gbh
```
Then record machine, batch, s/epoch in `init.md` §13, and copy `results/p1/` back (weights are not committed).

## 8. P3: Deitsch et al. 2019 (ELPV: CNN vs SVM), on this machine
No code released; re-implemented from the paper (`code/repro/p3/`). Details and deviations: `init.md` § P3.

| Item | Status | Notes |
|---|---|---|
| Envs (uv, locked) | [x] | p3cnn TF 2.10.1 GPU; p3svm OpenCV 3.4.18 + sklearn 0.22 (exact TF 1.4 / OpenCV 3.3.1 not feasible, user kept these) |
| Data + frozen split | [x] | ELPV @93e82ae, 1968/656 = paper Table 3 per type × label |
| SVM (KAZE/VGG + VLAD + linear SVM) | [x] | test acc_w **69.70 %**, F1_w **66.32 %**, AUC_w 74.88 % (paper 82.44 / 82.52 / 88.51) |
| CNN (VGG-19 regression, 100+100 ep, batch 16) | [x] | test acc_w **86.49 %**, F1_w 84.20 %, AUC_w 90.18 % (paper 88.42 / 88.39) |
| CNN seeds 1, 2 (variance check) | [x] | acc_w 85.87 / 84.98; mean ± sd (3 seeds) **85.78 ± 0.76**, F1_w 83.38 ± 0.77 → gap to paper is systematic |
| SVM bounded check (unstated details, training-CV selection) | [x] | KAZE thr 0.0001, no PCA, C 1 → acc_w **79.65 %**, F1_w 77.09 %, AUC_w 84.08 % |
| P3 summary vs paper | [x] | `init.md` § P3 results; CNN > SVM reproduces; SVM gap = keypoint density |

## 9. Cross-dataset comparison (inference only), 2026-10-22, RTX 4050 Laptop 6 GB
Every trained model in `models/` (P1 YOLOv5s + GBH, P2 YOLOv5s + Improved + YOLOv8s, P3 CNN seed 0), unchanged, on all
three evaluation splits. Report: **`COMPARISON.md`**; scripts `code/repro/cross/`; numbers `results/cross/metrics.json`.
- [x] Detector inference dumps, 5 models × 3 datasets (`run_dumps.sh`), each at its own paper's input size / NMS
- [x] Shared tasks: class-agnostic detection (PV, ELDDS); defective-vs-functional tiles (PV 4×4, ELDDS 6×6, ELPV cells)
- [x] One evaluator (numpy re-implementation of yolov5 v7.0 metrics), checked against the official home numbers (≤ 0.009 mAP@.5); CNN tile path reproduces P3 home exactly
- [x] Bootstrap 95 % CIs; sensitivity S1 (foreign input size) and S2 (ELDDS 12×12 tiles)
- [x] Figures + qualitative examples in `results/cross/figs/`
- [x] Class-mapped detection (crack / line / dark, coarse variant) + class-confusion figure (`class_map.py`, §3.1)
- [x] Paired bootstrap tests of the architecture differences + per-class mapped AP CIs (`paired_tests.py`, §3.2)
- [x] Audit of every claim in `COMPARISON.md` against the numbers (corrections in TL;DR, §1, §3–§10)

## 10. Still to do (not needed for the comparison)
- [ ] P1 YOLOv5-2 ablation: died at epoch 54/500 on 2026-10-20. Resume: `train.py --resume ../../results/p1/p1_yolov5_2/weights/last.pt` in `code/gbh-yolov5`, **WORKERS=0** on the RTX 4050.
- [ ] Email the P1 authors (zfwang@ccnu.edu.cn): train/val lists, the annotation version with 4235 boxes, the exact GBH yaml (270 layers / 7.24 M) or weights. Email the P2 authors: GAM/ASFF code. (See `CLAIMS.md`, "Is the gap caused by our code or training?")
- [ ] Seed spread: 2 more seeds of P2 YOLOv5s and Improved (≈ 8 h on the RTX 4050), to compare the paper's +0.023 against run-to-run variance.
- [ ] P2 ablations v5_gam, v5_asff + DIoU-only eval: `bash code/repro/run_p2_after_p1.sh` (waits for yolov5_2, then runs them one after another; never in parallel with P1).
