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
| **p2_v5_improved** | 16 × 4 | [~] started 15:52 | ~2 h est. | |
| **p2_v8s** | 16 × 4 | [ ] queued | ~1.8 h est. | |

## 6. P2 evaluation (test split, 640)
Run after training: `bash code/repro/p2/eval_p2.sh && envs/p2/Scripts/python.exe -I code/repro/summarize.py`
- [ ] p2_v5s: no TTA / TTA
- [ ] p2_v5_improved (DIoU-NMS): no TTA / TTA
- [ ] p2_v8s: no TTA / TTA (wrapper needed: 8.0.99 ignores `val augment=True`)
- [ ] Confusion matrix (conf 0.2, IoU 0.5), PR and F1 curves
- [ ] `results/summary.md` paper vs ours

## 7. P1 → runs on the RTX 4070 laptop (8 GB)
Decision 2026-10-17: P1 moves to the RTX 4070 laptop (faster, more VRAM). Laptop B only finishes P2.

| Run | Batch on 4070 (est.) | Status | Est. time |
|---|---|---|---|
| mem probe | — | [ ] | 5 min |
| smoke (1 epoch each) | 16 / 8 | [ ] | ~5 min |
| p1_yolov5s (500 ep @ 960) | **16 = paper** | [ ] | ~3.5–4.5 h |
| p1_gbh (released yaml, 500 ep @ 960) | 8 × 8 accum | [ ] | ~5.5–7 h |
- [ ] P1 evaluation (val, 960, per-class mAP@.5, P, R, s/img)

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
