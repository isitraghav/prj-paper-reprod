# Reproduction runbook: GBH-YOLOv5 (P1) and Improved-YOLOv5 / YOLOv8 on ELDDS1400C5 (P2)

This file is the single source of truth for the project. Work through it **top to bottom, one step at a time**. Tick each `[ ]` and add a line to the **Progress log** at the end before moving on. The repo is meant to be carried between laptops, so read **§2 Precautions** before running anything on a new machine.

---

## 1. Goal and scope

We re-train the main models of two papers under their published settings, for a comparison paper.

| ID | Paper | Code | Data |
|---|---|---|---|
| **P1** | Li, Wang, Zhang. *GBH-YOLOv5: Ghost Convolution with BottleneckCSP and Tiny Target Prediction Head Incorporating YOLOv5 for PV Panel Defect Detection*. Electronics 2023, 12, 561. `papers/yoloGBH-paper1.pdf` | Released: github.com/CCNUZFW/GBH-YOLOv5 (a fork of the YOLOv5 v5.0 era) | github.com/CCNUZFW/PV-Multi-Defect |
| **P2** | Mazen, Abul Seoud, Shaker. *Deep Learning for Automatic Defect Detection in PV Modules Using Electroluminescence Images*. IEEE Access 2023, 11, 57783. `papers/Deep_Learning_for_...pdf` | **Not released**, so we re-implement it on ultralytics/yolov5 v7.0 + ultralytics 8.0.x | ELDDS1400C5 on Roboflow Universe: `fma04-fayoum-edu-eg/eldds1400c5-dataset`. It was uploaded by P2's corresponding author (fma04@fayoum.edu.eg). |

**In scope (decided): main models only**
- P1: **YOLOv5s baseline** and **GBH-YOLOv5**, for Table 4 rows 1 and 4, Table 5 rows 1 and 4, and Table 6 rows 1 and 4.
- P2: **YOLOv5s_only**, **YOLOv5s+GAM+ASFF+DIoU** ("Improved YOLOv5") and **YOLOv8s**, each evaluated **with and without TTA** (Tables 1 and 2, Figs. 11, 12 and 16).
- Out of scope for now: the P1 intermediate ablations (YOLOv5-1, YOLOv5-2), the P1 third-party baselines (Faster-RCNN, SVM, Mask-RCNN, SSD, YOLOv3), and the P2 single-module ablations (GAM_only, ASFF_only, DIoU_only). The configs are cheap to add later.

**P1 architecture decision: follow the paper text strictly, not the released yaml.** There is a mandatory consistency check in step 6.

---

## 2. Precautions (read on every machine)

### Secrets
- **Never write the Roboflow API key into any file in this repo, any log, or any commit.** Export it in the shell only:
  `export ROBOFLOW_API_KEY=...` (or put it in `~/.bashrc` / `~/.zshrc`, which is outside the repo).
- The key was pasted into a chat once. **Regenerate it** in Roboflow → Settings → API after the dataset is downloaded.
- `.gitignore` blocks `.env`, but still run `git diff --cached` before every commit.

### Moving between laptops
- Envs, datasets, weights and run outputs are **not committed** (see `.gitignore`). On a new machine, redo §4 (envs) and §5 (data). Everything is pinned, so the result is identical.
- **Commit the frozen split files** (`datasets/splits/pv_multi_defect/*.txt`) and all configs and scripts. **Never regenerate the split on the second machine.** Both machines must train on the same split.
- To continue a training run on another laptop, copy `results/<p>/<run>/weights/last.pt` across manually (USB or scp) and use `--resume` (§9). Don't commit `.pt` files.
- Record the machine for **every** run in the Progress log (GPU, VRAM, physical batch, torch/CUDA). Results from different GPUs are compared in the paper, so this matters.
- Paths in configs are **relative to the repo root**. Never hard-code `/home/raghav/...`.

### Hardware limits
| Machine | GPU | VRAM | RAM | Use for |
|---|---|---|---|---|
| Laptop A (this one) | RTX 2050 | 4 GB | 7 GB | env/data setup, smoke tests, P2 (100-epoch runs) |
| Laptop B | RTX 3050 | 6 GB | ? | P1 500-epoch runs at 960 px (long) |

- **Neither GPU fits the papers' physical batch sizes** (P1: 16 @ 960 px; P2: 32 @ 640 px). Both codebases accumulate gradients to a nominal batch of 64 (`nbs=64`, `accumulate = round(64 / batch)`) and scale weight decay by `batch × accumulate / 64`. So with a smaller batch the **optimizer steps match the paper**, and only BatchNorm batch statistics differ. Use the **largest batch that fits** (find it in step 9) and log it.
- Warm-up length depends on iterations per epoch (`max(3 epochs, 1000 iters)` in YOLOv5). This is a small, unavoidable side effect of a smaller batch. Log it as well.
- RAM is small: use `--workers 2`, **no `--cache-images`/`cache=ram`**, and close the browser during training. If the dataloader gets OOM-killed, drop to `--workers 1`.
- Laptops throttle and sleep. Plug in, disable suspend, and keep the lid open, or set lid-close to do nothing. Run long jobs with `nohup` or `tmux` (§9).
- Check the temperature now and then: `nvidia-smi -l 30`.

### Untrusted downloads
- Cloned repos and downloaded data go into their own directories (`code/gbh-yolov5`, `code/yolov5-v7.0`, `datasets/raw/*`). Our own scripts live in `code/repro/` and take data paths as **arguments**. Run our scripts with `python -I`.
- Read any shell script from a download before running it. We don't need any of them: skip the `data/scripts/get_*.sh` scripts.

### Faithfulness rules
- Change only what is listed in §3 "Settings". Anything else that has to change goes into §11 **Deviations** before the run, with a reason.
- Don't tune anything to the test/val set. Train once per model with a fixed seed.

---

## 3. Settings to reproduce (from the papers)

| | P1: GBH-YOLOv5 and its YOLOv5s baseline | P2: all three models |
|---|---|---|
| Codebase | CCNUZFW/GBH-YOLOv5 @ `51ee042edf913bc1a6c3f25b6f296d6f8d6ef079` | ultralytics/yolov5 tag **v7.0** (`915bbf294bb74c859f0b41f1c23bc395014ea679`); `ultralytics==8.0.99` (last release before P2 was received, 13 May 2023) |
| Init weights | COCO `yolov5s.pt` from the yolov5 **v5.0** release | `yolov5s.pt` from the **v7.0** release; `yolov8s.pt` from ultralytics 8.0.99 assets |
| Optimizer | **Adam** (`--adam`) | **Adam** |
| Initial LR | **0.001** | **0.01** |
| Batch (paper) | **16** (we accumulate; see §2) | **32** (we accumulate) |
| Epochs | **500** | **100** |
| Image size | **960 × 960** | **640 × 640** |
| Other | Rest of hyp = repo `data/hyp.scratch.yaml`; Mosaic on; autoanchor on; seed `init_seeds(2)` built in | YOLOv5: default `data/hyps/hyp.scratch-low.yaml`; YOLOv8: defaults + `close_mosaic=10` (P2 states this) |
| Paper env | RTX 3090, PyTorch 1.8, CUDA 11.1 | Kaggle P100, Python 3.8.8, PyTorch 1.11.0, CUDA 11.1 |
| Data split | 886 train / 222 val (1108 images, 600 × 600) | 1187 train / 190 test |

### Targets to compare against

**P1, Table 4 / 5 / 6 (val set)**
| Model | mAP@.5 | P | R | broken | hot_spot | black_border | scratch | no_electricity | layers | params (paper unit "10⁴", likely 10⁶) | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv5s | 78.1 | 83.2 | 73.4 | 78.5 | 87.8 | 85.4 | 69.3 | 88.0 | 224 | 7.06 | 0.484 |
| GBH-YOLOv5 | 97.8 | 96.4 | 93.3 | 99.5 | 97.5 | 97.2 | 97.4 | 98.0 | 270 | 7.24 | 0.587 |

**P2, Table 1 / 2 (test set)**
| Model | P | R | mAP@.5 | mAP@.5:.95 | mAP@.5 w/o TTA (T2) | mAP@.5 with TTA (T2) | GFLOPs |
|---|---|---|---|---|---|---|---|
| YOLOv5s_only | 0.726 | 0.723 | 0.738 | 0.502 | 0.739 | 0.766 | 16 |
| YOLOv5s+GAM+ASFF+DIoU | 0.735 | 0.760 | 0.763 | 0.516 | 0.76222 | 0.777 | 30.0 |
| YOLOv8s | — | — | — | — | 0.765 | 0.775 | 28.7 |

(Also: YOLOv5s+GAM_only has 21.4 GFLOPs and YOLOv5s+ASFF_only has 25.8 GFLOPs. We use these to locate GAM and ASFF; see step 7.)

### What can match exactly and what cannot
| Item | Status | Note |
|---|---|---|
| Optimizer, LR, epochs, image size, init weights | ✅ exact | |
| P1 code base | ✅ exact | authors' repo, pinned commit |
| P2 dataset | ✅ very likely exact | uploaded by P2's own author, 1378 images ≈ 1187 + 190. Verify in step 5. |
| P1 dataset content | ✅ exact | the repo's data commits all date from 2023-01-15; later commits only touch the README |
| **P1 train/val split** | ❌ **cannot match** | The authors split randomly with no seed (`prepare_data.py`) and never published the lists. The repo has 1106 xml / 1113 jpg against the paper's 1108. **Highest risk to P1 numbers.** Optional fix: email zfwang@ccnu.edu.cn for their train/val lists; if they reply, use them in step 5 instead of ours. |
| Physical batch | ❌ smaller | same optimizer steps through accumulation; BN statistics differ |
| P1 architecture | ⚠ ambiguous | paper text ≠ released yaml; Table 6 is the arbiter (step 6) |
| P2 GAM / ASFF placement | ⚠ inferred | no code was released; placement is calibrated to Table 2 GFLOPs |
| P2 class count | ⚠ | the text says 5 classes, Fig. 2 shows 6. Use whatever the authors' dataset has. |
| GPU / CUDA build | ❌ different | torch 1.11 has no cu111 wheel, so we use cu113 |
| Variance | ⚠ | the papers report ±95 % CI; we do 1 run per model (expect about ±1–2 mAP) |

---

## 4. Environment setup (per machine)

Prerequisites: an NVIDIA driver (any recent one; it is backward compatible with CUDA 11.x runtimes), `git`, `uv`, and `gh` (optional).

```bash
cd <repo-root>                         # wherever this repo lives on this machine
export PRJ=$(pwd)

# ---- P1 env: matches the GBH README (py3.8, torch 1.8.1, cu111)
uv venv envs/p1 --python 3.8
uv pip install --python envs/p1/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu111 --index-strategy unsafe-best-match \
  torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
  "numpy<1.24" "opencv-python<4.9" "matplotlib<3.8" "Pillow<10" PyYAML scipy tqdm \
  "tensorboard<2.12" seaborn pandas thop "pycocotools>=2.0" "protobuf<4"

# ---- P2 env: matches P2 (py3.8, torch 1.11.0)
uv venv envs/p2 --python 3.8
uv pip install --python envs/p2/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  ultralytics==8.0.99 roboflow "numpy<1.24" "opencv-python<4.9" "Pillow<10" \
  gitpython psutil thop seaborn pandas "tensorboard<2.12" "protobuf<4"
```

**Current method (2026-10-17, supersedes the `uv pip install` lines above):** both envs are locked uv projects. On any machine:
```bash
UV_PROJECT_ENVIRONMENT="$PWD/envs/p1" uv sync --project code/repro/env/p1 --frozen   # py 3.8.20, torch 1.8.1+cu111
UV_PROJECT_ENVIRONMENT="$PWD/envs/p2" uv sync --project code/repro/env/p2 --frozen   # py 3.8.8,  torch 1.11.0+cu113
```
(`UV_PROJECT_ENVIRONMENT` must be absolute, otherwise it is resolved inside the project folder. On Windows the interpreter is `envs/pX/Scripts/python.exe`.)

Sanity check (both envs):
```bash
for e in p1 p2; do envs/$e/bin/python -c "import torch;print('$e',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0));print((torch.ones(2,device='cuda')*2).sum())"; done
```
- [ ] Step 4 done on Laptop A
- [ ] Step 4 done on Laptop B

If `torch 1.8.1+cu111` fails on a newer GPU (sm_86 is fine for both the 2050 and the 3050), log it in §11 and use `torch==1.10.2+cu111` instead.

---

## 5. Code and data download

```bash
# ---- code (untrusted; own dirs, pinned)
git clone https://github.com/CCNUZFW/GBH-YOLOv5 code/gbh-yolov5
git -C code/gbh-yolov5 checkout 51ee042edf913bc1a6c3f25b6f296d6f8d6ef079
git clone --branch v7.0 --depth 1 https://github.com/ultralytics/yolov5 code/yolov5-v7.0

# ---- weights
mkdir -p weights
curl -L -o weights/yolov5s_v5.0.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
curl -L -o weights/yolov5s_v7.0.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
curl -L -o weights/yolov8s.pt      https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
sha256sum weights/*.pt > weights/SHA256SUMS   # commit SHA256SUMS only, then compare on Laptop B

# ---- P1 data
git clone https://github.com/CCNUZFW/PV-Multi-Defect datasets/raw/PV-Multi-Defect
git -C datasets/raw/PV-Multi-Defect checkout 2c4df0c1477032d6b0dd9372a517cdbfd6a09e09

# ---- P2 data (needs ROBOFLOW_API_KEY in the environment; never on disk)
envs/p2/bin/python -I code/repro/download_eldds.py --out datasets/raw/ELDDS1400C5
```

P1 conversion and frozen split (**run once, on one machine only, then commit the split files**):
```bash
envs/p1/bin/python -I code/repro/voc2yolo_pv.py \
  --src datasets/raw/PV-Multi-Defect --dst datasets/pv_multi_defect_yolo \
  --splits datasets/splits/pv_multi_defect --seed 0 --val-frac 0.2
```
What `voc2yolo_pv.py` must do (to be written in this step):
1. List every distinct `<name>` across all xml files and **fail on unknown names**. Classes, in paper order: `broken, hot_spot, black_border, scratch, no_electricity`.
2. Keep only images that have an xml (expected 1106) and print the jpgs that have no xml.
3. Convert VOC → YOLO txt using each xml's own `<size>`.
4. Stratified (by class presence) seeded 80/20 split, about 885/221. Write `train.txt`/`val.txt` of image paths. **If these files already exist, reuse them and don't re-split.**
5. Print per-class box counts and percentages, and compare them to P1 Fig. 5: hot_spot 49.09, scratch 36.62, black_border 6.02, no_electricity 4.28, broken 3.99 (4235 boxes total).

What `download_eldds.py` must do: read `os.environ["ROBOFLOW_API_KEY"]`, list the dataset's versions, and download the version **without Roboflow augmentation** in `yolov5` format. Print the image count per split and the class names, and write `datasets/raw/ELDDS1400C5/MANIFEST.txt` (version id, counts, classes; **no key**).

Checks:
- [ ] P1: 1106 images converted, the class list matches, and the class percentages are close to Fig. 5
- [ ] P1: split files created and **committed**
- [ ] P2: split counts recorded. Target: train 1187, test 190; if a `valid` split exists, note how it maps onto the paper's test set.
- [ ] P2: class names recorded (Fig. 2 has 6: Examined, ShortCircuitString, ShortCircuitCell(LowPowerCell), Crack, MicroCrack, DarkSpot/OtherError)
- [ ] `code/repro/draw_boxes.py`: draw 5 random images per dataset with their boxes into `results/sanity/` and check them by eye

---

## 6. P1 model configs and the architecture check

Files we add (committed; they live in `code/repro/p1/` and are copied into the clone by the run script):
- `pv_multi_defect.yaml`: `train:`/`val:` point to the split txt files, `nc: 5`, names in paper order.
- `hyp.gbh.yaml`: a copy of `hyp.scratch.yaml` with **only** `lr0: 0.001`.
- `yolov5s.yaml`: the ultralytics **v5.0** `models/yolov5s.yaml` (the fork doesn't ship it) with `nc: 5`.
- `gbh_paper.yaml`: **paper-strict** GBH-YOLOv5. Built from the authors' `models/yolov5s_pv.yaml`:
  - Focus stem and SPP (as in Fig. 1).
  - **BottleneckCSP in the backbone and the neck.** The released yaml has C3 in the neck; the paper says C3 is replaced by BottleneckCSP.
  - **GhostConv in place of Conv** ("Ghost convolution was used instead of the traditional convolution"), in the backbone and in the neck's 1×1 and downsampling convs.
  - **4 detect heads (P2–P5)** with the authors' anchor set: `[5,6,8,14,15,11]`, `[10,13,16,30,33,23]`, `[30,61,62,45,59,119]`, `[116,90,156,198,373,326]`.

**Mandatory check before any training:**
```bash
cd code/gbh-yolov5
for c in $PRJ/code/repro/p1/yolov5s.yaml $PRJ/code/repro/p1/gbh_paper.yaml models/yolov5s_pv.yaml; do
  $PRJ/envs/p1/bin/python models/yolo.py --cfg $c 2>&1 | grep -E "Model Summary|layers"; done
```
Compare with Table 6: YOLOv5s **224 layers / 7.06 M**, GBH **270 layers / 7.24 M**.
- If `gbh_paper.yaml` is close to 270 / 7.24: train it.
- If the **released** `yolov5s_pv.yaml` matches and the strict one doesn't: **stop and decide with the user**. Record the decision in §11.
- [ ] Layer and parameter table pasted into the Progress log

---

## 7. P2 model code and configs (we re-implement it; no released code)

Into `code/yolov5-v7.0/` (our edits are kept as a patch `code/repro/p2/yolov5_v7.0_p2.patch`, so the clone stays reproducible):
- `models/common.py`: `GAM_Attention(c1, c2, rate=4)`. Channel attention is a permute → MLP (c → c/r → c) → sigmoid. Spatial attention is 7×7 conv (c → c/r) → BN → ReLU → 7×7 conv (c/r → c) → BN → sigmoid. This follows Liu et al. 2021, arXiv:2112.05561, and P2 Eq. (1).
- `models/yolo.py`: `ASFF_Detect`. ASFF levels 1–3 (Liu et al. 2019, "Learning Spatial Fusion for Single-Shot Object Detection") fuse the three PAN outputs with softmax-normalised weights (P2 Eq. 5–6) before the standard YOLOv5 Detect convs. Register it in `parse_model` and in the places that use the `Detect` isinstance checks.
- `utils/general.py::non_max_suppression`: add `diou=False`. When it is True, replace `torchvision.ops.nms` with a DIoU-NMS (suppress if `IoU − ρ²/c² > iou_thres`). Expose `--diou-nms` in `val.py` and `detect.py`. It is used at inference only.
- `models/yolov5s.yaml` with `nc` = the ELDDS class count, and `models/yolov5s-gam-asff.yaml`.

**GFLOPs calibration (mandatory before training):** P2 only says GAM is in the "backbone and neck". Choose GAM positions (for example after SPPF and after each neck C3) so that, at 640 px:
- GAM only ≈ **21.4 G**
- GAM + ASFF ≈ **30.0 G**
- ASFF only ≈ **25.8 G**
- YOLOv5s ≈ 16 G; YOLOv8s ≈ 28.7 G

```bash
cd code/yolov5-v7.0 && $PRJ/envs/p2/bin/python models/yolo.py --cfg models/yolov5s-gam-asff.yaml
```
- [ ] GFLOPs table and chosen placement recorded in the Progress log and §11

---

## 8. Smoke tests (each model, 3 epochs)

The goal is to find the **largest physical batch that fits**, the seconds per epoch, and peak VRAM. Start from the paper batch and halve it until there's no CUDA OOM.

```bash
# P1 (in code/gbh-yolov5)
$PRJ/envs/p1/bin/python train.py --cfg $PRJ/code/repro/p1/gbh_paper.yaml \
  --weights $PRJ/weights/yolov5s_v5.0.pt --data $PRJ/code/repro/p1/pv_multi_defect.yaml \
  --hyp $PRJ/code/repro/p1/hyp.gbh.yaml --adam --epochs 3 --batch-size 16 --img-size 960 960 \
  --workers 2 --project $PRJ/results/smoke --name p1_gbh
# repeat with yolov5s.yaml

# P2 YOLOv5 (in code/yolov5-v7.0)
$PRJ/envs/p2/bin/python train.py --cfg models/yolov5s-gam-asff.yaml --weights $PRJ/weights/yolov5s_v7.0.pt \
  --data $PRJ/code/repro/p2/eldds.yaml --optimizer Adam --epochs 3 --batch-size 32 --img 640 \
  --seed 0 --workers 2 --project $PRJ/results/smoke --name p2_v5_improved
# repeat with models/yolov5s.yaml

# P2 YOLOv8
$PRJ/envs/p2/bin/yolo detect train model=$PRJ/weights/yolov8s.pt data=$PRJ/code/repro/p2/eldds.yaml \
  optimizer=Adam lr0=0.01 epochs=3 batch=32 imgsz=640 close_mosaic=10 seed=0 deterministic=True \
  workers=2 project=$PRJ/results/smoke name=p2_v8s
```
For each model, record in the Progress log: machine, max batch, s/epoch, peak VRAM (`nvidia-smi --query-gpu=memory.used --format=csv -l 5`), and mAP after 3 epochs (must be > 0).

Then project the full-run times (P1: 500 epochs, P2: 100 epochs) and decide which runs go to which laptop. P1 goes to Laptop B if it is more than about 1 day per model on A.

- [ ] Smoke tests done; batch per model per machine recorded

---

## 9. Full training

Always run inside `tmux` (or with `nohup ... &> log &`) so a closed terminal doesn't kill the run. Put each run in `results/p1/<name>` or `results/p2/<name>`.

| Run | Codebase / env | Key args |
|---|---|---|
| `p1_yolov5s` | gbh-yolov5 / p1 | `--cfg yolov5s.yaml --adam --epochs 500 --img-size 960 960 --hyp hyp.gbh.yaml --batch-size <B>` |
| `p1_gbh` | gbh-yolov5 / p1 | `--cfg gbh_paper.yaml`, otherwise the same |
| `p2_v5s` | yolov5-v7.0 / p2 | `--cfg models/yolov5s.yaml --optimizer Adam --epochs 100 --img 640 --seed 0 --batch-size <B>` |
| `p2_v5_improved` | yolov5-v7.0 / p2 | `--cfg models/yolov5s-gam-asff.yaml`, otherwise the same (DIoU-NMS is applied at val, step 10) |
| `p2_v8s` | p2 (ultralytics) | `optimizer=Adam lr0=0.01 epochs=100 imgsz=640 close_mosaic=10 seed=0 deterministic=True batch=<B>` |

**Resume after an interruption:**
- P1 / P2-v5: `train.py --resume results/<p>/<run>/weights/last.pt`
- v8: `yolo detect train resume model=results/p2/p2_v8s/weights/last.pt`

**Moving a run to the other laptop:** copy the whole `results/<p>/<run>/` dir, and resume there with the **same physical batch** if it fits. If it doesn't, log the change.

- [ ] p1_yolov5s  - [ ] p1_gbh  - [ ] p2_v5s  - [ ] p2_v5_improved  - [ ] p2_v8s

---

## 10. Evaluation

**P1** (val split, 960 px, `best.pt`): overall and per-class mAP@.5, P and R, plus per-image time.
```bash
$PRJ/envs/p1/bin/python test.py --weights $PRJ/results/p1/<run>/weights/best.pt \
  --data $PRJ/code/repro/p1/pv_multi_defect.yaml --img-size 960 --batch-size 1 --task val --verbose
```
**P2** (test split, 640 px): each model **without** and **with** TTA.
```bash
$PRJ/envs/p2/bin/python val.py --weights <best.pt> --data $PRJ/code/repro/p2/eldds.yaml --img 640 --task test [--augment] [--diou-nms]
$PRJ/envs/p2/bin/yolo detect val model=<best.pt> data=$PRJ/code/repro/p2/eldds.yaml split=test imgsz=640 [augment=True]
```
- `--diou-nms` applies to the improved model only.
- Save the confusion matrix at conf 0.2 / IoU 0.5 (Fig. 16), the PR curve and the F1 curve.

`code/repro/summarize.py` collects everything into `results/summary.csv` and `results/summary.md` (paper vs. ours vs. Δ, plus machine and batch for each run). **Commit `summary.*`**, but don't commit the weights.

- [ ] P1 evaluated  - [ ] P2 evaluated (×2 TTA)  - [ ] summary committed

---

## 11. Deviations from the papers (keep this up to date, it goes into our paper)

1. Physical batch is smaller than the paper's. The effective batch stays 64 through `nbs` accumulation; BN statistics and warm-up iterations differ. (Fill in per run.)
2. GPUs are RTX 2050 / 3050 instead of RTX 3090 (P1) / Tesla P100 (P2).
3. P2 env uses torch 1.11.0 **+cu113**, because there is no cu111 wheel for 1.11.
4. P1 train/val split is re-created (stratified by rarest class present, seed 0, exactly 886/222), since the original split is unpublished and unseeded. All **1108** images are used, as in the paper: the 2 jpgs without an xml (`img3059`, `img9513`, defect-free) are background images with empty label files. (The "1113 jpg" in §12 counted the 5 README illustrations `tf1-5.jpg`.) Box counts: 3981 vs Fig. 5's 4235; shares broken 2.46 / hot_spot 52.22 / black_border 6.43 / scratch 34.34 / no_electricity 4.55 % vs Fig. 5's 3.99 / 49.09 / 6.02 / 36.62 / 4.28.
5. P1 authors' `prepare_data.py` and `data/voc_pv.yaml` are not used: they contain 8 classes from an unrelated project (`write, read, lookup, ...`).
6. P1 architecture comes from the paper text. The released `models/yolov5s_pv.yaml` (C3 in the neck, GhostConv in the backbone only) is documented, and the outcome of the step-6 check is: _TBD_.
7. P2 GAM and ASFF are re-implemented by us (`code/repro/p2/yolov5_v7.0_p2.patch`, yamls from `code/repro/p2/make_yamls.py`). GAM placement calibrated to Table 2 GFLOPs @640, nc=6: **grouped GAM (groups=4, channel shuffle, rate 4) after SPPF (backbone) and after each of the 3 neck outputs (P3/P4/P5)**. Ours vs paper: YOLOv5s 16.0 / 16; GAM-only 21.6 / 21.4; ASFF-only 24.4 / 25.8; GAM+ASFF **30.0 / 30.0**; YOLOv8s 28.8 (nc=80) / 28.7. A single full (ungrouped) GAM anywhere gives 21.2 / 29.7 and only sits in one place, contradicting "backbone and neck". Paper's own numbers aren't additive (5.4 + 9.8 ≠ 14.0), so ASFF-only can't be matched at the same time as GAM+ASFF. ASFF follows Liu et al. 2019 (maxpool+3×3 s2 conv down, 1×1 conv + nearest up, 1×1 weight convs with 16 ch, softmax, 3×3 expand), every level fusing the unfused PAN maps.
8. P2 ultralytics version: 8.0.99 (the paper doesn't state it; this was the last release before submission).
9. One run per model (the papers report ±95 % CI).
10. P2 dataset: Roboflow has a single version (v1, no augmentation, stretched to 640×640): train 1188 / test 190, 11 class names. Ids 0–5 are the paper's 6 classes (Examined + 5 defects; P2 says "five categories of defects" and "six classes"). Ids 6–10 (Contamination, DarkSpot, Examined, MicroCrack, Scratch, 25 boxes) occur only in 2 train images (`panel_04…`, `panel_10…`) that have no paper-class label; those 2 images are **excluded** → **1186 train / 190 test** (paper 1187 / 190), `nc=6`. No val split exists, so val = test for training-time best.pt selection, as in the paper.
11. P2 DIoU: the paper's section shows both DIoU-NMS and DIoU-loss equations, but contribution 2 says "replacement of the default NMS with DIoU". We implement DIoU-NMS at inference only (CIoU training loss unchanged).
12. ultralytics 8.0.99 silently ignores `yolo val augment=True` (the validator calls `model(img)` without `augment`). YOLOv8 TTA numbers use our wrapper that forces augmented inference.
13. GBH repo `train.py` @51ee042 has a leading space on line 1 (IndentationError) and can't run as published. Fixed by `code/repro/p1/gbh_51ee042_fix.patch` (whitespace only).
14. Windows (Laptop B): the NVIDIA driver silently spills CUDA memory into system RAM ("Sysmem Fallback"), so an over-large batch doesn't OOM, it just becomes very slow. Batch fit is judged by speed and VRAM, not by OOM.
16. P2 physical batch: **16 for all three models** (paper 32). Measured on the 6 GB card: batch 32 spills for every model and crashed YOLOv5s (`CUDA error: resource already mapped`). 16 is the largest divisor of 64 that fits all three. With `nbs=64` accumulation, each optimizer step sees 64 images, the same as the paper's 32 × 2, with identical weight-decay scaling (0.0005 × 16·4/64 = 0.0005). Only difference: BN batch statistics (16 vs 32). Warm-up is identical in epochs: both codebases use max(3 epochs, 100 iters), which is 3 epochs for the paper (38 it/ep) and for us (75 it/ep). (P1 differs: GBH uses max(3 epochs, **1000** iters), i.e. ~18 epochs at batch 16 vs ~4.5 at batch 4; to note when P1 runs.) GAM+ASFF at batch 16 sits at 5.7 GB reserved and spills slightly (epoch time 1:54 vs 0:58); A first full run with `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` made it worse (9.35 GB reserved, ~3.1 min/ep) and was aborted at epoch 7 (`results/p2/aborted_v5_improved_allocconf/`); the setting was removed and the run restarted from scratch (no numerical effect either way).
17. P2 GAM+ASFF loads only 198/535 tensors from COCO `yolov5s.pt`: yolov5 matches by layer index + shape, and the inserted GAM layers shift every neck/head index, so the neck/head start from scratch. Any yaml-based yolov5 GAM model behaves this way (likely the authors' too; unverifiable).
18. DIoU-NMS is a Python greedy loop (~90 ms/img vs ~6 ms for torchvision NMS). It is exempted from v7's NMS time limit, which otherwise silently drops all detections for the rest of a batch (seen: mAP@.5 0.184 instead of 0.326). Computed in fp32 (fp16 overflows areas > 65504). With distance term off it reproduces torchvision NMS exactly (tested on real predictions). Inference timings are therefore not comparable to P2's.
19. Windows: YOLOv8s uses `workers=1` (ultralytics 8.0.99's trainer commits ~16 GB and its val loader runs 2× workers, all persistent: workers=2 meant 6 worker processes, exhausted commit at epoch 3 and Claude Code's background jobs were reaped; run restarted from scratch, aborted run kept in `results/p2/aborted_v8s_oom/`). Worker count is not a paper parameter. YOLOv5 runs: dataloader `workers=2` (4 exceeds the commit limit: trainer ~10 GB + ~3.5 GB/worker → misleading CUDA OOM). yolov5 v7's font download (`Arial.ttf`) fails on a 308 redirect under Python 3.8, so the fonts were fetched with curl into `%APPDATA%/Ultralytics` (plot labels only).
15. Envs: `roboflow` is **not** in envs/p2 (it pins opencv-python-headless 4.10, which clashes with opencv-python). The download runs in a throwaway `uv run --with roboflow` env. `ipython` added to p2 (imported by yolov5 v7.0 `models/common.py`). Exact package lists: `code/repro/env/p{1,2}.freeze.txt`.

---

## 12. Facts gathered during planning (2026-10-17)

- The GBH repo default `train.py` args are `--epochs 300`, `--batch-size 16`, `--img-size 960 960`, SGD unless `--adam`, `hyp.scratch.yaml` lr0 0.01. **We override** these with `--adam --epochs 500` and lr0 0.001, per the paper.
- `GhostConv` lives in the GBH repo's `models/experimental.py` (not `common.py`), and `parse_model` handles it. BottleneckCSP is in `common.py`.
- The GBH README points to Baidu-pan weights (code `lmn8`). We use the official v5.0 `yolov5s.pt` COCO weights instead, because the paper says "pretrained weights based on the COCO dataset".
- PV-Multi-Defect: 1113 jpg / 1106 xml, 600 × 600 grayscale, VOC xml, about 43 MB. All data commits date from 2023-01-15.
- ELDDS1400C5: on Roboflow, 1378 images. Roboflow blocks scripted page fetches, so download through the API with a key.
- P2's 3 tables are images in the PDF; the numbers in §3 were read off the rendered pages.

---

## 13. Progress log

| Date | Machine | Step | Result / notes |
|---|---|---|---|
| 2026-10-17 | Laptop A (RTX 2050 4 GB, 7 GB RAM) | 0 Planning | Repo initialised. Papers, runbook and `.gitignore` committed. Nothing executed yet. |
| 2026-10-17 | Laptop B (RTX 3050 Laptop 6 GB, 15 GB RAM, Ryzen 7 7840HS, Win 11, driver 595.97) | 4 Envs | uv 0.12.0, CPython 3.8.20. p1: torch 1.8.1+cu111, cuDNN 8005. p2: torch 1.11.0+cu113, cuDNN 8200, ultralytics 8.0.99. Both see the GPU. Windows python path is `envs/pX/Scripts/python.exe`. Freeze lists in `code/repro/env/`. |
| 2026-10-17 | Laptop B | 5 Code/data | GBH @51ee042, yolov5 v7.0 @915bbf2, PV-Multi-Defect @2c4df0c. Weights SHA256 in `weights/SHA256SUMS` (first v7.0 download was truncated, re-downloaded). P1: 1108 imgs → split 886/222 committed (`datasets/splits/pv_multi_defect`). P2: ELDDS v1 → 1186/190, 6 classes, split committed (`datasets/splits/eldds1400c5`). Sanity images in `results/sanity/` look correct. |
| 2026-10-17 | Laptop B | 6 P1 arch check | Fused counts (test.py fuses; Table 6 is "on the test set"): YOLOv5s **224 / 7.065 M = paper exactly**. BCSP in backbone only (neck C3) = **228 / 7.153 M = paper YOLOv5-1 exactly**. Released `yolov5s_pv.yaml` 290 / 7.04 M; paper-strict `gbh_paper.yaml` 320 / 6.63 M; paper GBH 270 / 7.24 M. A 32-way grid over Conv/Ghost × C3/BCSP × 3/4 heads hits neither 270 nor 290 / 7.72 (YOLOv5-2). **Awaiting user decision** on which GBH yaml to train. |
| 2026-10-17 | Laptop B | 7 P2 GFLOPs | See §11 item 7. Chosen: `code/repro/p2/yolov5s-gam-asff.yaml` = 30.0 G. |
| 2026-10-17 | Laptop B | Decisions (user) | P1 GBH = **released** `yolov5s_pv.yaml`. Train P2 now; **P1 deferred** (long runs). Exact paper params wherever possible; envs moved to locked uv projects (`code/repro/env/p{1,2}/`), P2 Python now 3.8.8. |
| 2026-10-17 | Laptop B | 8 Batch probe | Synthetic AMP fwd+bwd (`code/repro/mem_probe.py`), max batch without spill: P1 GBH@960 **6**, P1 YOLOv5s@960 **12**, P2 YOLOv5s **24**, P2 GAM+ASFF **16**, P2 YOLOv8s **24** (paper: 16 / 16 / 32 / 32 / 32). |
| 2026-10-17 | Laptop B | 8 Smoke (P2, 3 ep, batch 16, workers 2) | v5s 45 s/ep, mAP@.5 → 0.376 · GAM+ASFF 1–2 min/ep → 0.295 · v8s 55 s/ep → 0.522. Batch 32 crashed. Eval pipeline checked (2 DIoU-NMS bugs fixed, §11 item 18; v8 TTA wrapper 0.522 → 0.560). |
| 2026-10-17 15:12 | Laptop B | 9 Full P2 | Started `run_p2.sh full v5s v5_improved v8s` (100 ep each, batch 16 × 4). |
| 2026-10-17 | Laptop B | Kaggle (dropped) | Tried running P1 (and P2 at batch 32) on Kaggle via API: first sessions had no GPU/internet (account not phone-verified), then P100 requests were substituted by T4 x2 or queued; user dropped Kaggle. Tooling kept in `code/repro/kaggle/` (unused). |
| 2026-10-17 | Laptop B | Decision (user) | **P1 runs on an RTX 4070 laptop (8 GB)**: est. YOLOv5s at exact batch 16, GBH 8 × 8 (see CHECKLIST §7). Laptop B finishes P2 + P2 eval only (`run_queue_local.sh`). |
| 2026-10-17 21:26 | Laptop B | 9 Full P2 done | v5s 0.65 h, GAM+ASFF 3.42 h (after one aborted start), v8s 1.38 h (after one aborted start, workers=1). |
| 2026-10-17 21:33 | Laptop B | 10 P2 eval | No TTA mAP@.5 (ours/paper): v5s 0.754/0.739, improved (DIoU-NMS) 0.757/0.762, v8s 0.764/0.765. TTA@640: 0.766/0.766, 0.758/0.777, 0.753/0.775. TTA@832 (sensitivity): 0.774, 0.773, 0.761. See `results/summary.md`, CHECKLIST §6. |

---

## P3: Deitsch et al. 2019, *Automatic classification of defective PV module cells in EL images* (Solar Energy 185)
Paper: `papers/el-auto-cnn-svg.pdf`. **No code released** (only the dataset). Scope (user, 2026-10-17): the two reported
models, **CNN** (VGG-19 deep regression) and **best SVM** (KAZE/VGG + VLAD + linear SVM). No ablations / other SVM variants.

### Settings (paper → ours)
| | Paper | Ours |
|---|---|---|
| Data | ELPV, 2624 cells 300×300, 75/25 stratified, 1968/656 (split unpublished) | `zae-bayern/elpv-dataset` @ `93e82ae` (labels unchanged since 2018-03-07); split recreated with **exactly Table 3's counts per type × label**, seed 0 (`datasets/splits/elpv`, `code/repro/p3/elpv_split.py`) |
| Labels / weights | p ∈ {0, ⅓, ⅔, 1}; defective iff p > 0; confidence weights 1/0.33/0.67/1; class weights c_j = S/(2n_j) | same (`elpv_common.py`) |
| CNN | ImageNet VGG-19 conv + GAP + FC4096 + FC2048 + 1 neuron, MSE; aug: scale ≤2 %, rot ≤3°, shift ≤2 %, h/v flips, 90° rotations; stage 1 FC only Adam 1e-3 (0.9/0.999/1e-8); stage 2 all layers SGD 5e-4 m 0.9; batch 16; ≤100 epochs each; Keras 2.0 + TF 1.4 | same, **batch 16 physical** (fits); full 100 + 100 epochs, final weights; **TF 2.10.1 tf.keras** (Keras ImageNet VGG-19 weights), py 3.10.11, CUDA 11.8 runtime + cuDNN 8.9.5 pip wheels (`code/repro/env/p3cnn`) |
| SVM | KAZE detector + VGG descriptor (120-d), OpenCV 3.3.1 defaults, no masking; VLAD (mini-batch k-means), α 0.5, ×5 codebooks, joint PCA whitening, L2; linear SVM (LIBLINEAR, L2, squared hinge), C ∈ 10^{−2..6} by inner 5-fold CV on macro F1; weighted | same with **OpenCV 3.4.18 contrib**, scikit-learn 0.22.2 (LIBLINEAR, MiniBatchKMeans, PCA), py 3.8.20 (`code/repro/env/p3svm`) |
| Targets | CNN acc **88.42 %**, F1 **88.39 %**; SVM acc **82.44 %**, F1 **82.52 %**; SVM KAZE/VGG AUC 88.51 % | — |

### P3 deviations
P3-1. Framework: TF 2.10.1 / tf.keras instead of Keras 2.0 + TF 1.4. Same ImageNet VGG-19 weights file. Exact versions checked 2026-10-17 and **not feasible**: TF 1.4 wheels exist only for Python 3.5/3.6, which uv 0.12 cannot provide; TF 1.4 needs CUDA 8 / cuDNN 6, which cannot run on Ampere/Ada GPUs (CPU-only training ≈ 1–2 days); opencv-contrib-python 3.3.x has been removed from PyPI. User decision: keep TF 2.10 (GPU, uv) and OpenCV 3.4.18, document the gap.
P3-2. CNN output activation not stated in the paper. **Linear** (pure regression). A sigmoid output saturated and never trained (stage-1 loss frozen at 0.5626 for 3 epochs vs linear 9.06 → 0.116; training loss only, no test data).
P3-3. "90° rotated samples are additionally included": implemented as a random 90° rotation with p = 0.5, so an epoch stays 1968 samples (paper: 196800 = 1968 × 100 augmented samples).
P3-4. "Maximum of 100 epochs" per stage: we run the full 100 + 100 and evaluate the final weights once (user decision; no early stopping, no test peeking).
P3-5. Class-balancing weights c_j (Eq. 5) also applied to the CNN as per-sample loss weights (paper describes them for the training set in general).
P3-6. SVM VLAD codebook size and k-means subset size not stated → K = 100, 100k descriptors per codebook (seeds 0–4).
P3-7. SVM PCA-whitening output dimension not stated. Full-rank whitening (1968 samples, 60000-d) gave inner-CV F1 ≈ 0.52 (chance); the dimension is therefore selected with C by the inner CV from {32, 64, 128, 256, 512} (training data only). PCA/VLAD fitted once on the full training set (unsupervised), CV selects (dim, C).
P3-8. OpenCV 3.4.18 instead of 3.3.1 (no 3.3.1 wheels for uv-installable Pythons). KAZE defaults find 0 keypoints on 83/1968 training cells (mostly mono) → empty VLAD for those.
P3-9. Evaluation: binary (defective iff p > 0), CNN threshold 0.5, SVM decision 0. We report confidence-weighted accuracy/macro-F1 (Table 2 caption: weights are used "for the evaluation") and unweighted, plus ROC AUC, for all/mono/poly. The paper doesn't spell out which variant its 88.42 % / 82.44 % are.
P3-10. SVM result is far below the paper (see below). Training-split-only check of the unstated VLAD K (inner CV, PCA dim and C re-selected per K): K=16 0.728, 32 0.734, 64 0.744, 100 0.737 (K=256 not finished: run killed by low memory). K is therefore not the cause; K = 100 kept. No further tuning (would drift from the paper).
P3-11. CNN run 1 was killed at stage-1 epoch 32 (system low on memory: ran concurrently with the SVM K check). Restarted from scratch alone, with per-epoch checkpoints (weights + optimizer state + augmentation RNG; resume tested: optimizer iterations restored, loss continues). Killed run kept in `results/p3/aborted_cnn_oom_ep32/`.

### P3 progress log
| Date | Machine | Step | Result |
|---|---|---|---|
| 2026-10-17 | Laptop B | env | p3cnn: py 3.10.11, TF 2.10.1 (built CUDA 11.2/cuDNN 8), CUDA 11.8 + cuDNN 8.9.5 wheels, GPU OK. p3svm: py 3.8.20, OpenCV 3.4.18, sklearn 0.22.2.post1. |
| 2026-10-17 | Laptop B | data | ELPV @93e82ae, split 1968/656 = Table 3 exactly (committed `datasets/splits/elpv`). |
| 2026-10-17 | Laptop B | CNN smoke | batch 16 fits physically (no accumulation); stage 1 ~41 s/ep, stage 2 ~111 s/ep. Sigmoid output did not train → linear (P3-2). |
| 2026-10-17 22:48 | Laptop B | **SVM done** | KAZE/VGG (mean 87 kp/cell, 83 train cells with 0), VLAD K=100 ×5, PCA-whitened to 32 dims, C = 0.1 (inner-CV F1 0.737). **Test: weighted acc 69.70 %, weighted macro-F1 66.32 %, AUC_w 74.88 %** (unweighted acc 66.16, F1 63.72, AUC 71.33); mono acc_w 66.82, poly 71.56. Paper: acc 82.44, F1 82.52, AUC 88.51. |
| 2026-10-17 23:16 | Laptop B | CNN full | started alone, 100 + 100 epochs, batch 16, checkpointed. |
