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
4. P1 train/val split is re-created (stratified, seed 0, 80/20), since the original split is unpublished and unseeded. 1106 annotated images instead of 1108.
5. P1 authors' `prepare_data.py` and `data/voc_pv.yaml` are not used: they contain 8 classes from an unrelated project (`write, read, lookup, ...`).
6. P1 architecture comes from the paper text. The released `models/yolov5s_pv.yaml` (C3 in the neck, GhostConv in the backbone only) is documented, and the outcome of the step-6 check is: _TBD_.
7. P2 GAM and ASFF are re-implemented by us, and the GAM placement is calibrated to the Table 2 GFLOPs: _TBD_.
8. P2 ultralytics version: 8.0.99 (the paper doesn't state it; this was the last release before submission).
9. One run per model (the papers report ±95 % CI).

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
