#!/usr/bin/env bash
# One-shot setup + run for the RTX 4050 machine: installs envs, fetches missing
# code/weights/datasets, then trains the 4 remaining models and evals DIoU-only.
#
# Prerequisites (must be satisfied before running):
#   1. export ROBOFLOW_API_KEY=<your_key>   (never write the key to disk)
#   2. uv.exe reachable (it is in PATH on this machine)
#   3. git with Git LFS already pulled (models/ are LFS objects, not needed here)
#
# Remaining models to train / eval:
#   P2 v5_gam   — GAM-only ablation, yolov5s-gam.yaml, no DIoU-NMS at eval
#   P2 v5_asff  — ASFF-only ablation, yolov5s-asff.yaml, no DIoU-NMS at eval
#   P2 v5s_diou — eval-only; re-evals existing v5s weights with --diou-nms
#   P1 yolov5_2 — Conv backbone + 4-head (Table 4-6 in paper), same as GBH but Conv not GhostConv
#
# Expected runtimes on RTX 4050 6 GB:
#   P2 v5_gam / v5_asff : ~15 min each (100 ep, 640 px, batch 16)
#   P2 v5s_diou eval    : ~2 min
#   P1 yolov5_2         : ~2-3 h (500 ep, 960 px, batch 4)
#
# usage: bash code/repro/setup_run_remaining.sh   (from the repo root)
set -euo pipefail

PRJ="$(cd "$(dirname "$0")/../.." && (pwd -W 2>/dev/null || pwd))"
cd "$PRJ"

Q="$PRJ/results/remaining.log"
mkdir -p "$PRJ/results"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

# ---------------------------------------------------------------------------
# 0. Guard: ROBOFLOW_API_KEY must be set in the environment
# ---------------------------------------------------------------------------
if [ -z "${ROBOFLOW_API_KEY:-}" ]; then
  echo "ERROR: ROBOFLOW_API_KEY not set."
  echo "  export ROBOFLOW_API_KEY=<your_key>   then re-run."
  exit 1
fi
say "setup_run_remaining start"

# ---------------------------------------------------------------------------
# 1. Python environments (uv sync --frozen)
# ---------------------------------------------------------------------------
say "=== STEP 1: Python environments ==="

if [ ! -f "$PRJ/envs/p1/Scripts/python.exe" ] && [ ! -f "$PRJ/envs/p1/bin/python" ]; then
  say "  building envs/p1 ..."
  UV_PROJECT_ENVIRONMENT="$PRJ/envs/p1" uv sync --project "$PRJ/code/repro/env/p1" --frozen
  say "  envs/p1 done"
else
  say "  envs/p1 already present; skipping"
fi

if [ ! -f "$PRJ/envs/p2/Scripts/python.exe" ] && [ ! -f "$PRJ/envs/p2/bin/python" ]; then
  say "  building envs/p2 ..."
  UV_PROJECT_ENVIRONMENT="$PRJ/envs/p2" uv sync --project "$PRJ/code/repro/env/p2" --frozen
  say "  envs/p2 done"
else
  say "  envs/p2 already present; skipping"
fi

PY1="$PRJ/envs/p1/Scripts/python.exe"; [ -x "$PY1" ] || PY1="$PRJ/envs/p1/bin/python"
PY2="$PRJ/envs/p2/Scripts/python.exe"; [ -x "$PY2" ] || PY2="$PRJ/envs/p2/bin/python"

# ---------------------------------------------------------------------------
# 2. Clone yolov5 v7.0 codebase (P2)
# ---------------------------------------------------------------------------
say "=== STEP 2: yolov5 v7.0 codebase ==="

if [ ! -d "$PRJ/code/yolov5-v7.0/.git" ]; then
  say "  cloning yolov5 v7.0 ..."
  git clone --branch v7.0 --depth 1 https://github.com/ultralytics/yolov5 "$PRJ/code/yolov5-v7.0"
  say "  clone done: $(git -C "$PRJ/code/yolov5-v7.0" rev-parse HEAD)"
else
  say "  code/yolov5-v7.0 already present; skipping clone"
fi

# Apply P2 patch (idempotent: git apply --check first)
PATCH="$PRJ/code/repro/p2/yolov5_v7.0_p2.patch"
if git -C "$PRJ/code/yolov5-v7.0" apply --check "$PATCH" 2>/dev/null; then
  say "  applying P2 patch ..."
  git -C "$PRJ/code/yolov5-v7.0" apply "$PATCH"
  say "  patch applied"
else
  say "  P2 patch already applied (or inapplicable — check manually if errors occur)"
fi

# ---------------------------------------------------------------------------
# 3. Download missing weights
# ---------------------------------------------------------------------------
say "=== STEP 3: weights ==="

if [ ! -f "$PRJ/weights/yolov5s_v7.0.pt" ]; then
  say "  downloading yolov5s_v7.0.pt ..."
  curl -fL -o "$PRJ/weights/yolov5s_v7.0.pt" \
    https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
  say "  yolov5s_v7.0.pt downloaded ($(wc -c < "$PRJ/weights/yolov5s_v7.0.pt") bytes)"
else
  say "  weights/yolov5s_v7.0.pt already present; skipping"
fi

# ---------------------------------------------------------------------------
# 4. P1 dataset: PV-Multi-Defect -> pv_multi_defect_yolo
# ---------------------------------------------------------------------------
say "=== STEP 4: P1 dataset ==="

if [ ! -d "$PRJ/datasets/raw/PV-Multi-Defect/.git" ]; then
  say "  cloning PV-Multi-Defect ..."
  git clone https://github.com/CCNUZFW/PV-Multi-Defect "$PRJ/datasets/raw/PV-Multi-Defect"
  git -C "$PRJ/datasets/raw/PV-Multi-Defect" checkout 2c4df0c1477032d6b0dd9372a517cdbfd6a09e09
  say "  PV-Multi-Defect cloned @ $(git -C "$PRJ/datasets/raw/PV-Multi-Defect" rev-parse HEAD)"
else
  say "  datasets/raw/PV-Multi-Defect already present; skipping clone"
fi

if [ ! -f "$PRJ/datasets/pv_multi_defect_yolo/data.yaml" ]; then
  say "  running voc2yolo_pv.py (reuses frozen splits, no re-split) ..."
  "$PY1" -I "$PRJ/code/repro/voc2yolo_pv.py" \
    --src "$PRJ/datasets/raw/PV-Multi-Defect" \
    --dst "$PRJ/datasets/pv_multi_defect_yolo" \
    --splits "$PRJ/datasets/splits/pv_multi_defect" \
    --seed 0 --n-val 222
  say "  pv_multi_defect_yolo prepared"
else
  say "  datasets/pv_multi_defect_yolo already prepared; skipping"
fi

# ---------------------------------------------------------------------------
# 5. P2 dataset: ELDDS1400C5 download + prep
# ---------------------------------------------------------------------------
say "=== STEP 5: P2 dataset ==="

# Check if download is needed (MANIFEST.txt only = not downloaded)
ELDDS_TRAIN="$PRJ/datasets/raw/ELDDS1400C5/train/images"
if [ ! -d "$ELDDS_TRAIN" ]; then
  say "  downloading ELDDS1400C5 from Roboflow (key from env) ..."
  # Roboflow's package needs a throwaway env to avoid pinning conflicts with envs/p2
  uv run --no-project --python 3.10 --with roboflow \
    python -I "$PRJ/code/repro/download_eldds.py" \
    --out "$PRJ/datasets/raw/ELDDS1400C5"
  say "  ELDDS1400C5 downloaded"
else
  say "  datasets/raw/ELDDS1400C5 already downloaded; skipping"
fi

if [ ! -f "$PRJ/datasets/eldds_yolo/data.yaml" ]; then
  say "  running prep_eldds.py ..."
  "$PY2" -I "$PRJ/code/repro/prep_eldds.py" \
    --src "$PRJ/datasets/raw/ELDDS1400C5" \
    --dst "$PRJ/datasets/eldds_yolo" \
    --splits "$PRJ/datasets/splits/eldds1400c5"
  say "  eldds_yolo prepared"
else
  say "  datasets/eldds_yolo already prepared; skipping"
fi

# ---------------------------------------------------------------------------
# 6. Training: P2 ablations (v5_gam, v5_asff) — ~15 min each
# ---------------------------------------------------------------------------
say "=== STEP 6: P2 training (v5_gam, v5_asff) ==="

for RUN in v5_gam v5_asff; do
  DONE_MARKER="$PRJ/results/p2/p2_$RUN/weights/best.pt"
  if [ -f "$DONE_MARKER" ]; then
    say "  p2_$RUN already trained ($DONE_MARKER); skipping"
    continue
  fi
  say "  training p2_$RUN ..."
  bash "$PRJ/code/repro/p2/run_p2.sh" full "$RUN"
  say "  p2_$RUN done"
done

# ---------------------------------------------------------------------------
# 7. Eval: P2 v5s_diou (no training — re-evals existing v5s weights with DIoU-NMS)
# ---------------------------------------------------------------------------
say "=== STEP 7: P2 eval (v5s_diou) ==="

DIOU_LOG="$PRJ/results/p2/eval/v5s_diou_notta.log"
if [ -f "$DIOU_LOG" ]; then
  say "  v5s_diou_notta already evaluated; skipping"
else
  say "  evaluating v5s_diou (DIoU-NMS on existing v5s weights) ..."
  bash "$PRJ/code/repro/p2/eval_p2.sh" v5s_diou
  say "  v5s_diou eval done"
fi

# ---------------------------------------------------------------------------
# 8. Training: P1 yolov5_2 (~2-3 h, batch 4 for 6 GB GPU)
# ---------------------------------------------------------------------------
say "=== STEP 8: P1 yolov5_2 training ==="

P1_DONE="$PRJ/results/p1/p1_yolov5_2/weights/best.pt"
if [ -f "$P1_DONE" ]; then
  say "  p1_yolov5_2 already trained; skipping"
else
  say "  training p1_yolov5_2 (500 ep, 960 px, batch ${BATCH_YOLOV52:-4}, ~2-3 h) ..."
  bash "$PRJ/code/repro/p1/run_p1.sh" full yolov5_2
  say "  p1_yolov5_2 done"
fi

# ---------------------------------------------------------------------------
# 9. Eval: full P2 suite (all 6 variants) + P1 eval summary
# ---------------------------------------------------------------------------
say "=== STEP 9: Full P2 eval ==="
bash "$PRJ/code/repro/p2/eval_p2.sh"

say "=== DONE: all remaining models trained and evaluated ==="
say "Log: $Q"
say "Results: $PRJ/results/p2/eval/  and  $PRJ/results/p1/"
