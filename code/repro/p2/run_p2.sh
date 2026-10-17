#!/usr/bin/env bash
# P2 training runs (Mazen et al. 2023), exactly the paper's settings except physical batch (init.md §11 item 1).
# Paper: Adam, lr0 0.01, batch 32, 100 epochs, 640x640, yolov5s.pt / yolov8s.pt, YOLOv8 close_mosaic 10.
# Batch: 16 (32 doesn't fit in 6 GB; spills and crashes). Both codebases accumulate to nbs=64, so every optimizer
# step sees 64 images, as the paper's 32 x 2 did, with the same weight-decay scaling.
# workers=2: on Windows each dataloader worker commits ~3.5 GB and the trainer ~10 GB; 4 workers exceed the commit limit
# (cudaMalloc then fails with a misleading OOM). Worker count does not change results.
#
# usage: bash code/repro/p2/run_p2.sh <smoke|full> <v5s|v5_improved|v8s>...   (from the repo root)
set -euo pipefail
PRJ="$(cd "$(dirname "$0")/../../.." && (pwd -W 2>/dev/null || pwd))"  # C:/... on Windows
MODE="$1"; shift
case "$MODE" in smoke) EPOCHS=3; OUT="$PRJ/results/smoke";; full) EPOCHS=100; OUT="$PRJ/results/p2";; *) echo "mode?"; exit 2;; esac
BATCH=16
PY="$PRJ/envs/p2/Scripts/python.exe"; [ -x "$PY" ] || PY="$PRJ/envs/p2/bin/python"
DATA="$PRJ/datasets/eldds_yolo/data.yaml"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"

for RUN in "$@"; do
  NAME="p2_$RUN"
  LOG="$OUT/$NAME.log"
  echo "[$(date '+%F %T')] start $NAME ($MODE, $EPOCHS ep, batch $BATCH)" | tee -a "$OUT/runs.txt"
  rc=0
  case "$RUN" in
    v5s|v5_improved)
      CFG="$PRJ/code/repro/p2/yolov5s.yaml"; [ "$RUN" = v5_improved ] && CFG="$PRJ/code/repro/p2/yolov5s-gam-asff.yaml"
      (cd "$PRJ/code/yolov5-v7.0" && "$PY" train.py --cfg "$CFG" --weights "$PRJ/weights/yolov5s_v7.0.pt" \
        --data "$DATA" --hyp data/hyps/hyp.scratch-low.yaml --optimizer Adam --epochs $EPOCHS --batch-size $BATCH \
        --img 640 --seed 0 --workers 2 --project "$OUT" --name "$NAME" --exist-ok) > "$LOG" 2>&1 || rc=$? ;;
    v8s)
      (cd "$OUT" && "$PY" -c "from ultralytics.yolo.cfg import entrypoint; entrypoint()" detect train \
        model="$PRJ/weights/yolov8s.pt" data="$DATA" optimizer=Adam lr0=0.01 epochs=$EPOCHS batch=$BATCH imgsz=640 \
        close_mosaic=10 seed=0 deterministic=True workers=2 project="$OUT" name="$NAME" exist_ok=True) > "$LOG" 2>&1 || rc=$? ;;
    *) echo "unknown run $RUN"; exit 2;;
  esac
  echo "[$(date '+%F %T')] done  $NAME exit $rc" | tee -a "$OUT/runs.txt"
done
