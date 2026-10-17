#!/usr/bin/env bash
# P1 training runs (Li et al. 2023, GBH-YOLOv5), the paper's settings except physical batch (init.md §11).
# Paper: Adam, lr 0.001, batch 16, 500 epochs, 960x960, COCO yolov5s.pt (v5.0); GBH = authors' released yaml.
# Batch: 16 doesn't fit in 6 GB at 960 px (measured max: GBH 6, YOLOv5s 12). We use the largest divisor of 64 that
# fits: GBH 4, YOLOv5s 8. With nbs=64 accumulation every optimizer step sees 64 images, as the paper's 16 x 4 did,
# with identical weight-decay scaling. Differences: BN batch statistics, and warm-up (max(3 epochs, 1000 iters):
# ~18 epochs at batch 16 vs ~4.5 (GBH) / ~9 (YOLOv5s) epochs here).
# workers=2: Windows commit limit (see run_p2.sh).
# Other GPUs: override with env vars, always a divisor of 64 that fits (check with code/repro/mem_probe.py), e.g.
#   RTX 4070 laptop 8 GB:  BATCH_YOLOV5S=16 BATCH_GBH=8 bash code/repro/p1/run_p1.sh full yolov5s gbh
#   (YOLOv5s 16 = paper exactly; GBH needs ~10.4 GB for 16). WORKERS=4 if RAM >= 16 GB.
#
# usage: bash code/repro/p1/run_p1.sh <smoke|full> <yolov5s|gbh>...   (from the repo root)
set -euo pipefail
PRJ="$(cd "$(dirname "$0")/../../.." && (pwd -W 2>/dev/null || pwd))"  # C:/... on Windows
MODE="$1"; shift
case "$MODE" in smoke) EPOCHS=1; OUT="$PRJ/results/smoke";; full) EPOCHS=500; OUT="$PRJ/results/p1";; *) echo "mode?"; exit 2;; esac
PY="$PRJ/envs/p1/Scripts/python.exe"; [ -x "$PY" ] || PY="$PRJ/envs/p1/bin/python"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"

for RUN in "$@"; do
  case "$RUN" in
    yolov5s) CFG=yolov5s.yaml; BATCH=${BATCH_YOLOV5S:-8};;
    gbh) CFG=gbh_released.yaml; BATCH=${BATCH_GBH:-4};;
    *) echo "unknown run $RUN"; exit 2;;
  esac
  NAME="p1_$RUN"
  echo "[$(date '+%F %T')] start $NAME ($MODE, $EPOCHS ep, batch $BATCH)" | tee -a "$OUT/runs.txt"
  rc=0
  (cd "$PRJ/code/gbh-yolov5" && "$PY" train.py --cfg "$PRJ/code/repro/p1/$CFG" --weights "$PRJ/weights/yolov5s_v5.0.pt" \
    --data "$PRJ/datasets/pv_multi_defect_yolo/data.yaml" --hyp "$PRJ/code/repro/p1/hyp.gbh.yaml" --adam \
    --epochs $EPOCHS --batch-size $BATCH --img-size 960 960 --workers ${WORKERS:-2} --device 0 \
    --project "$OUT" --name "$NAME" --exist-ok) > "$OUT/$NAME.log" 2>&1 || rc=$?
  echo "[$(date '+%F %T')] done  $NAME exit $rc" | tee -a "$OUT/runs.txt"
done
