#!/usr/bin/env bash
# P2 evaluation on the test split (190 images, 640 px), each model without and with TTA (P2 Tables 1-2).
# Stock val settings of each codebase (YOLOv5 v7.0: conf 0.001, IoU 0.6; ultralytics 8.0.99: conf 0.001, IoU 0.7).
# The improved model uses DIoU-NMS (P2 contribution 2). YOLOv8 TTA goes through v8_val.py (8.0.99 ignores augment=True).
# Primary TTA = 640 px (paper's stated size). Extra 'tta832' = TTA at 832 px (ultralytics' TTA guide uses ~1.3x the
# train size); reported separately as a sensitivity check, never as the reproduction number.
#
# usage: bash code/repro/p2/eval_p2.sh [v5s] [v5_improved] [v8s]     (from the repo root; default: all)
set -euo pipefail
PRJ="$(cd "$(dirname "$0")/../../.." && (pwd -W 2>/dev/null || pwd))"
PY="$PRJ/envs/p2/Scripts/python.exe"; [ -x "$PY" ] || PY="$PRJ/envs/p2/bin/python"
DATA="$PRJ/datasets/eldds_yolo/data.yaml"
OUT="$PRJ/results/p2/eval"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(v5s v5_improved v8s)

for RUN in "${RUNS[@]}"; do
  W="$PRJ/results/p2/p2_$RUN/weights/best.pt"
  [ -f "$W" ] || { echo "missing $W"; continue; }
  for TTA in notta tta tta832; do
    NAME="${RUN}_$TTA"
    case "$RUN" in
      v5s|v5_improved)
        EXTRA=(); [ $TTA != notta ] && EXTRA+=(--augment); [ $RUN = v5_improved ] && EXTRA+=(--diou-nms)
        IMG=640; [ $TTA = tta832 ] && IMG=832
        (cd "$PRJ/code/yolov5-v7.0" && "$PY" val.py --weights "$W" --data "$DATA" --img $IMG --task test \
          --batch-size 16 --workers 2 --verbose "${EXTRA[@]}" --project "$OUT" --name "$NAME" --exist-ok) \
          > "$OUT/$NAME.log" 2>&1 ;;
      v8s)
        EXTRA=(); [ $TTA != notta ] && EXTRA+=(--tta)
        IMG=640; [ $TTA = tta832 ] && IMG=832
        (cd "$OUT" && "$PY" "$PRJ/code/repro/p2/v8_val.py" "${EXTRA[@]}" model="$W" data="$DATA" split=test \
          imgsz=$IMG batch=16 workers=1 project="$OUT" name="$NAME" exist_ok=True) > "$OUT/$NAME.log" 2>&1 ;;
    esac
    echo "[$(date '+%F %T')] $NAME: $(tr '\r' '\n' < "$OUT/$NAME.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^ +all ' | tail -1)"
  done
done
