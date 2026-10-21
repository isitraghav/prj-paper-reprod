#!/usr/bin/env bash
# Cross-dataset inference: every trained detector (models/p1, models/p2), unchanged, on the test split of each dataset
# (PV-Multi-Defect val 222, ELDDS1400C5 test 190, ELPV test 656 cells). Each model at its own paper's inference size
# and its codebase's val settings (see dump_v5.py / dump_v8.py). Inference only, no training.
#
# usage: bash code/repro/cross/run_dumps.sh [model...]   (from the repo root; default: all 5)
set -uo pipefail
PRJ="$(cd "$(dirname "$0")/../../.." && (pwd -W 2>/dev/null || pwd))"
P1="$PRJ/envs/p1/Scripts/python.exe"; [ -x "$P1" ] || P1="$PRJ/envs/p1/bin/python"
P2="$PRJ/envs/p2/Scripts/python.exe"; [ -x "$P2" ] || P2="$PRJ/envs/p2/bin/python"
OUT="$PRJ/results/cross/preds"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PRJ/results/cross/lists"
# image lists = each paper's evaluation split (absolute paths, as the codebases' own split files)
L="$PRJ/results/cross/lists"
[ -f "$L/pv.txt" ] || cp "$PRJ/datasets/pv_multi_defect_yolo/val.txt" "$L/pv.txt"
[ -f "$L/eldds.txt" ] || cp "$PRJ/datasets/eldds_yolo/test.txt" "$L/eldds.txt"
[ -f "$L/elpv.txt" ] || sed "s#^#$PRJ/datasets/raw/elpv-dataset/src/elpv_dataset/data/#" "$PRJ/datasets/splits/elpv/test.txt" \
  | tr -d '\r' > "$L/elpv.txt"
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(p1_yolov5s p1_gbh p2_v5s p2_improved p2_v8s)

for M in "${MODELS[@]}"; do
  for DS in pv eldds elpv; do
    L="$PRJ/results/cross/lists/$DS.txt"; CSV="$OUT/${M}__$DS.csv"
    [ -f "${CSV%.csv}.json" ] && { echo "skip $M $DS (done)"; continue; }
    rc=0
    case "$M" in
      p1_yolov5s|p1_gbh)
        W="$PRJ/models/p1/$([ $M = p1_gbh ] && echo gbh_yolov5 || echo yolov5s).pt"
        (cd "$PRJ/code/gbh-yolov5" && "$P1" ../repro/cross/dump_v5.py --weights "$W" --list "$L" --img 960 --out "$CSV") ;;
      p2_v5s|p2_improved)
        if [ $M = p2_v5s ]; then W="$PRJ/models/p2/yolov5s.pt"; X=(); else W="$PRJ/models/p2/yolov5s_gam_asff.pt"; X=(--diou-nms); fi
        (cd "$PRJ/code/yolov5-v7.0" && "$P2" ../repro/cross/dump_v5.py --weights "$W" --list "$L" --img 640 --out "$CSV" "${X[@]}") ;;
      p2_v8s)
        "$P2" "$PRJ/code/repro/cross/dump_v8.py" --weights "$PRJ/models/p2/yolov8s.pt" --list "$L" --img 640 --out "$CSV" ;;
    esac > "$OUT/${M}__$DS.log" 2>&1 || rc=$?
    echo "[$(date '+%F %T')] $M on $DS exit $rc"
  done
done
