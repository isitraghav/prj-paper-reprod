#!/usr/bin/env bash
# Replication R1 (the earlier RTX 3050 run, kept outside the repo): its P1 weights, unchanged, on the three test sets.
# Same dumper and settings as code/repro/cross/run_dumps.sh; R1 models at their own training size (600 -> 608 px).
# Also runs our YOLOv5-1 (models/p1/yolov5_1.pt, not part of the main comparison) for the paired R1-vs-R2 tests.
# Inference only. The R1 checkpoints were checked first: their pickles import only torch / numpy / collections /
# yolov5 `models` classes.
#
# usage (repo root): bash code/repro/replication/run_r1_dumps.sh <R1 folder, e.g. ~/Downloads/trained_weights_and_results>
set -uo pipefail
PRJ="$(cd "$(dirname "$0")/../../.." && (pwd -W 2>/dev/null || pwd))"
R1="$(cd "$1" && (pwd -W 2>/dev/null || pwd))/elec_paper/GBH-YOLOv5/runs/train"
P1="$PRJ/envs/p1/Scripts/python.exe"; [ -x "$P1" ] || P1="$PRJ/envs/p1/bin/python"
OUT="$PRJ/results/replication_r1/preds"; L="$PRJ/results/cross/lists"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
for DS in pv eldds elpv; do
  for M in abl_yolov5s abl_yolov5s_1 abl_yolov5s_2 pv_gbh_b16; do
    (cd "$PRJ/code/gbh-yolov5" && "$P1" ../repro/cross/dump_v5.py --weights "$R1/$M/weights/last.pt" --list "$L/$DS.txt" \
      --img 608 --out "$OUT/r1_${M}__$DS.csv") > "$OUT/r1_${M}__$DS.log" 2>&1
    echo "r1 $M $DS exit $?"
  done
  (cd "$PRJ/code/gbh-yolov5" && "$P1" ../repro/cross/dump_v5.py --weights "$PRJ/models/p1/yolov5_1.pt" --list "$L/$DS.txt" \
    --img 960 --out "$OUT/r2_yolov5_1__$DS.csv") > "$OUT/r2_yolov5_1__$DS.log" 2>&1
  echo "r2 yolov5_1 $DS exit $?"
done
