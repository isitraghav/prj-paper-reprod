#!/usr/bin/env bash
# Sensitivity inference for COMPARISON.md §6 (main runs: run_dumps.sh).
# S1: detectors on the foreign dataset at that dataset's paper input size (P1 models on ELDDS @640, P2 models on PV @960).
# S2: P3 CNN on ELDDS 12x12 tiles (detector tile scores come from the main dumps).
# Then: envs/p3cnn/Scripts/python.exe -I code/repro/cross/sensitivity.py
#
# usage: bash code/repro/cross/run_sensitivity.sh   (from the repo root)
set -euo pipefail
PRJ="$(cd "$(dirname "$0")/../../.." && (pwd -W 2>/dev/null || pwd))"
P1="$PRJ/envs/p1/Scripts/python.exe"; [ -x "$P1" ] || P1="$PRJ/envs/p1/bin/python"
P2="$PRJ/envs/p2/Scripts/python.exe"; [ -x "$P2" ] || P2="$PRJ/envs/p2/bin/python"
P3="$PRJ/envs/p3cnn/Scripts/python.exe"; [ -x "$P3" ] || P3="$PRJ/envs/p3cnn/bin/python"
L="$PRJ/results/cross/lists"; O="$PRJ/results/cross/preds_sens"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$O"

"$P3" -I "$PRJ/code/repro/cross/make_tiles.py" --eldds-grid 12 --out tiles_eldds12.csv
"$P3" -I "$PRJ/code/repro/cross/cnn_tiles.py" --tiles tiles_eldds12.csv --out p3_cnn__tiles_eldds12
(cd "$PRJ/code/gbh-yolov5"
 "$P1" ../repro/cross/dump_v5.py --weights "$PRJ/models/p1/yolov5s.pt" --list "$L/eldds.txt" --img 640 --out "$O/p1_yolov5s__eldds@640.csv"
 "$P1" ../repro/cross/dump_v5.py --weights "$PRJ/models/p1/gbh_yolov5.pt" --list "$L/eldds.txt" --img 640 --out "$O/p1_gbh__eldds@640.csv")
(cd "$PRJ/code/yolov5-v7.0"
 "$P2" ../repro/cross/dump_v5.py --weights "$PRJ/models/p2/yolov5s.pt" --list "$L/pv.txt" --img 960 --out "$O/p2_v5s__pv@960.csv"
 "$P2" ../repro/cross/dump_v5.py --weights "$PRJ/models/p2/yolov5s_gam_asff.pt" --list "$L/pv.txt" --img 960 --diou-nms --out "$O/p2_improved__pv@960.csv")
"$P2" "$PRJ/code/repro/cross/dump_v8.py" --weights "$PRJ/models/p2/yolov8s.pt" --list "$L/pv.txt" --img 960 --out "$O/p2_v8s__pv@960.csv"
