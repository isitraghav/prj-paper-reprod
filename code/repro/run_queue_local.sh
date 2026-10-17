#!/usr/bin/env bash
# Local queue (Laptop B): wait for P2 training -> P2 eval + summary [-> P1 smoke -> P1 full, only with --with-p1].
# P1 runs on the RTX 4070 laptop by default (2026-10-17 decision). Stops before P1 full if a P1 smoke run fails. Progress lines go to results/queue.txt.
# usage: bash code/repro/run_queue_local.sh   (from the repo root)
set -uo pipefail
cd "$(dirname "$0")/../.."
Q=results/queue.txt
say() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

say "queue start; waiting for P2 training (p2_v8s done line after the relaunch)"
until tail -n +9 results/p2/runs.txt 2>/dev/null | grep -q "done  p2_v8s"; do sleep 60; done
say "P2 training finished: $(tail -n +9 results/p2/runs.txt | grep done | tr '\n' ';')"

say "P2 eval start"
bash code/repro/p2/eval_p2.sh 2>&1 | tee -a "$Q"
envs/p2/Scripts/python.exe -I code/repro/summarize.py > results/summary_stdout.txt 2>&1 && say "summary written" || say "summary FAILED"

[ "${1:-}" = "--with-p1" ] || { say "queue done (P2 only; P1 runs on the RTX 4070 laptop)"; exit 0; }

say "P1 smoke start"
rm -rf results/smoke/p1_yolov5s results/smoke/p1_gbh
bash code/repro/p1/run_p1.sh smoke yolov5s gbh
if grep -E "done  p1_(yolov5s|gbh) exit [^0]" <(tail -n 4 results/smoke/runs.txt); then
  say "P1 smoke FAILED; not starting full P1 runs"; exit 1
fi
say "P1 smoke ok: $(for r in yolov5s gbh; do tr '\r' '\n' < results/smoke/p1_$r.log | grep -E '^ +all ' | tail -1 | sed 's/  */ /g'; done | tr '\n' ';')"

say "P1 full start"
bash code/repro/p1/run_p1.sh full yolov5s gbh
say "queue done: $(tail -n 2 results/p1/runs.txt | tr '\n' ';')"
