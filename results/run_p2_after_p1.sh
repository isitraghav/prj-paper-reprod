#!/usr/bin/env bash
set -euo pipefail
PRJ="/c/Users/isitr/Desktop/prj-paper-reprod"
cd "$PRJ"

echo "[$(date '+%F %T')] Waiting for P1 yolov5_2 to finish..."
until grep -q "done  p1_yolov5_2 exit 0" results/p1/runs.txt 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%F %T')] P1 yolov5_2 done (exit 0), starting P2 ablations"

# Run sequentially (no simultaneous jobs)
bash code/repro/p2/run_p2.sh full v5_gam
echo "[$(date '+%F %T')] v5_gam done, starting v5_asff"
bash code/repro/p2/run_p2.sh full v5_asff
echo "[$(date '+%F %T')] v5_asff done, running full P2 eval"

bash code/repro/p2/eval_p2.sh
echo "[$(date '+%F %T')] All done"
