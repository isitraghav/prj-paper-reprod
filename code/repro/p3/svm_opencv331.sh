#!/usr/bin/env bash
# P3 SVM with the paper's OpenCV 3.3.1 (strict defaults, same script as results/p3/svm): builds the image once, then
# counts KAZE keypoints (3.3.1 vs 3.4.18) and runs code/repro/p3/train_svm.py inside the container.
# usage (repo root): bash code/repro/p3/svm_opencv331.sh
set -euo pipefail
PRJ="$(cd "$(dirname "$0")/../../.." && (pwd -W 2>/dev/null || pwd))"
OUT="$PRJ/results/p3/svm_cv331"; mkdir -p "$OUT"
docker build -t prj-p3-opencv331 "$PRJ/code/repro/p3/docker" > "$OUT/docker_build.log" 2>&1
"$PRJ/envs/p3svm/Scripts/python.exe" -I "$PRJ/code/repro/p3/kaze_count.py" > "$OUT/kaze_count.jsonl"
docker run --rm -v "$PRJ:/repo" -w /repo prj-p3-opencv331 python3 code/repro/p3/kaze_count.py >> "$OUT/kaze_count.jsonl"
docker run --rm -v "$PRJ:/repo" -w /repo prj-p3-opencv331 python3 code/repro/p3/train_svm.py --out results/p3/svm_cv331 --jobs 8 \
  > "$OUT/train.log" 2>&1
