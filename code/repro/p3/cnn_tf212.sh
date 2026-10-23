#!/usr/bin/env bash
# P3 CNN (code/repro/p3/train_cnn.py, unchanged recipe) on TensorFlow 2.12 in Docker, on the GPU only: the container
# gets --gpus all and train_cnn.py exits before training if TensorFlow sees no GPU.
# usage (repo root): bash code/repro/p3/cnn_tf212.sh <seed>   -> results/p3/cnn_tf212_seed<seed>/, models/p3/cnn_vgg19_tf212_seed<seed>.h5
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash: keep container paths (/repo) unconverted
SEED="$1"
PRJ="$(cd "$(dirname "$0")/../../.." && (pwd -W 2>/dev/null || pwd))"
OUT="results/p3/cnn_tf212_seed$SEED"; mkdir -p "$PRJ/$OUT" "$PRJ/weights/keras_tf212"
docker build -t prj-p3-tf212 "$PRJ/code/repro/p3/docker_tf212" > "$PRJ/results/p3/docker_tf212_build.log" 2>&1
docker image inspect prj-p3-tf212 --format '{{.Id}}' >> "$PRJ/results/p3/docker_tf212_build.log"
# ImageNet VGG-19 weights are cached outside the container (weights/ is not committed)
docker run --rm --gpus all -v "$PRJ:/repo" -v "$PRJ/weights/keras_tf212:/root/.keras" -w /repo prj-p3-tf212 \
  python3 code/repro/p3/train_cnn.py --out "$OUT" --seed "$SEED"
cp "$PRJ/$OUT/stage2.h5" "$PRJ/models/p3/cnn_vgg19_tf212_seed$SEED.h5"
