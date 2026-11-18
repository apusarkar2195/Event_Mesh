#!/usr/bin/env bash
# Run SLAM on all 3 TUM RGB-D sequences (fr1_desk, fr2_xyz, fr3_office).
# Usage: from repo root, ./scripts/run_tum_sequences.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS_BASE="${DATASETS_BASE:-datasets/tum}"

run_one() {
  local config="$1"
  local dataset_name="$2"
  local dataset_path="${DATASETS_BASE}/${dataset_name}"
  echo "=============================================="
  echo "Running: $config"
  echo "  dataset_path: $dataset_path"
  echo "=============================================="
  # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python slam.py \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python slam.py \
    --config "$config" \
    --dataset_path "$dataset_path" \
    -o "Results.use_gui=False"
}

run_one "configs/rgbd/tum/fr1_desk.yaml" "rgbd_dataset_freiburg1_desk"
run_one "configs/rgbd/tum/fr2_xyz.yaml" "rgbd_dataset_freiburg2_xyz"
run_one "configs/rgbd/tum/fr3_office.yaml" "rgbd_dataset_freiburg3_long_office_household"

echo "Done: all 3 TUM sequences finished."
