#!/usr/bin/env bash
# Run SLAM on all Replica RGB-D sequences (office0-4, room0-2).
# On failure (e.g. OOM), continues with the next config.
# Usage: from repo root, ./scripts/run_replica_sequences.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS_BASE="${DATASETS_BASE:-datasets/replica}"

CONFIGS=(
  "configs/rgbd/replica/office0.yaml"
  "configs/rgbd/replica/office1.yaml"
  "configs/rgbd/replica/office2.yaml"
  "configs/rgbd/replica/office3.yaml"
  "configs/rgbd/replica/office4.yaml"
  "configs/rgbd/replica/room0.yaml"
  "configs/rgbd/replica/room1.yaml"
  "configs/rgbd/replica/room2.yaml"
)

run_one() {
  local config="$1"
  local base_name
  base_name="$(basename "$config" .yaml)"
  local scene="${base_name%_sp}"
  local dataset_path="${DATASETS_BASE}/${scene}"

  echo "=============================================="
  echo "Running: $config"
  echo "  dataset_path: $dataset_path"
  echo "=============================================="

  # if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python slam.py \
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python slam.py \
    --config "$config" \
    --dataset_path "$dataset_path" \
    -o "Results.use_gui=False"; then
    echo "[OK] Finished: $config"
    return 0
  else
    local ret=$?
    echo "[FAIL] Exit code $ret (e.g. OOM) for $config — continuing to next."
    return $ret
  fi
}

failed=0
for config in "${CONFIGS[@]}"; do
  if ! run_one "$config"; then
    ((failed++)) || true
  fi
done

echo ""
echo "Done: ${#CONFIGS[@]} sequences run, $failed failed."
if [ "$failed" -gt 0 ]; then
  exit 1
fi
