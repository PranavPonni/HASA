#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SWEEP_COUNT="${SWEEP_COUNT:-1}"
CONFIG="${CONFIG:-train}"
export SELFTOUCH_MAX_TRAIN_BATCHES=0
export SELFTOUCH_ENABLE_PLOTS="${SELFTOUCH_ENABLE_PLOTS:-0}"
export SELFTOUCH_WANDB_IMAGES="${SELFTOUCH_WANDB_IMAGES:-0}"
export SELFTOUCH_WANDB_TACTILE_METRICS="${SELFTOUCH_WANDB_TACTILE_METRICS:-1}"
export SELFTOUCH_WANDB_PCA="${SELFTOUCH_WANDB_PCA:-1}"
export SELFTOUCH_WANDB_TACTILE_PROFILE="${SELFTOUCH_WANDB_TACTILE_PROFILE:-1}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

if ! [[ "$SWEEP_COUNT" =~ ^[0-9]+$ ]]; then
  echo "SWEEP_COUNT must be a non-negative integer; got '${SWEEP_COUNT}'" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

param_files=(
  parameter/selftouch_contrastive_gru/parameter_base/parameter_base.yaml
  parameter/selftouch_contrastive_temporal/parameter_base/parameter_base.yaml
  parameter/selftouch_contrastive_fcn/parameter_base/parameter_base.yaml
  parameter/selftouch_contrastive_transformer/parameter_base/parameter_base.yaml
  parameter/selftouch_contrastive_mamba/parameter_base/parameter_base.yaml
)

echo "Running ${#param_files[@]} contrastive pos+trq self-touch sweeps sequentially with SWEEP_COUNT=${SWEEP_COUNT}"

for index in "${!param_files[@]}"; do
  param_file="${param_files[$index]}"
  run_number=$((index + 1))

  if [ ! -f "$param_file" ]; then
    echo "Missing parameter file: ${param_file}" >&2
    exit 2
  fi

  echo
  echo "[$run_number/${#param_files[@]}] Starting ${param_file}"
  "$PYTHON_BIN" main.py \
    -mode sweep \
    -param_file "$param_file" \
    -config "$CONFIG" \
    -sweep_count "$SWEEP_COUNT"
  echo "[$run_number/${#param_files[@]}] Finished ${param_file}"
done

echo
echo "All contrastive pos+trq self-touch sweeps finished."
