#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SWEEP_COUNT="${SWEEP_COUNT:-5}"
CONFIG="${CONFIG:-train}"

if ! [[ "$SWEEP_COUNT" =~ ^[0-9]+$ ]]; then
  echo "SWEEP_COUNT must be a non-negative integer; got '${SWEEP_COUNT}'" >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

param_files=(
  parameter/sat_act_pos_rugby/parameter_base/parameter_base.yaml
  parameter/t_act_pos_rugby/parameter_base/parameter_base.yaml
  parameter/sat_act_pos_bolt/parameter_base/parameter_base.yaml
  parameter/t_act_pos_bolt/parameter_base/parameter_base.yaml
  parameter/sat_act_pos_bigbolt/parameter_base/parameter_base.yaml
  parameter/t_act_pos_bigbolt/parameter_base/parameter_base.yaml
  parameter/sat_act_pos_kmwipe/parameter_base/parameter_base.yaml
  parameter/t_act_pos_kmwipe/parameter_base/parameter_base.yaml
)

echo "Running ${#param_files[@]} ACT position sweeps sequentially with SWEEP_COUNT=${SWEEP_COUNT}"

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
echo "All ACT position sweeps finished."
