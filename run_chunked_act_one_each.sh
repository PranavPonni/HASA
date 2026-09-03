#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

run_one() {
  local label="$1"
  local param_file="$2"

  echo
  echo "===== ${label}: starting chunked ACT one-run sweep ====="
  date
  python3 main.py \
    -mode sweep \
    -param_file "${param_file}" \
    -config train \
    -sweep_count 1
  echo "===== ${label}: finished ====="
  date
}

run_one "bigbolt" "parameter/sat_act_pos_bigbolt/parameter_base/chunked_act.yaml"
run_one "bolt" "parameter/sat_act_pos_bolt/parameter_base/chunked_act.yaml"
run_one "kmwipe" "parameter/sat_act_pos_kmwipe/parameter_base/chunked_act.yaml"
run_one "rugby" "parameter/sat_act_pos_rugby/parameter_base/chunked_act.yaml"
