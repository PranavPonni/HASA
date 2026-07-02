#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-logs/selftouch_models}"

models=(
  selftouch_fcn
  selftouch_fcn_pos
  selftouch_fcn_vel
  selftouch_fcn_trq
  selftouch_fcn_cmd
  selftouch_fcn_posvel
  selftouch_fcn_postrq
  selftouch_fcn_poscmd
  selftouch_fcn_velcmd
  selftouch_fcn_veltrq
  selftouch_fcn_trqcmd
  selftouch_fcn_posveltrq
  selftouch_fcn_postrqcmd
  selftouch_fcn_poscmdvel
  selftouch_fcn_posveltrqcmd
  selftouch_tcn
  selftouch_transformer
  selftouch_temporal_mixer
  selftouch_gru_attention
  selftouch_contrastive_fcn
  selftouch_contrastive_gru
  selftouch_contrastive_temporal
  selftouch_contrastive_transformer
)

if [ "$#" -gt 0 ]; then
  models=("$@")
fi

mkdir -p "$LOG_DIR"

run_model() {
  local model="$1"
  local param_file="parameter/${model}/parameter_base/parameter_base.yaml"

  if [ ! -f "$param_file" ]; then
    echo "Missing parameter file: ${param_file}" >&2
    return 1
  fi

  local cmd=(
    python3 main.py
    -mode sweep
    -param_file "$param_file"
    -config train
  )
  "${cmd[@]}"
}

status=0
for model in "${models[@]}"; do
  echo "Starting ${model}; log: ${LOG_DIR}/${model}.log"
  run_model "$model" >"${LOG_DIR}/${model}.log" 2>&1 &
done

while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
  if ! wait -n; then
    status=1
  fi
done

exit "$status"
