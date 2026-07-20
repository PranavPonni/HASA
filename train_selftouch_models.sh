#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-logs/selftouch_models}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
SWEEP_COUNT="${SWEEP_COUNT:-1}"

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
  echo "MAX_PARALLEL_JOBS must be a positive integer; got '${MAX_PARALLEL_JOBS}'" >&2
  exit 2
fi
if ! [[ "$SWEEP_COUNT" =~ ^[0-9]+$ ]] || [ "$SWEEP_COUNT" -lt 1 ]; then
  echo "SWEEP_COUNT must be a positive integer; got '${SWEEP_COUNT}'" >&2
  exit 2
fi

export SELFTOUCH_TRAIN_MICRO_BATCH_SIZE="${SELFTOUCH_TRAIN_MICRO_BATCH_SIZE:-1}"
export SELFTOUCH_EVAL_BATCH_SIZE="${SELFTOUCH_EVAL_BATCH_SIZE:-2}"
export SELFTOUCH_EVAL_MICRO_BATCH_SIZE="${SELFTOUCH_EVAL_MICRO_BATCH_SIZE:-2}"
export SELFTOUCH_CUDA_MEMORY_FRACTION="${SELFTOUCH_CUDA_MEMORY_FRACTION:-0.80}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

models=(
  selftouch_fcn
  selftouch_fcn_pos
  selftouch_fcn_pos_tminus10
  selftouch_fcn_pos_tminus5
  selftouch_fcn_pos_tminus2
  selftouch_fcn_pos_tplus2
  selftouch_fcn_pos_tplus5
  selftouch_fcn_pos_tplus10
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
  selftouch_fcn_postrqcmd_tminus10
  selftouch_fcn_postrqcmd_tminus5
  selftouch_fcn_postrqcmd_tminus2
  selftouch_fcn_postrqcmd_tplus2
  selftouch_fcn_postrqcmd_tplus5
  selftouch_fcn_postrqcmd_tplus10
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

echo "Low-GPU mode: parallel=${MAX_PARALLEL_JOBS} train_micro_batch=${SELFTOUCH_TRAIN_MICRO_BATCH_SIZE} eval_micro_batch=${SELFTOUCH_EVAL_MICRO_BATCH_SIZE} cuda_fraction=${SELFTOUCH_CUDA_MEMORY_FRACTION}"
echo "Runs per model: ${SWEEP_COUNT}"

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
    -sweep_count "$SWEEP_COUNT"
  )
  "${cmd[@]}"
}

status=0
running=0
for model in "${models[@]}"; do
  while [ "$running" -ge "$MAX_PARALLEL_JOBS" ]; do
    if ! wait -n; then
      status=1
    fi
    running=$((running - 1))
  done

  echo "Starting ${model}; log: ${LOG_DIR}/${model}.log"
  run_model "$model" >"${LOG_DIR}/${model}.log" 2>&1 &
  running=$((running + 1))
done

while [ "$running" -gt 0 ]; do
  if ! wait -n; then
    status=1
  fi
  running=$((running - 1))
done

exit "$status"
