#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-logs/selftouch_fcn_pos_offsets}"
SWEEP_COUNT="${SWEEP_COUNT:-11}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
  echo "MAX_PARALLEL_JOBS must be a positive integer; got '${MAX_PARALLEL_JOBS}'" >&2
  exit 2
fi

export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export SELFTOUCH_TORCH_THREADS="${SELFTOUCH_TORCH_THREADS:-1}"
export SELFTOUCH_TRAIN_MICRO_BATCH_SIZE="${SELFTOUCH_TRAIN_MICRO_BATCH_SIZE:-1}"
export SELFTOUCH_EVAL_BATCH_SIZE="${SELFTOUCH_EVAL_BATCH_SIZE:-2}"
export SELFTOUCH_EVAL_MICRO_BATCH_SIZE="${SELFTOUCH_EVAL_MICRO_BATCH_SIZE:-2}"
export SELFTOUCH_CUDA_MEMORY_FRACTION="${SELFTOUCH_CUDA_MEMORY_FRACTION:-0.80}"
export SELFTOUCH_TRAIN_STEP_SLEEP="${SELFTOUCH_TRAIN_STEP_SLEEP:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

variants=(
  selftouch_fcn_pos_tplus2
  selftouch_fcn_pos_tplus5
  selftouch_fcn_pos_tplus10
  selftouch_fcn_pos_tminus2
  selftouch_fcn_pos_tminus5
  selftouch_fcn_pos_tminus10
)

mkdir -p "$LOG_DIR"

status=0
running=0

echo "Running ${#variants[@]} selftouch FCN position-offset variants with MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
echo "SWEEP_COUNT=${SWEEP_COUNT}; default matches the 10-11 replicate discipline used by the strongest modality variants."
echo "Low-VRAM defaults: micro-batch=${SELFTOUCH_TRAIN_MICRO_BATCH_SIZE}, eval_batch=${SELFTOUCH_EVAL_BATCH_SIZE}, cuda_fraction=${SELFTOUCH_CUDA_MEMORY_FRACTION}"

for variant in "${variants[@]}"; do
  while [ "$running" -ge "$MAX_PARALLEL_JOBS" ]; do
    if wait -n; then
      :
    else
      status=1
    fi
    running=$((running - 1))
  done

  log_path="${LOG_DIR}/${variant}.log"
  sweep_count_arg=()
  if [ -n "$SWEEP_COUNT" ]; then
    sweep_count_arg=(-sweep_count "$SWEEP_COUNT")
  fi

  echo "Starting ${variant}; log: ${log_path}"
  (
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${variant}"
    echo "SWEEP_COUNT=${SWEEP_COUNT:-continuous} MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
    python3 main.py \
      -mode sweep \
      -param_file "parameter/${variant}/parameter_base/parameter_base.yaml" \
      -config train \
      "${sweep_count_arg[@]}"
    code="$?"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END ${variant} status=${code}"
    exit "$code"
  ) >"${log_path}" 2>&1 &
  running=$((running + 1))
done

while [ "$running" -gt 0 ]; do
  if wait -n; then
    :
  else
    status=1
  fi
  running=$((running - 1))
done

exit "$status"
