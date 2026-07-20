#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-logs/selftouch_fcn_variants}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
SEED="${SEED:-11}"

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
  echo "MAX_PARALLEL_JOBS must be a positive integer; got '${MAX_PARALLEL_JOBS}'" >&2
  exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer; got '${SEED}'" >&2
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
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
if [ -z "${SELFTOUCH_CUDA_MEMORY_FRACTION:-}" ]; then
  case "$MAX_PARALLEL_JOBS" in
    1) SELFTOUCH_CUDA_MEMORY_FRACTION=0.80 ;;
    2) SELFTOUCH_CUDA_MEMORY_FRACTION=0.40 ;;
    3) SELFTOUCH_CUDA_MEMORY_FRACTION=0.27 ;;
    *) SELFTOUCH_CUDA_MEMORY_FRACTION=0.20 ;;
  esac
fi
export SELFTOUCH_CUDA_MEMORY_FRACTION
export SELFTOUCH_TRAIN_STEP_SLEEP="${SELFTOUCH_TRAIN_STEP_SLEEP:-0}"
export SELFTOUCH_MAX_TRAIN_BATCHES=0
export SELFTOUCH_SEED="$SEED"
export SELFTOUCH_DETERMINISTIC=true
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

variants=(
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
)

mkdir -p "$LOG_DIR"

status=0
running=0

echo "Running ${#variants[@]} selftouch FCN variants with MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
echo "Fixed seed=${SEED}; every model consumes the full training loader."
echo "Low-VRAM defaults: micro-batch=${SELFTOUCH_TRAIN_MICRO_BATCH_SIZE}, eval_batch=${SELFTOUCH_EVAL_BATCH_SIZE}, cuda_fraction=${SELFTOUCH_CUDA_MEMORY_FRACTION}"
echo "The launcher keeps up to ${MAX_PARALLEL_JOBS} model jobs active at once."

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

  echo "Starting ${variant}; log: ${log_path}"
  (
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${variant}"
    echo "SEED=${SEED} MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
    "$PYTHON_BIN" main.py \
      -mode train \
      -param_file "parameter/${variant}/parameter_base/parameter_base.yaml" \
      -config train \
      -seed "$SEED"
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
