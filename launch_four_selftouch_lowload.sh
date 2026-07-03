#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/selftouch_four_lowload"
mkdir -p "$LOG_DIR"

COMMON_ENV=(
  WANDB_MODE=disabled
  WANDB_CONSOLE=off
  WANDB_DISABLE_CODE=true
  WANDB_DISABLE_GIT=true
  MPLBACKEND=Agg
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  TORCH_NUM_THREADS=1
  SELFTOUCH_TORCH_THREADS=1
  SELFTOUCH_DEVICE=cuda
  SELFTOUCH_TRAIN_MICRO_BATCH_SIZE=1
  SELFTOUCH_EVAL_BATCH_SIZE=8
  SELFTOUCH_CUDA_MEMORY_FRACTION=0.10
  SELFTOUCH_TRAIN_STEP_SLEEP=0
  SELFTOUCH_MAX_TRAIN_BATCHES=20
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

launch_variant() {
  local variant="$1"
  local delay="$2"
  local param_file="$ROOT_DIR/parameter/${variant}/parameter_base/parameter_base.yaml"
  local log_file="$LOG_DIR/${variant}.log"
  local pid_file="$LOG_DIR/${variant}.pid"

  (
    cd "$ROOT_DIR" || exit 1
    sleep "$delay"
    echo "[$(date '+%F %T')] starting ${variant}" >> "$log_file"
    exec env "${COMMON_ENV[@]}" python3 main.py -mode train -param_file "$param_file" -config train
  ) >> "$log_file" 2>&1 &

  echo "$!" > "$pid_file"
  echo "started ${variant} pid=$(cat "$pid_file") log=$log_file delay=${delay}s"
}

launch_variant selftouch_fcn_pos 0
launch_variant selftouch_fcn_cmd 3
launch_variant selftouch_fcn_trq 6
launch_variant selftouch_fcn_vel 9

echo "Logs: $LOG_DIR"
