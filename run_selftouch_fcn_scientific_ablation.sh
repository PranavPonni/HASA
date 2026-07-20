#!/usr/bin/env bash
set -uo pipefail

# Reproducible FCN input ablation.  Default: 14 variants x 5 fixed seeds with
# tactile history enabled.  Add the matched no-tactile-history control with:
#   HISTORY_MODES="with_history no_history" ./run_selftouch_fcn_scientific_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-fcn_ablation_v2}"
SEEDS="${SEEDS:-11 22 33 44 55}"
HISTORY_MODES="${HISTORY_MODES:-with_history}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
LOG_DIR="${LOG_DIR:-logs/${EXPERIMENT_TAG}}"
FORCE_RERUN="${FORCE_RERUN:-0}"

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

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
  echo "MAX_PARALLEL_JOBS must be a positive integer; got '${MAX_PARALLEL_JOBS}'" >&2
  exit 2
fi

for history_mode in $HISTORY_MODES; do
  if [ "$history_mode" != "with_history" ] && [ "$history_mode" != "no_history" ]; then
    echo "HISTORY_MODES entries must be with_history or no_history; got '${history_mode}'" >&2
    exit 2
  fi
done

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

case "$MAX_PARALLEL_JOBS" in
  1) CUDA_FRACTION="${SELFTOUCH_CUDA_MEMORY_FRACTION:-0.80}" ;;
  2) CUDA_FRACTION="${SELFTOUCH_CUDA_MEMORY_FRACTION:-0.40}" ;;
  3) CUDA_FRACTION="${SELFTOUCH_CUDA_MEMORY_FRACTION:-0.27}" ;;
  *) CUDA_FRACTION="${SELFTOUCH_CUDA_MEMORY_FRACTION:-0.20}" ;;
esac

mkdir -p "$LOG_DIR"
status=0
running=0
scheduled=0
skipped=0

echo "Scientific FCN ablation"
echo "variants=${#variants[@]} seeds=[$SEEDS] history_modes=[$HISTORY_MODES]"
echo "full batches/epoch, eval/save/plot every 25 epochs, max_parallel=${MAX_PARALLEL_JOBS}"

for history_mode in $HISTORY_MODES; do
  if [ "$history_mode" = "with_history" ]; then
    history_arg="on"
  else
    history_arg="off"
  fi

  for seed in $SEEDS; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
      echo "Every seed must be a non-negative integer; got '${seed}'" >&2
      exit 2
    fi

    for variant in "${variants[@]}"; do
      while [ "$running" -ge "$MAX_PARALLEL_JOBS" ]; do
        if wait -n; then :; else status=1; fi
        running=$((running - 1))
      done

      run_name="${EXPERIMENT_TAG}_${history_mode}_seed${seed}"
      metrics="model_weight/${variant}/${run_name}/plots/raw_prediction_metrics.csv"
      if [ "$FORCE_RERUN" != "1" ] && [ -f "$metrics" ] && rg -q '^500,' "$metrics"; then
        echo "SKIP completed ${variant}/${run_name}"
        skipped=$((skipped + 1))
        continue
      fi

      log_path="${LOG_DIR}/${variant}_${history_mode}_seed${seed}.log"
      echo "START ${variant} history=${history_mode} seed=${seed} log=${log_path}"
      (
        export PYTHONHASHSEED="$seed"
        export CUBLAS_WORKSPACE_CONFIG=":4096:8"
        export SELFTOUCH_SEED="$seed"
        export SELFTOUCH_DETERMINISTIC=true
        export SELFTOUCH_MAX_TRAIN_BATCHES=0
        export SELFTOUCH_CUDA_MEMORY_FRACTION="$CUDA_FRACTION"
        export SELFTOUCH_TRAIN_MICRO_BATCH_SIZE="${SELFTOUCH_TRAIN_MICRO_BATCH_SIZE:-1}"
        export SELFTOUCH_EVAL_BATCH_SIZE="${SELFTOUCH_EVAL_BATCH_SIZE:-2}"
        export SELFTOUCH_EVAL_MICRO_BATCH_SIZE="${SELFTOUCH_EVAL_MICRO_BATCH_SIZE:-2}"
        export SELFTOUCH_TORCH_THREADS="${SELFTOUCH_TORCH_THREADS:-1}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
        export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
        export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
        export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hasa-matplotlib-cache}"
        export WANDB_MODE="${WANDB_MODE:-offline}"
        export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
        export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
        export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"
        "$PYTHON_BIN" main.py \
          -mode train \
          -param_file "parameter/${variant}/parameter_base/parameter_base.yaml" \
          -config train \
          -seed "$seed" \
          -run_name "$run_name" \
          -tactile_history "$history_arg"
      ) >"$log_path" 2>&1 &
      running=$((running + 1))
      scheduled=$((scheduled + 1))
    done
  done
done

while [ "$running" -gt 0 ]; do
  if wait -n; then :; else status=1; fi
  running=$((running - 1))
done

echo "Training matrix finished: scheduled=${scheduled} skipped=${skipped} status=${status}"
if [ "$status" -ne 0 ]; then
  echo "At least one run failed. Inspect ${LOG_DIR}; rerunning this script resumes completed runs." >&2
  exit "$status"
fi

seed_args=()
for seed in $SEEDS; do
  seed_args+=("$seed")
done

history_args=()
for history_mode in $HISTORY_MODES; do
  history_args+=("$history_mode")
done

"$PYTHON_BIN" aggregate_selftouch_fcn_ablation.py \
  --run-prefix "$EXPERIMENT_TAG" \
  --expected-seeds "${seed_args[@]}" \
  --expected-modes "${history_args[@]}" \
  --strict
