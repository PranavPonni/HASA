#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-logs/selftouch_fcn_variants}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
SWEEP_COUNT="${SWEEP_COUNT:-1}"

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
  echo "MAX_PARALLEL_JOBS must be a positive integer; got '${MAX_PARALLEL_JOBS}'" >&2
  exit 2
fi
if ! [[ "$SWEEP_COUNT" =~ ^[0-9]+$ ]] || [ "$SWEEP_COUNT" -lt 1 ]; then
  echo "SWEEP_COUNT must be a positive integer; got '${SWEEP_COUNT}'" >&2
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

# Send every run to wandb.ai. All experiment settings (including project,
# entity, seed, batching, schedule, and cadence) come from parameter_base.yaml.
export WANDB_MODE=online

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
echo "W&B mode: ${WANDB_MODE}"
echo "Sweep runs per variant: ${SWEEP_COUNT}"
echo "Each run uses its own parameter/<variant>/parameter_base/parameter_base.yaml unchanged."
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

  param_file="parameter/${variant}/parameter_base/parameter_base.yaml"
  log_path="${LOG_DIR}/${variant}.log"

  if [ ! -f "$param_file" ]; then
    echo "Parameter file not found: ${param_file}" >&2
    status=1
    continue
  fi

  echo "Starting ${variant}; log: ${log_path}"
  (
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${variant}"
    echo "PARAM_FILE=${param_file} MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS} SWEEP_COUNT=${SWEEP_COUNT} WANDB_MODE=${WANDB_MODE}"
    "$PYTHON_BIN" main.py \
      -mode sweep \
      -param_file "$param_file" \
      -config train \
      -sweep_count "$SWEEP_COUNT"
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
