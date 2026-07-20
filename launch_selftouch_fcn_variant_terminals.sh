#!/usr/bin/env bash
set -euo pipefail

TERMINAL_CMD="${TERMINAL_CMD:-x-terminal-emulator}"
LOG_DIR="${LOG_DIR:-logs/selftouch_fcn_variants}"
SWEEP_COUNT="${SWEEP_COUNT:-1}"
MAX_ACTIVE_TERMINAL_JOBS="${MAX_ACTIVE_TERMINAL_JOBS:-1}"
SLOT_DIR="${SLOT_DIR:-/tmp/selftouch_fcn_variant_slots}"

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

if ! [[ "$MAX_ACTIVE_TERMINAL_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_ACTIVE_TERMINAL_JOBS" -lt 1 ]; then
  echo "MAX_ACTIVE_TERMINAL_JOBS must be a positive integer; got '${MAX_ACTIVE_TERMINAL_JOBS}'" >&2
  exit 2
fi

if ! command -v "$TERMINAL_CMD" >/dev/null 2>&1; then
  echo "Terminal command not found: $TERMINAL_CMD" >&2
  echo "Set TERMINAL_CMD, for example: TERMINAL_CMD=gnome-terminal $0" >&2
  exit 2
fi

if [ -z "${DISPLAY:-}" ]; then
  echo "DISPLAY is not set, so GUI terminals cannot be opened from this shell." >&2
  echo "Run this script from a terminal inside your desktop session." >&2
  exit 2
fi

if command -v xhost >/dev/null 2>&1 && ! xhost >/dev/null 2>&1; then
  cat >&2 <<EOF
Cannot connect to X display '${DISPLAY}' from this shell.

Most likely this root shell does not have permission to open windows on your
desktop X server. From an already-open desktop terminal as the desktop user,
run:

  xhost +SI:localuser:root

Then come back to this root shell and run:

  cd /home/handling04/Documents/HASA
  ./launch_selftouch_fcn_variant_terminals.sh

If you are connected through SSH or VS Code without a real desktop display,
GUI terminal windows cannot be opened from here.
EOF
  exit 2
fi

echo "Opening ${#variants[@]} terminals with $TERMINAL_CMD"
echo "Logs: $LOG_DIR"
echo "Active training slots: $MAX_ACTIVE_TERMINAL_JOBS"

failed=0

for variant in "${variants[@]}"; do
  log_path="${LOG_DIR}/${variant}.log"
  title="${variant}"
  run_cmd=$(cat <<EOF
cd /home/handling04/Documents/HASA
mkdir -p "$LOG_DIR"
mkdir -p "$SLOT_DIR"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] START ${variant}" | tee -a "$log_path"
echo "log: $log_path"
echo "waiting for training slot, max active jobs: $MAX_ACTIVE_TERMINAL_JOBS"
slot_fd=""
slot_id=""
while [ -z "\$slot_fd" ]; do
  for slot in \$(seq 1 "$MAX_ACTIVE_TERMINAL_JOBS"); do
    lock_path="$SLOT_DIR/slot_\${slot}.lock"
    exec {try_fd}>"\$lock_path"
    if flock -n "\$try_fd"; then
      slot_fd="\$try_fd"
      slot_id="\$slot"
      break
    fi
    exec {try_fd}>&-
  done
  if [ -z "\$slot_fd" ]; then
    echo "[\$(date '+%H:%M:%S')] ${variant} waiting for free slot..."
    sleep 10
  fi
done
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] ${variant} acquired slot \${slot_id}" | tee -a "$log_path"
WANDB_MODE=offline \\
WANDB_DIR=/home/handling04/Documents/HASA/wandb \\
WANDB_CACHE_DIR=/home/handling04/Documents/HASA/.wandb-cache \\
WANDB_CONSOLE=off \\
WANDB_DISABLE_CODE=true \\
WANDB_DISABLE_GIT=true \\
OMP_NUM_THREADS=1 \\
MKL_NUM_THREADS=1 \\
OPENBLAS_NUM_THREADS=1 \\
NUMEXPR_NUM_THREADS=1 \\
TORCH_NUM_THREADS=1 \\
SELFTOUCH_TORCH_THREADS=1 \\
SELFTOUCH_TRAIN_MICRO_BATCH_SIZE=1 \\
SELFTOUCH_EVAL_BATCH_SIZE=2 \\
SELFTOUCH_EVAL_MICRO_BATCH_SIZE=2 \\
SELFTOUCH_CUDA_MEMORY_FRACTION=0.80 \\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
python3 main.py \\
  -mode sweep \\
  -param_file "parameter/${variant}/parameter_base/parameter_base.yaml" \\
  -config train \\
  -sweep_count "$SWEEP_COUNT" 2>&1 | tee -a "$log_path"
code=\${PIPESTATUS[0]}
exec {slot_fd}>&-
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] END ${variant} status=\${code}" | tee -a "$log_path"
echo
echo "Finished ${variant} with status \${code}. Press Enter to close this terminal."
read -r _
exit "\${code}"
EOF
)

  set +e
  "$TERMINAL_CMD" -T "$title" -e bash -lc "$run_cmd" &
  pid=$!
  sleep 0.5
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    wait "$pid"
    code=$?
    if [ "$code" -ne 0 ]; then
      echo "Failed to open terminal for ${variant}; status=${code}" >&2
      failed=$((failed + 1))
    fi
  fi
  set -e
  sleep 0.2
done

if [ "$failed" -gt 0 ]; then
  echo "Failed to open ${failed} terminal(s)." >&2
  exit 1
fi

echo "Launched all variant terminals."
