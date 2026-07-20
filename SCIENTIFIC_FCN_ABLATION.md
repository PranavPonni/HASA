# Reproducible 14-variant FCN input ablation

The scientific rerun uses the same backbone, data split, batch size, 500 epochs,
full 300 batches per epoch, update-based learning-rate schedule, and evaluation
cadence for every input variant.  Runs are deterministic and are aggregated over
the fixed seeds `11 22 33 44 55`.

## Primary experiment: 14 variants x 5 seeds

```bash
cd /home/handling04/Documents/HASA
./run_selftouch_fcn_scientific_ablation.sh
```

This schedules 70 runs.  It defaults to one GPU job at a time and resumes by
skipping runs whose metric history contains epoch 500.

## Add the no-tactile-history control

This is the full state-input experiment plus a matched no-tactile-history
control, for 140 total runs:

```bash
HISTORY_MODES="with_history no_history" \
./run_selftouch_fcn_scientific_ablation.sh
```

For a three-seed pilot before committing to the full matrix:

```bash
SEEDS="11 22 33" HISTORY_MODES="with_history no_history" \
./run_selftouch_fcn_scientific_ablation.sh
```

Use `MAX_PARALLEL_JOBS=2` only if two full models fit safely in GPU memory.  The
launcher derives a per-process CUDA memory fraction from this value.

## Outputs

Each run is written to:

```text
model_weight/<variant>/fcn_ablation_v2_<history-mode>_seed<seed>/
```

Each run includes:

- `tactile_profile_epoch_*.png`: predicted and raw mean tactile traces overlaid.
- `tactile_identity_epoch_*.png`: every predicted taxel against raw tactile;
  correct predictions cluster on the dashed `pred = raw` line.
- `tactile_taxel_error_epoch_*.png`: taxel-by-time absolute-error heatmap.
- `tactile_residual_epoch_*.png`: signed raw-minus-predicted mean trace.
- `raw_prediction_metrics.csv`: epoch, optimizer step, seed, history mode, and
  raw MAE/RMSE/correlation/R2/p95/bias metrics.

After the matrix finishes, the launcher runs the aggregator and writes:

```text
analysis/selftouch_fcn_ablation_rerun/
```

The report and CSV tables contain per-run results and mean, sample SD, and 95%
Student-t confidence intervals.  Ranking and convergence plots use optimizer
updates rather than nominal epochs.

The analysis can also be regenerated manually:

```bash
.venv/bin/python3 aggregate_selftouch_fcn_ablation.py \
  --run-prefix fcn_ablation_v2 \
  --expected-seeds 11 22 33 44 55 \
  --expected-modes with_history no_history \
  --strict
```

## How to judge prediction quality

The colored predicted mean trace should closely cover the black raw mean trace,
with a narrow shaded error margin.  That is necessary but not sufficient: the
trace averages 90 taxels and can hide one taxel being too high while another is
too low.  A strong model should satisfy all of the following on held-out data:

1. Predicted and raw mean traces nearly overlap without post-hoc alignment.
2. Identity-plot points remain close to `pred = raw` across the tactile range.
3. Raw-taxel MAE, RMSE, p95 error, and absolute bias are low.
4. Correlation is high and R2 is positive and close to one.
5. The taxel-error heatmap does not contain persistent bright taxels or periods.
6. These properties repeat across seeds and are summarized by a narrow CI.

The no-history condition still retains the shared phase and contact-combination
metadata.  It isolates the effect of tactile autoregressive history; it should
be described as a state-plus-context baseline rather than a completely pure
proprioception-only model.
