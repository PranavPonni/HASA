# MotionLearning Training Pipeline Guide

This guide explains how to run, debug, and extend the training pipeline in this repository.

## 1) Quick architecture map

- Entrypoint: `python main.py -mode ... -param_file ...`
- Orchestration and mode dispatch live in `main.py`.
- Dynamic controller import pattern:
  - `model/<model_name>/controller.py`
  - class name read from `Required.controller_name` in parameter YAML.
- Data path, scaling, splitting, rearranging live in `data_preproc.py` and model-specific `data_loader.py`.
- Base iterator batching logic is in `dataloader_base.py`.
- Model-specific networks live under `model/<model_name>/`.
- Saved model checkpoints are placed under `model_weight/...`.

## 2) How model selection actually works (critical)

The model module loaded at runtime is controlled by parameter file location, not by the YAML `Model.model` value.

`main.py` computes:
- `model_save_path` by replacing `parameter` with `model_weight` in the param file directory path
- `model_name` as `basename(dirname(model_save_path))`

So if your parameter file is:
- `parameter/t_rnn_pos/parameter_base/parameter_base.yaml`

then runtime imports:
- `model.t_rnn_pos.controller`

Implication:
- Keep parameter file directory and model directory aligned.
- If you copy a YAML into the wrong folder, you may load the wrong controller.

## 3) Environment setup

There is no top-level pinned requirements file. Use one of the logged W&B run requirements as a baseline and install missing packages.

A practical baseline (from one recorded run):
- torch==2.4.1
- torchvision==0.19.1
- numpy==1.24.4
- pandas==2.0.3
- matplotlib==3.1.2
- opencv-python==4.12.0.88
- einops==0.8.1
- wandb==0.21.1
- ruamel.yaml==0.18.15
- pytorch-msssim==1.0.0
- schedulefree==1.4.1

Install minimum dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib opencv-python einops wandb ruamel.yaml pytorch-msssim schedulefree tqdm
```

Optional runtime dependencies:
- ROS-related packages are needed for motion mode in controllers that use `ros_bridge.py` and robot interfaces.
- ffmpeg is useful for video exports from visualizers.

## 4) Parameter file structure

Most runs use a YAML with these top-level blocks:
- `Dataset`
- `Model`
- `Motion`
- `Required`
- `Test`
- `Train`
- `Pretrain`
- `Sweep`

Commonly edited fields:

### Dataset
- `data_dir`: root directory containing episode folders.
- `modality`: per signal scaling + rearrange descriptor.
- `sequence_length`: number of timesteps expected per episode.
- `shift_data`: label shift amount.
- `test_data`: list of episode folder names for held-out test.

### Model
Depends on model family.

Examples:
- self-touch models: `hand_dim`, `tactile_dim`, architecture fields.
- external-touch models: may require pretrained self-touch references:
  - `st_param`
  - `st_model`

### Required
- `controller_name`: usually `RNN_controller`.
- `model_name`: often present, but runtime controller import is effectively resolved by param file directory as explained above.

### Train
- `batch_size`, `lr`, `num_epochs`, `model_save_iter`, `project`, `loss_coef`, and sometimes `noise`, `cls_rate`.

### Sweep
- W&B sweep config scaffold under `Sweep.tune`, plus `method`, `metric`, `project`.

## 5) Data layout expectations

### Episode organization
Typical expected shape:
- `Dataset.data_dir` contains many episode folders, e.g. `episode0`, `episode1`, ...
- each episode folder contains timestep pickles.

In default preprocessing path, files are requested as:
- `timestep1.pkl` to `timestep<sequence_length>.pkl`

Data loader behavior if a timestep file is missing/corrupted:
- if possible, previous timestep is reused as fallback.
- otherwise an exception is raised.

### Pickle content expectations
Each timestep pickle is expected to contain modality keys declared in YAML, for example:
- tactile: `tactile_index_tip`, `tactile_thumb_tip`
- hand state: `hand_jnt_pos`, `hand_jnt_cmd_pos`, optionally `hand_jnt_vel`, `hand_jnt_trq`

### Scaling and rearrange
The pipeline usually does:
1. train/test split via `Dataset.test_data`
2. compute scaling params from train split
3. save `scaling_param.pkl` under parameter file directory
4. scale train/test
5. rearrange modality tensors with einops expression from YAML

## 6) Standard run commands

For the controlled 14-variant, fixed-seed FCN input ablation, use the dedicated
workflow in `SCIENTIFIC_FCN_ABLATION.md`; do not use the hyperparameter-sweep
launcher for that experiment.

From repository root:

### Train
```bash
python main.py -mode train -param_file parameter/t_rnn_pos/parameter_base/parameter_base.yaml
```

### Test
```bash
python main.py -mode test -param_file parameter/t_rnn_pos/parameter_base/parameter_base.yaml
```

### Motion (robot execution path)
```bash
python main.py -mode motion -param_file parameter/t_rnn_pos/parameter_base/parameter_base.yaml
```

### Sweep
```bash
python main.py -mode sweep -param_file parameter/t_rnn_pos/parameter_base/parameter_base.yaml -config train
```

### Pretrain (if implemented for that controller)
```bash
python main.py -mode pretrain -param_file parameter/t_rnn_pos/parameter_base/parameter_base.yaml
```

## 7) Output locations

### Checkpoints
Saved under path derived from parameter location:
- from `parameter/...` to `model_weight/...`

Example:
- param file: `parameter/t_rnn_pos/parameter_base/parameter_base.yaml`
- checkpoint dir: `model_weight/t_rnn_pos/parameter_base/`

### Scaling parameters
Saved to:
- `<param_file_dir>/scaling_param.pkl`

### Logs and visual artifacts
Common outputs under:
- `log/graphs/`
- `log/gifs/`
- `log/vids/`
- `log/images/`

W&B local run data is in:
- `wandb/`

## 8) Workflow to start a new experiment safely

1. Duplicate a known-good parameter YAML in the same model family folder.
2. Change only:
   - `Dataset.data_dir`
   - `Dataset.test_data`
   - `Train.project`
   - key model/training hyperparameters
3. Keep `sequence_length` aligned with actual available timesteps per episode.
4. Run one short smoke training (`num_epochs` small, e.g. 2-5).
5. Confirm:
   - checkpoint is written
   - `scaling_param.pkl` exists
   - W&B logs are appearing
6. Then launch full run.

## 9) How to add a new dataset

1. Put episodes under a dedicated directory, e.g. `data_server/<dataset_name>/<split_name>`.
2. Ensure each episode has consistent timestep pickle naming and modality keys.
3. Create/clone parameter YAML in correct model family folder.
4. Update `Dataset.data_dir`, `Dataset.test_data`, and modality list/rearrange rules.
5. Run short train smoke test.

## 10) How to add a new model family

Create folder:
- `model/<new_model_name>/`

Required files:
- `controller.py`: class matching `Required.controller_name` implementing:
  - `train_controller`
  - `test_controller`
  - `motion_controller` (can be stub if not needed)
- `data_loader.py`: custom loader returning tensors shaped for your model.
- model implementation file(s), e.g. `<new_model_name>.py`.

Then create parameter folder:
- `parameter/<new_model_name>/parameter_base/parameter_base.yaml`

Important:
- The folder name `<new_model_name>` in both `model/` and `parameter/` must match.

## 11) W&B sweep behavior

Sweep mode:
- creates sweep config from YAML
- starts W&B agent
- writes tuned params to per-run parameter directory

Generated run params are written via `data_preproc.write_yaml(...)` into new param directories under `parameter/...`.

## 12) Sync utility

Repository has rsync-based sync helper:
- script: `sync/sync.py`
- config: `sync/sync.yaml`

Usage:

```bash
python sync/sync.py <key> <in|out|syn> --config sync/sync.yaml
```

Where:
- `out`: local -> remote
- `in`: remote -> local
- `syn`: bidirectional by running both commands sequentially

## 13) Utilities for data maintenance

Examples:
- `data_server/check.sh`: checks file counts per episode dir against expected range.
- `data_server/exec.sh`: renames episode directories with offset.
- `build_scaler.py`: robust scaler stats builder over large episode trees.

## 14) Known pitfalls and gotchas

1. Runtime model selection depends on parameter file path folder names, not just YAML model text.
2. Some controllers contain hard-coded test paths and checkpoint epoch values; update these before using `-mode test`.
3. In multiple controllers, `wandb.finish()` appears inside the epoch loop block. That can prematurely close runs; verify behavior in your target controller.
4. `torch.cuda.init()` is called in controller constructors. On CPU-only environments this can fail early.
5. Parameter trees include many historical sweep outputs; always start from a known base template in `parameter/<model>/parameter_base/`.
6. One parameter template (`parameter/selftouch_fcn_pos/parameter_base/parameter_base.yaml`) appears heavily customized; validate keys against actual controller/data loader expectations before using it as canonical base.

## 15) Practical debug checklist

If training crashes:

1. Confirm parameter file path points to intended model family folder.
2. Verify `Dataset.data_dir` exists and contains enough episodes/timesteps.
3. Verify timestep pickle names are consistent with `sequence_length` expectations.
4. Check all modality keys in data match YAML `Dataset.modality` keys.
5. Verify `st_param` and `st_model` paths if using external-touch models.
6. Confirm `scaling_param.pkl` is produced.
7. Run with tiny `batch_size` and short `num_epochs` to isolate shape/runtime issues.

## 16) Suggested command recipes

### Small smoke run
```bash
python main.py -mode train -param_file parameter/s_rnn_pos/parameter_base/parameter_base.yaml
```
Then interrupt after first checkpoint and inspect logs/artifacts.

### Train then test same family
```bash
python main.py -mode train -param_file parameter/t_rnn_pos/parameter_base/parameter_base.yaml
python main.py -mode test  -param_file parameter/t_rnn_pos/parameter_base/parameter_base.yaml
```

### Sweep launch
```bash
python main.py -mode sweep -param_file parameter/sat_rnn_pos/parameter_base/parameter_base.yaml -config train
```

## 17) File map reference

Core:
- `main.py`
- `controller_base.py`
- `dataloader_base.py`
- `data_preproc.py`
- `util.py`

Representative model families:
- `model/selftouch_fcn_pos/`
- `model/selftouch_transformer/`
- `model/t_rnn_pos/`
- `model/s_rnn_pos/`
- `model/sat_rnn_pos/`

Params and weights:
- `parameter/`
- `model_weight/`

Data and logs:
- `data_server/`
- `log/`
- `wandb/`
