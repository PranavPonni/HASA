import os


def configure_reproducibility(seed, deterministic=True):
    """Seed every RNG used by the training pipeline.

    This is intentionally called before dataset construction and model
    initialization.  Deterministic algorithms are requested in warning mode so
    an unsupported CUDA kernel is reported without killing a long experiment.
    """
    import random

    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    deterministic = bool(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass
    print(f"[repro] seed={seed} | deterministic={deterministic}")
    return seed


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_or_none(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def wandb_service_settings():
    import wandb

    base_kwargs = {
        "init_timeout": _env_int("WANDB_INIT_TIMEOUT", 300),
        "_service_wait": _env_int("WANDB__SERVICE_WAIT", 300),
    }
    keys = list(base_kwargs)
    while keys:
        kwargs = {key: base_kwargs[key] for key in keys}
        try:
            return wandb.Settings(**kwargs)
        except (TypeError, ValueError):
            keys.pop()
    return wandb.Settings()


def amp_enabled(mode_param, device):
    return bool(mode_param.get("use_amp", False)) and getattr(device, "type", "") == "cuda"


def make_grad_scaler(enabled):
    import torch

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(enabled):
    import torch

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def configure_torch_threads(mode_param):
    import torch

    configured = (mode_param or {}).get("torch_num_threads")
    if configured in (None, ""):
        configured = os.environ.get("SELFTOUCH_TORCH_THREADS", os.environ.get("TORCH_NUM_THREADS", 1))
    try:
        threads = int(configured)
    except (TypeError, ValueError):
        threads = 1
    if threads <= 0:
        return None

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, threads))
    except RuntimeError:
        pass
    print(f"[speed] torch CPU threads per process: {threads}")
    return threads


def maybe_watch_model(wandb_module, model, mode_param):
    if not bool(mode_param.get("wandb_watch", False)):
        return
    wandb_module.watch(
        model,
        log=mode_param.get("wandb_watch_log", "gradients"),
        log_freq=int(mode_param.get("wandb_watch_freq", 100)),
    )


def wandb_log_heartbeat(wandb_module):
    """Confirm a newly initialized W&B run without uploading model data."""
    run = getattr(wandb_module, "run", None)
    if run is None:
        return False
    wandb_module.log({"runtime/initialized": 1}, commit=False)
    return True


def get_wandb_entity(*params):
    for param in params:
        if not isinstance(param, dict):
            continue
        entity = param.get("wandb_entity") or param.get("entity")
        if entity:
            return entity
    return os.environ.get("WANDB_ENTITY")


def wandb_init_kwargs(mode_param, project=None, config=None):
    kwargs = {"project": project or mode_param["project"]}
    entity = get_wandb_entity(mode_param)
    if entity:
        kwargs["entity"] = entity
    if config is not None:
        config = dict(config)
        for key in ("seed", "deterministic", "run_name"):
            if key in mode_param:
                config[key] = mode_param[key]
        kwargs["config"] = config
    if mode_param.get("run_name"):
        kwargs["name"] = str(mode_param["run_name"])
    kwargs["settings"] = wandb_service_settings()
    return kwargs


def should_run_period(epoch, total_epoch, every):
    try:
        every = int(every)
    except (TypeError, ValueError):
        every = 0
    return epoch == 0 or epoch + 1 == total_epoch or (every > 0 and (epoch + 1) % every == 0)


class EarlyStopper:
    def __init__(self, mode_param):
        mode_param = mode_param or {}
        self.enabled = bool(mode_param.get("early_stop_enabled", False))
        self.patience = int(mode_param.get("early_stop_patience", 60))
        self.min_delta = float(mode_param.get("early_stop_min_delta", 1e-4))
        self.min_epochs = int(mode_param.get("early_stop_min_epochs", 120))
        self.monitor = str(mode_param.get("early_stop_monitor", "total_loss"))
        self.mode = str(mode_param.get("early_stop_mode", "min")).lower()
        self.best = None
        self.bad_epochs = 0

    def step(self, value, epoch):
        if not self.enabled:
            return False
        value = float(value)
        if self.mode == "max":
            improved = self.best is None or value > self.best + self.min_delta
        else:
            improved = self.best is None or value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return (epoch + 1) >= self.min_epochs and self.bad_epochs >= self.patience


def lr_for_epoch(base_lr, epoch, mode_param, optimizer_step=None):
    schedule = (mode_param or {}).get("lr_schedule") or {}
    milestones = schedule.get("milestones", [])
    factors = schedule.get("factors", [])
    unit = str(schedule.get("unit", "epoch")).strip().lower()
    if unit in {"optimizer_step", "optimizer_steps", "step", "steps", "update", "updates"}:
        progress = int(optimizer_step or 0)
    else:
        progress = int(epoch)
    factor = 1.0
    for milestone, next_factor in zip(milestones, factors):
        if progress >= int(milestone):
            factor = float(next_factor)
    return float(base_lr) * factor


def apply_lr_schedule(optimizer, base_lr, epoch, mode_param, optimizer_step=None):
    if optimizer_step is None:
        optimizer_step = int(getattr(optimizer, "_selftouch_optimizer_step", 0))
    lr = lr_for_epoch(base_lr, epoch, mode_param, optimizer_step=optimizer_step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def empty_cuda_cache(device):
    if getattr(device, "type", "") != "cuda":
        return
    import torch

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def as_float(value):
    import torch

    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def data_batch_size(data):
    import torch

    for value in data.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    return 0


def slice_data_batch(data, start, end):
    import torch

    output = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= end:
            output[key] = value[start:end]
        else:
            output[key] = value
    return output


def move_batch_to_device(data, device):
    import torch

    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def micro_batch_size(mode_param, mode, batch_size):
    mode_param = mode_param or {}
    if mode == "train":
        configured = os.environ.get(
            "SELFTOUCH_TRAIN_MICRO_BATCH_SIZE",
            mode_param.get("train_micro_batch_size", 0),
        )
    else:
        configured = os.environ.get(
            "SELFTOUCH_EVAL_MICRO_BATCH_SIZE",
            mode_param.get("eval_micro_batch_size", 0),
        )
    try:
        size = int(configured or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0 or batch_size <= 0:
        return 0
    return max(1, min(size, int(batch_size)))


def _optimizer_step(model, optimizer, scaler, enabled_amp, mode_param):
    import torch

    grad_clip = float((mode_param or {}).get("grad_clip_norm", 0.0) or 0.0)
    if enabled_amp:
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    optimizer._selftouch_optimizer_step = int(
        getattr(optimizer, "_selftouch_optimizer_step", 0)
    ) + 1


def _forward_loss(model, data, loss_coef, enabled_amp):
    with autocast(enabled_amp):
        return model.forward_loss(**data, loss_coef=loss_coef)


def selftouch_loss_step(
    *,
    model,
    optimizer,
    scaler,
    loss_coef,
    data,
    device,
    mode,
    use_amp,
    mode_param,
):
    """Run one self-touch loss step, optionally with true train micro-batching."""
    import torch

    is_train = mode == "train"
    enabled_amp = bool(use_amp) and getattr(device, "type", "") == "cuda"
    batch_count = data_batch_size(data)
    micro_size = micro_batch_size(mode_param, mode, batch_count)

    if is_train and optimizer is not None:
        base_lr = float((mode_param or {}).get("lr", optimizer.param_groups[0]["lr"]))
        apply_lr_schedule(
            optimizer,
            base_lr,
            epoch=0,
            mode_param=mode_param,
            optimizer_step=int(getattr(optimizer, "_selftouch_optimizer_step", 0)),
        )
        optimizer.zero_grad(set_to_none=True)

    if micro_size <= 0 or micro_size >= batch_count:
        data_on_device = move_batch_to_device(data, device)
        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            losses, preds = _forward_loss(model, data_on_device, loss_coef, enabled_amp)
        total_loss = losses[0]
        if is_train:
            if enabled_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            _optimizer_step(model, optimizer, scaler, enabled_amp, mode_param)
        return losses, preds

    notice_name = f"_selftouch_{mode}_microbatch_notice"
    if not getattr(model, notice_name, False):
        accumulation = (batch_count + micro_size - 1) // micro_size
        print(
            f"[cuda] {mode} micro-batch={micro_size}"
            f" | effective_batch={batch_count}"
            f" | accumulation_steps={accumulation}"
        )
        setattr(model, notice_name, True)

    weighted_losses = None
    pred_chunks = None
    pred_count = 0
    total_count = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for start in range(0, batch_count, micro_size):
            end = min(start + micro_size, batch_count)
            chunk_count = end - start
            weight = float(chunk_count) / float(max(batch_count, 1))
            chunk = move_batch_to_device(slice_data_batch(data, start, end), device)
            losses, preds = _forward_loss(model, chunk, loss_coef, enabled_amp)
            losses = tuple(losses)
            preds = tuple(preds)
            total_loss = losses[0]
            if weighted_losses is None:
                weighted_losses = [0.0 for _ in losses]
                pred_chunks = [[] for _ in preds]
            pred_count = len(preds)
            for idx, loss_value in enumerate(losses):
                weighted_losses[idx] += float(loss_value.detach()) * weight
            if is_train:
                scaled_loss = total_loss * weight
                if enabled_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            else:
                for idx, pred in enumerate(preds):
                    pred_chunks[idx].append(pred.detach().cpu())
            total_count += chunk_count
            del chunk, losses, preds, total_loss

    if is_train:
        _optimizer_step(model, optimizer, scaler, enabled_amp, mode_param)
        empty = tuple(torch.empty(0) for _ in range(pred_count))
        return tuple(weighted_losses or []), empty

    averaged = tuple(float(value) for value in (weighted_losses or []))
    preds = tuple(torch.cat(chunks, dim=0) for chunks in pred_chunks)
    return averaged, preds


def configure_cuda_memory_fraction(mode_param, device):
    if getattr(device, "type", "") != "cuda":
        return None

    fraction = _float_or_none(os.environ.get("SELFTOUCH_CUDA_MEMORY_FRACTION"))
    if fraction is None:
        fraction = _float_or_none((mode_param or {}).get("cuda_memory_fraction"))
    if fraction is None:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError("cuda_memory_fraction must be in the interval (0, 1].")

    import torch

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(float(fraction), device_index)
    print(f"[cuda] per-process memory fraction capped at {fraction:.3f}")
    return fraction


def contrastive_eval_step(model, data, device, loss_coef, temperature, use_amp, supcon_loss):
    """Evaluate contrastive self-touch models in CPU-sourced GPU chunks."""
    import torch

    batch_count = data_batch_size(data)
    chunk_size = micro_batch_size({}, "test", batch_count) or batch_count
    loss_sums = {}
    pred_chunks = {}
    supcon_sum = 0.0
    total_count = 0

    for start in range(0, batch_count, chunk_size):
        end = min(start + chunk_size, batch_count)
        chunk_count = end - start
        chunk = move_batch_to_device(slice_data_batch(data, start, end), device)
        pos = chunk["hand_jnt_pos"]
        vel = chunk["hand_jnt_vel"]
        trq = chunk["hand_jnt_trq"]
        cmd = chunk["hand_jnt_cmd_pos"]
        labels = chunk["label"]
        with autocast(bool(use_amp)):
            supcon = supcon_loss(
                torch.nn.functional.normalize(model.encode(pos, vel, trq, cmd), dim=-1),
                labels,
                temperature,
            )
            losses, preds = model.forward_loss(
                tactile_index_tip=chunk["tactile_index_tip"],
                tactile_thumb_tip=chunk["tactile_thumb_tip"],
                tactile_middle_tip=chunk["tactile_middle_tip"],
                hand_jnt_pos=pos,
                hand_jnt_vel=vel,
                hand_jnt_trq=trq,
                hand_jnt_cmd_pos=cmd,
                labels=labels,
                loss_coef=loss_coef,
            )
        supcon_sum += float(supcon.detach().cpu()) * chunk_count
        for key, value in losses.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu()) * chunk_count
        for key, value in preds.items():
            pred_chunks.setdefault(key, []).append(value.detach().cpu())
        total_count += chunk_count
        del chunk, losses, preds, supcon

    denom = max(total_count, 1)
    averaged_losses = {key: torch.tensor(value / denom) for key, value in loss_sums.items()}
    combined_preds = {key: torch.cat(values, dim=0) for key, values in pred_chunks.items()}
    return torch.tensor(supcon_sum / denom), averaged_losses, combined_preds
