import os


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
        kwargs["config"] = config
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


def lr_for_epoch(base_lr, epoch, mode_param):
    schedule = (mode_param or {}).get("lr_schedule") or {}
    milestones = schedule.get("milestones", [])
    factors = schedule.get("factors", [])
    factor = 1.0
    for milestone, next_factor in zip(milestones, factors):
        if epoch >= int(milestone):
            factor = float(next_factor)
    return float(base_lr) * factor


def apply_lr_schedule(optimizer, base_lr, epoch, mode_param):
    lr = lr_for_epoch(base_lr, epoch, mode_param)
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


def configure_cuda_memory_fraction(mode_param, device):
    if getattr(device, "type", "") != "cuda":
        return None

    fraction = _float_or_none((mode_param or {}).get("cuda_memory_fraction"))
    if fraction is None:
        fraction = _float_or_none(os.environ.get("SELFTOUCH_CUDA_MEMORY_FRACTION"))
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
