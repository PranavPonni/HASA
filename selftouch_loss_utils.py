import torch
import torch.nn.functional as F


def _loss_name(loss_cfg):
    return str((loss_cfg or {}).get("tactile_loss", "smooth_l1")).lower()


def _cfg_float(cfg, key, default=0.0):
    return float((cfg or {}).get(key, default))


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Supervised contrastive loss over batch labels, with no tactile target use."""
    if labels is None or features is None or features.shape[0] < 2:
        return features.sum() * 0.0
    labels = labels.reshape(-1)
    features = F.normalize(features, dim=-1)
    logits = features @ features.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    if not bool(positive.any()):
        return features.sum() * 0.0

    exp_logits = torch.exp(logits) * (~eye).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_count = positive.sum(dim=1).clamp_min(1)
    per_sample = -(log_prob * positive.float()).sum(dim=1) / positive_count
    valid = positive.any(dim=1)
    return per_sample[valid].mean()


def reconstruction_loss(pred, target, loss_cfg=None):
    """State reconstruction loss with Huber/SmoothL1 by default."""
    name = _loss_name(loss_cfg)
    if name in {"mse", "l2"}:
        return F.mse_loss(pred, target)
    if name in {"mae", "l1", "mean_absolute_error"}:
        return F.l1_loss(pred, target)
    return F.smooth_l1_loss(pred, target)


def _diff_time(x):
    if x.ndim < 3 or x.shape[1] < 2:
        return None
    return x[:, 1:, :] - x[:, :-1, :]


def _curvature_time(x):
    if x.ndim < 3 or x.shape[1] < 3:
        return None
    return x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]


def _time_smooth(x, kernel_size=9):
    if x.ndim < 3 or x.shape[1] < 2:
        return x
    kernel_size = max(int(kernel_size), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, int(x.shape[1]))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        return x

    pad = kernel_size // 2
    y = x.transpose(1, 2)
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel_size, stride=1)
    if y.shape[-1] != x.shape[1]:
        y = y[..., : x.shape[1]]
    return y.transpose(1, 2)


def _highpass_time(x, kernel_size=9):
    return x - _time_smooth(x, kernel_size)


def _centered_flatten(x):
    flat = x.reshape(x.shape[0], -1)
    return flat - flat.mean(dim=1, keepdim=True)


def _center_time_profile(x):
    profile = x.mean(dim=-1)
    return profile - profile.mean(dim=1, keepdim=True)


def _correlation_loss(pred, target):
    if pred.shape[0] == 0:
        return pred.sum() * 0.0
    pred_c = _centered_flatten(pred)
    target_c = _centered_flatten(target)
    numerator = (pred_c * target_c).mean(dim=1)
    denom = pred_c.pow(2).mean(dim=1).sqrt() * target_c.pow(2).mean(dim=1).sqrt()
    corr = numerator / denom.clamp_min(1e-6)
    return (1.0 - corr).mean()


def _spatial_correlation_loss(pred, target):
    if pred.ndim < 3 or target.ndim < 3 or pred.shape[-1] < 2:
        return pred.sum() * 0.0
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    numerator = (pred_c * target_c).mean(dim=-1)
    denom = pred_c.pow(2).mean(dim=-1).sqrt() * target_c.pow(2).mean(dim=-1).sqrt()
    corr = numerator / denom.clamp_min(1e-6)
    return (1.0 - corr).mean()


def _topk_count(size, count, ratio=0.0):
    ratio = float(ratio or 0.0)
    if ratio > 0.0:
        return min(max(int(round(size * ratio)), 1), size)
    return min(max(int(count), 1), size)


def _topk_abs_error_loss(pred, target, count, ratio=0.0):
    flat_error = torch.abs(pred - target).reshape(pred.shape[0], -1)
    if flat_error.shape[-1] == 0:
        return pred.sum() * 0.0
    k = _topk_count(flat_error.shape[-1], count, ratio)
    return flat_error.topk(k, dim=-1).values.mean()


def _scale_view(raw_scale, pred):
    if raw_scale is None:
        return None
    if not torch.is_tensor(raw_scale):
        raw_scale = torch.as_tensor(raw_scale, dtype=pred.dtype, device=pred.device)
    else:
        raw_scale = raw_scale.to(device=pred.device, dtype=pred.dtype)
    while raw_scale.ndim < pred.ndim:
        raw_scale = raw_scale.unsqueeze(0)
    return raw_scale


def _raw_scaled_error(pred, target, raw_scale):
    scale = _scale_view(raw_scale, pred)
    if scale is None:
        return None
    return torch.abs(pred - target) * scale


def _raw_delta(pred, target, raw_slope):
    slope = _scale_view(raw_slope, pred)
    if slope is None:
        return None
    return (pred - target) * slope


def _raw_l1_loss(pred, target, raw_slope, normalizer, clip=None):
    delta = _raw_delta(pred, target, raw_slope)
    if delta is None:
        return pred.sum() * 0.0
    error = delta.abs()
    if clip is not None and float(clip) > 0.0:
        error = error.clamp(max=float(clip))
    return error.mean() / float(normalizer)


def _raw_huber_loss(pred, target, raw_slope, normalizer, beta=80.0, clip=None):
    delta = _raw_delta(pred, target, raw_slope)
    if delta is None:
        return pred.sum() * 0.0
    if clip is not None and float(clip) > 0.0:
        delta = delta.clamp(min=-float(clip), max=float(clip))
    beta = max(float(beta), 1e-6)
    error = torch.where(
        delta.abs() < beta,
        0.5 * delta.pow(2) / beta,
        delta.abs() - 0.5 * beta,
    )
    return error.mean() / float(normalizer)


def _raw_mean_bias_loss(pred, target, raw_slope, normalizer):
    delta = _raw_delta(pred, target, raw_slope)
    if delta is None:
        return pred.sum() * 0.0
    return delta.mean(dim=(1, 2)).abs().mean() / float(normalizer)


def _raw_timestep_mean_loss(pred, target, raw_slope, normalizer):
    delta = _raw_delta(pred, target, raw_slope)
    if delta is None:
        return pred.sum() * 0.0
    return delta.mean(dim=-1).abs().mean() / float(normalizer)


def _raw_taxel_mean_loss(pred, target, raw_slope, normalizer):
    delta = _raw_delta(pred, target, raw_slope)
    if delta is None:
        return pred.sum() * 0.0
    return delta.mean(dim=1).abs().mean() / float(normalizer)


def _raw_scaled_topk_abs_error_loss(pred, target, raw_scale, count, ratio=0.0):
    raw_error = _raw_scaled_error(pred, target, raw_scale)
    if raw_error is None:
        return pred.sum() * 0.0
    flat_error = raw_error.reshape(pred.shape[0], -1)
    if flat_error.shape[-1] == 0:
        return pred.sum() * 0.0
    k = _topk_count(flat_error.shape[-1], count, ratio)
    return flat_error.topk(k, dim=-1).values.mean()


def _target_peak_error_loss(pred, target, count, ratio=0.0):
    flat_target = target.detach().reshape(target.shape[0], -1)
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_error = torch.abs(flat_pred - target.reshape(target.shape[0], -1))
    if flat_target.shape[-1] == 0:
        return pred.sum() * 0.0

    centered = flat_target - flat_target.mean(dim=1, keepdim=True)
    activity = centered.abs()
    k = _topk_count(activity.shape[-1], count, ratio)
    indices = activity.topk(k, dim=-1).indices
    return flat_error.gather(1, indices).mean()


def _activity_mask(target, ratio):
    flat = target.detach().reshape(target.shape[0], -1)
    if flat.shape[-1] == 0:
        return torch.zeros_like(target, dtype=torch.bool)
    centered = flat - flat.mean(dim=1, keepdim=True)
    activity = centered.abs()
    ratio = min(max(float(ratio), 1e-4), 1.0)
    if hasattr(torch, "quantile"):
        threshold = torch.quantile(activity, 1.0 - ratio, dim=1, keepdim=True)
    else:
        k = min(max(int(activity.shape[-1] * ratio), 1), activity.shape[-1])
        threshold = activity.topk(k, dim=-1).values[:, -1:]
    return (activity >= threshold).reshape_as(target)


def _active_region_loss(pred, target, ratio):
    mask = _activity_mask(target, ratio)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.l1_loss(pred[mask], target[mask])


def _intensity_weighted_loss(pred, target, gamma):
    flat_target = target.detach().reshape(target.shape[0], -1)
    centered = flat_target - flat_target.mean(dim=1, keepdim=True)
    activity = centered.abs()
    denom = activity.mean(dim=1, keepdim=True).clamp_min(1e-6)
    weights = (activity / denom).clamp(max=8.0).pow(float(gamma))
    weights = weights.reshape_as(target)
    return (torch.abs(pred - target) * weights).mean()


def _contact_logit_loss(pred, target, ratio, temperature):
    mask = _activity_mask(target, ratio).float()
    flat_target = target.detach().reshape(target.shape[0], -1)
    spread = flat_target.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    logits = (
        (pred - target.detach().mean(dim=(1, 2), keepdim=True))
        .abs()
        / (spread.reshape(target.shape[0], 1, 1) * max(float(temperature), 1e-4))
    )
    return F.binary_cross_entropy_with_logits(logits, mask)


def _spread_normalized_mae(pred, target):
    error = torch.abs(pred - target).reshape(pred.shape[0], -1).mean(dim=1)
    flat_target = target.reshape(target.shape[0], -1)
    if flat_target.shape[-1] == 0:
        return pred.sum() * 0.0
    if hasattr(torch, "quantile"):
        lo = torch.quantile(flat_target.detach(), 0.05, dim=1)
        hi = torch.quantile(flat_target.detach(), 0.95, dim=1)
        spread = hi - lo
    else:
        spread = flat_target.detach().max(dim=1).values - flat_target.detach().min(dim=1).values
    spread = spread.clamp_min(1e-4)
    return (error / spread).mean()


def _start_aligned_sequence_loss(pred, target):
    if pred.ndim < 3 or target.ndim < 3 or pred.shape[1] == 0 or target.shape[1] == 0:
        return pred.sum() * 0.0
    shift = (target[:, :1, :] - pred[:, :1, :]).detach()
    aligned_pred = pred + shift
    return F.l1_loss(aligned_pred, target)


def active_tactile_loss(pred, target, loss_cfg=None):
    """Supervised tactile sequence loss on the raw training target.

    The base loss stays elementwise reconstruction. Optional terms are enabled
    only by explicit nonzero config weights and are used during training to
    improve start alignment and curve shape, not to post-process predictions.
    """
    cfg = loss_cfg or {}
    loss = reconstruction_loss(pred, target, cfg)

    mae_weight = _cfg_float(cfg, "tactile_mae_weight")
    if mae_weight:
        loss = loss + mae_weight * F.l1_loss(pred, target)

    raw_scale = cfg.get("tactile_raw_scale")
    raw_scale_weight = _cfg_float(cfg, "tactile_raw_scale_weight")
    if raw_scale_weight:
        raw_error = _raw_scaled_error(pred, target, raw_scale)
        if raw_error is not None:
            loss = loss + raw_scale_weight * raw_error.mean()

    raw_slope = cfg.get("tactile_raw_slope")
    raw_mean_normalizer = max(_cfg_float(cfg, "tactile_raw_mean_loss_scale", 20.0), 1e-6)
    raw_error_normalizer = max(_cfg_float(cfg, "tactile_raw_error_loss_scale", raw_mean_normalizer), 1e-6)
    raw_error_clip = _cfg_float(cfg, "tactile_raw_error_clip", 0.0)
    raw_l1_weight = _cfg_float(cfg, "tactile_raw_l1_weight")
    if raw_l1_weight:
        loss = loss + raw_l1_weight * _raw_l1_loss(
            pred,
            target,
            raw_slope,
            raw_error_normalizer,
            raw_error_clip,
        )

    raw_huber_weight = _cfg_float(cfg, "tactile_raw_huber_weight")
    if raw_huber_weight:
        loss = loss + raw_huber_weight * _raw_huber_loss(
            pred,
            target,
            raw_slope,
            raw_error_normalizer,
            _cfg_float(cfg, "tactile_raw_huber_beta", 80.0),
            raw_error_clip,
        )

    raw_bias_weight = _cfg_float(cfg, "tactile_raw_bias_weight")
    if raw_bias_weight:
        loss = loss + raw_bias_weight * _raw_mean_bias_loss(
            pred,
            target,
            raw_slope,
            raw_mean_normalizer,
        )

    raw_timestep_mean_weight = _cfg_float(cfg, "tactile_raw_timestep_mean_weight")
    if raw_timestep_mean_weight:
        loss = loss + raw_timestep_mean_weight * _raw_timestep_mean_loss(
            pred,
            target,
            raw_slope,
            raw_mean_normalizer,
        )

    raw_taxel_mean_weight = _cfg_float(cfg, "tactile_raw_taxel_mean_weight")
    if raw_taxel_mean_weight:
        loss = loss + raw_taxel_mean_weight * _raw_taxel_mean_loss(
            pred,
            target,
            raw_slope,
            raw_mean_normalizer,
        )

    spread_mae_weight = _cfg_float(cfg, "tactile_spread_mae_weight")
    if spread_mae_weight:
        loss = loss + spread_mae_weight * _spread_normalized_mae(pred, target)

    initial_weight = _cfg_float(cfg, "tactile_initial_weight")
    if initial_weight and pred.ndim >= 3 and target.ndim >= 3 and pred.shape[1] > 0 and target.shape[1] > 0:
        loss = loss + initial_weight * F.l1_loss(pred[:, 0, :], target[:, 0, :])

    start_aligned_weight = _cfg_float(cfg, "tactile_start_aligned_weight")
    if start_aligned_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + start_aligned_weight * _start_aligned_sequence_loss(pred, target)

    level_weight = _cfg_float(cfg, "tactile_level_weight")
    if level_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + level_weight * F.smooth_l1_loss(pred.mean(dim=-1), target.mean(dim=-1))

    bias_weight = _cfg_float(cfg, "tactile_bias_weight")
    if bias_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + bias_weight * F.l1_loss(
            pred.mean(dim=(1, 2)),
            target.mean(dim=(1, 2)),
        )

    timestep_mean_weight = _cfg_float(cfg, "tactile_timestep_mean_weight")
    if timestep_mean_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + timestep_mean_weight * F.l1_loss(pred.mean(dim=-1), target.mean(dim=-1))

    temporal_profile_weight = _cfg_float(cfg, "tactile_temporal_profile_weight")
    if temporal_profile_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + temporal_profile_weight * F.l1_loss(pred.mean(dim=-1), target.mean(dim=-1))

    profile_weight = _cfg_float(cfg, "tactile_profile_weight")
    if profile_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + profile_weight * F.smooth_l1_loss(
            _center_time_profile(pred),
            _center_time_profile(target),
        )

    taxel_profile_weight = _cfg_float(cfg, "tactile_taxel_profile_weight")
    if taxel_profile_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + taxel_profile_weight * F.l1_loss(pred.mean(dim=1), target.mean(dim=1))

    batch_profile_weight = _cfg_float(cfg, "tactile_batch_profile_weight")
    if batch_profile_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + batch_profile_weight * F.smooth_l1_loss(
            pred.mean(dim=(0, 2)),
            target.mean(dim=(0, 2)),
        )

    centered_profile_weight = _cfg_float(cfg, "tactile_centered_profile_weight")
    if centered_profile_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + centered_profile_weight * F.smooth_l1_loss(
            _center_time_profile(pred),
            _center_time_profile(target),
        )

    amplitude_weight = _cfg_float(cfg, "tactile_amplitude_weight")
    if amplitude_weight and pred.ndim >= 3 and target.ndim >= 3:
        pred_std = pred.std(dim=-1, unbiased=False)
        target_std = target.std(dim=-1, unbiased=False)
        loss = loss + amplitude_weight * F.smooth_l1_loss(pred_std, target_std)

    derivative_weight = _cfg_float(cfg, "tactile_derivative_weight")
    if derivative_weight:
        pred_d = _diff_time(pred)
        target_d = _diff_time(target)
        if pred_d is not None and target_d is not None:
            loss = loss + derivative_weight * F.smooth_l1_loss(pred_d, target_d)

    curvature_weight = _cfg_float(cfg, "tactile_curvature_weight")
    if curvature_weight:
        pred_c = _curvature_time(pred)
        target_c = _curvature_time(target)
        if pred_c is not None and target_c is not None:
            loss = loss + curvature_weight * F.smooth_l1_loss(pred_c, target_c)

    highpass_weight = _cfg_float(cfg, "tactile_highpass_weight")
    if highpass_weight and pred.ndim >= 3 and target.ndim >= 3:
        kernel = _cfg_float(cfg, "tactile_highpass_kernel", 9)
        pred_hp = _highpass_time(pred, kernel)
        target_hp = _highpass_time(target, kernel)
        loss = loss + highpass_weight * F.l1_loss(pred_hp, target_hp)

    shape_weight = _cfg_float(cfg, "tactile_shape_weight")
    if shape_weight and pred.ndim >= 3 and target.ndim >= 3:
        kernel = _cfg_float(cfg, "tactile_highpass_kernel", 9)
        loss = loss + shape_weight * _correlation_loss(
            _highpass_time(pred, kernel),
            _highpass_time(target, kernel),
        )

    temporal_std_weight = _cfg_float(cfg, "tactile_temporal_std_weight")
    if temporal_std_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + temporal_std_weight * F.smooth_l1_loss(
            pred.std(dim=1, unbiased=False),
            target.std(dim=1, unbiased=False),
        )

    delta_std_weight = _cfg_float(cfg, "tactile_delta_std_weight")
    if delta_std_weight:
        pred_d = _diff_time(pred)
        target_d = _diff_time(target)
        if pred_d is not None and target_d is not None:
            loss = loss + delta_std_weight * F.smooth_l1_loss(
                pred_d.std(dim=1, unbiased=False),
                target_d.std(dim=1, unbiased=False),
            )

    active_weight = _cfg_float(cfg, "tactile_active_weight")
    if active_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + active_weight * _active_region_loss(
            pred,
            target,
            _cfg_float(cfg, "tactile_active_threshold_ratio", 0.08),
        )

    intensity_weight = _cfg_float(cfg, "tactile_intensity_weight")
    if intensity_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + intensity_weight * _intensity_weighted_loss(
            pred,
            target,
            _cfg_float(cfg, "tactile_intensity_gamma", 2.0),
        )

    contact_weight = _cfg_float(cfg, "tactile_contact_weight")
    if contact_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + contact_weight * _contact_logit_loss(
            pred,
            target,
            _cfg_float(cfg, "tactile_active_threshold_ratio", 0.08),
            _cfg_float(cfg, "tactile_contact_temperature", 0.08),
        )

    peak_weight = _cfg_float(cfg, "tactile_peak_weight")
    if peak_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + peak_weight * _target_peak_error_loss(
            pred,
            target,
            _cfg_float(cfg, "tactile_peak_count", _cfg_float(cfg, "tactile_topk_count", 12)),
            _cfg_float(cfg, "tactile_peak_ratio", _cfg_float(cfg, "tactile_topk_ratio", 0.0)),
        )

    corr_weight = _cfg_float(cfg, "tactile_corr_weight")
    if corr_weight:
        loss = loss + corr_weight * _correlation_loss(pred, target)

    spatial_corr_weight = _cfg_float(cfg, "tactile_spatial_corr_weight")
    if spatial_corr_weight and pred.ndim >= 3 and target.ndim >= 3:
        loss = loss + spatial_corr_weight * _spatial_correlation_loss(pred, target)

    topk_weight = _cfg_float(cfg, "tactile_topk_weight")
    if topk_weight:
        loss = loss + topk_weight * _topk_abs_error_loss(
            pred,
            target,
            _cfg_float(cfg, "tactile_topk_count", 12),
            _cfg_float(cfg, "tactile_topk_ratio", 0.0),
        )

    raw_topk_weight = _cfg_float(cfg, "tactile_raw_topk_weight")
    if raw_topk_weight:
        loss = loss + raw_topk_weight * _raw_scaled_topk_abs_error_loss(
            pred,
            target,
            raw_scale,
            _cfg_float(cfg, "tactile_topk_count", 12),
            _cfg_float(cfg, "tactile_topk_ratio", 0.0),
        )

    return loss
