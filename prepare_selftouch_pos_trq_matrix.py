#!/usr/bin/env python3
import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "parameter" / "selftouch_fcn" / "parameter_base" / "parameter_base.yaml"
TACTILE_ACCURACY_MODE = "within_tolerance"
TACTILE_ACCURACY_TOLERANCE = 200.0
TACTILE_ACTIVE_TAXEL_THRESHOLD = 200.0
TACTILE_PEAK_TAXEL_RATIO = 0.05
SELFTOUCH_SEQUENCE_LENGTH = 400
TACTILE_RAW_CLIP_VALUE = 5000.0
TACTILE_MODALITY_KEYS = (
    "tactile_index_tip",
    "tactile_thumb_tip",
    "tactile_middle_tip",
    "tactile_ring_tip",
)
TACTILE_HISTORY_STEPS = 2
TACTILE_HISTORY_FINGERS = ["index", "thumb", "middle", "ring"]


COMMON_MODEL = {
    "hand_dim": 16,
    "use_joint_pos_only": False,
    "tactile_dim": 90,
    "input_modalities": ["hand_jnt_pos", "hand_jnt_trq"],
    "use_finger_aware_input": True,
    "finger_feature_dim": 32,
    "exclude_combo_from_encoder": True,
    "hidden_dim": 128,
    "encoder_dim": 128,
    "decoder_dim": 128,
    "decoder_out_dim": 128,
    "d_model": 128,
    "dropout": 0.1,
    "combo_decoder_dim": 256,
    "use_combo_condition": False,
    "combo_dim": 6,
    "use_phase_condition": False,
    "phase_dim": 10,
    "use_mean_residual_head": False,
    "use_tactile_history": True,
    "tactile_history_steps": TACTILE_HISTORY_STEPS,
    "tactile_history_fingers": TACTILE_HISTORY_FINGERS,
    "temporal_window_steps": 1,
    "use_derived_features": False,
    "output_activation": "identity",
    "output_min": 0.1,
    "output_max": 0.9,
    "projection_hidden_dim": 128,
    "projection_dim": 64,
    "num_classes": 6,
}


MODEL_SPECS = [
    {
        "name": "selftouch_fcn",
        "architecture": "pos_trq_tcn",
        "backbone": "tcn",
        "extra": {
            "rec_dim": 128,
            "temporal_blocks": 6,
            "temporal_kernel_size": 5,
            "temporal_dilations": [1, 2, 4, 8, 16, 32],
        },
    },
    {
        "name": "selftouch_transformer",
        "architecture": "pos_trq_causal_transformer",
        "backbone": "patchtst",
        "extra": {
            "rec_dim": 128,
            "nhead": 4,
            "encoder_layers": 4,
            "patch_len": 1,
            "patch_stride": 1,
            "ffn_dim": 512,
            "causal": True,
            "max_seq_len": 512,
        },
    },
    {
        "name": "selftouch_gru_attention",
        "architecture": "pos_trq_causal_gru_attention",
        "backbone": "gru_attention",
        "extra": {
            "hidden_dim": 256,
            "encoder_dim": 256,
            "decoder_dim": 256,
            "decoder_out_dim": 256,
            "d_model": 256,
            "rec_dim": 256,
            "finger_feature_dim": 48,
            "combo_decoder_dim": 384,
            "dropout": 0.08,
            "nhead": 8,
            "gru_layers": 3,
            "bidirectional_gru": False,
            "causal": True,
            "ffn_dim": 768,
            "use_tactile_history": False,
            "tactile_history_steps": 1,
            "tactile_history_fingers": TACTILE_HISTORY_FINGERS,
        },
    },
    {
        "name": "selftouch_temporal_mixer",
        "architecture": "pos_trq_tsmixer",
        "backbone": "tsmixer",
        "extra": {
            "rec_dim": 128,
            "mixer_layers": 6,
            "mixer_seq_len": 399,
            "mixer_use_patches": True,
            "patch_len": 4,
            "patch_stride": 2,
            "time_mixing_expansion": 2.0,
            "channel_mixing_expansion": 2.0,
        },
    },
    {
        "name": "selftouch_mamba",
        "architecture": "pos_trq_mamba",
        "backbone": "mamba",
        "extra": {
            "rec_dim": 128,
            "mamba_layers": 6,
            "mamba_d_state": 16,
            "mamba_expand": 2,
            "mamba_conv_kernel": 4,
            "mamba_fast_scan": True,
        },
    },
    {
        "name": "selftouch_contrastive_fcn",
        "architecture": "pos_trq_contrastive_tcn",
        "backbone": "tcn",
        "contrastive": True,
        "extra": {
            "rec_dim": 128,
            "temporal_blocks": 6,
            "temporal_kernel_size": 5,
            "temporal_dilations": [1, 2, 4, 8, 16, 32],
        },
    },
    {
        "name": "selftouch_contrastive_transformer",
        "architecture": "pos_trq_contrastive_causal_transformer",
        "backbone": "patchtst",
        "contrastive": True,
        "extra": {
            "rec_dim": 128,
            "nhead": 4,
            "encoder_layers": 4,
            "patch_len": 1,
            "patch_stride": 1,
            "ffn_dim": 512,
            "causal": True,
            "max_seq_len": 512,
        },
    },
    {
        "name": "selftouch_contrastive_gru",
        "architecture": "pos_trq_contrastive_causal_gru_attention",
        "backbone": "gru_attention",
        "contrastive": True,
        "extra": {
            "hidden_dim": 256,
            "encoder_dim": 256,
            "decoder_dim": 256,
            "decoder_out_dim": 256,
            "d_model": 256,
            "rec_dim": 256,
            "finger_feature_dim": 48,
            "combo_decoder_dim": 384,
            "dropout": 0.08,
            "nhead": 8,
            "gru_layers": 3,
            "bidirectional_gru": False,
            "causal": True,
            "ffn_dim": 768,
            "use_tactile_history": False,
            "tactile_history_steps": 1,
            "tactile_history_fingers": TACTILE_HISTORY_FINGERS,
        },
    },
    {
        "name": "selftouch_contrastive_temporal",
        "architecture": "pos_trq_contrastive_tsmixer",
        "backbone": "tsmixer",
        "contrastive": True,
        "extra": {
            "rec_dim": 128,
            "mixer_layers": 6,
            "mixer_seq_len": 399,
            "mixer_use_patches": True,
            "patch_len": 4,
            "patch_stride": 2,
            "time_mixing_expansion": 2.0,
            "channel_mixing_expansion": 2.0,
        },
    },
    {
        "name": "selftouch_contrastive_mamba",
        "architecture": "pos_trq_contrastive_mamba",
        "backbone": "mamba",
        "contrastive": True,
        "extra": {
            "rec_dim": 128,
            "mamba_layers": 6,
            "mamba_d_state": 16,
            "mamba_expand": 2,
            "mamba_conv_kernel": 4,
            "mamba_fast_scan": True,
        },
    },
]


def load_template():
    with TEMPLATE.open("r") as handle:
        return yaml.safe_load(handle)


def one_value(value):
    return {"values": [copy.deepcopy(value)]}


def build_tune(model):
    model_keys = [
        "rec_dim",
        "dropout",
        "hidden_dim",
        "encoder_dim",
        "decoder_dim",
        "decoder_out_dim",
        "d_model",
        "use_finger_aware_input",
        "finger_feature_dim",
        "exclude_combo_from_encoder",
        "combo_decoder_dim",
        "use_combo_condition",
        "combo_dim",
        "use_phase_condition",
        "phase_dim",
        "use_mean_residual_head",
        "use_tactile_history",
        "tactile_history_steps",
        "tactile_history_fingers",
        "temporal_window_steps",
        "use_derived_features",
        "output_activation",
        "temporal_blocks",
        "temporal_kernel_size",
        "temporal_dilations",
        "nhead",
        "encoder_layers",
        "ffn_dim",
        "patch_len",
        "patch_stride",
        "max_seq_len",
        "gru_layers",
        "bidirectional_gru",
        "mixer_layers",
        "mixer_seq_len",
        "mixer_use_patches",
        "time_mixing_expansion",
        "channel_mixing_expansion",
        "mamba_layers",
        "mamba_d_state",
        "mamba_expand",
        "mamba_conv_kernel",
        "mamba_fast_scan",
        "projection_hidden_dim",
        "projection_dim",
        "num_classes",
        "causal",
    ]
    return {
        "Model": {
            key: one_value(model[key])
            for key in model_keys
            if key in model
        },
        "Train": {
            "batch_size": one_value(64),
            "lr": one_value(0.0002),
            "num_epochs": one_value(500),
            "early_stop_enabled": one_value(False),
        },
        "Dataset": {
            "sequence_length": one_value(SELFTOUCH_SEQUENCE_LENGTH),
            "shift_data": one_value(1),
            "test_split_policy": one_value("blocked_tail"),
            "test_split_fraction": one_value(0.2),
            "add_selftouch_combo_condition": one_value(True),
            "add_selftouch_phase_condition": one_value(False),
            "tactile_accuracy_mode": one_value(TACTILE_ACCURACY_MODE),
            "tactile_accuracy_tolerance": one_value(TACTILE_ACCURACY_TOLERANCE),
            "tactile_active_taxel_threshold": one_value(TACTILE_ACTIVE_TAXEL_THRESHOLD),
            "tactile_peak_taxel_ratio": one_value(TACTILE_PEAK_TAXEL_RATIO),
            "tactile_raw_clip_value": one_value(TACTILE_RAW_CLIP_VALUE),
        },
    }


def normalize_loss(loss_cfg, contrastive):
    cfg = copy.deepcopy(loss_cfg)
    if contrastive:
        cfg.setdefault("contrastive", 0.35)
        cfg.setdefault("classification", 0.25)
        cfg.setdefault("contrastive_temperature", 0.05)
        cfg.setdefault("contact_overlap_contrastive", True)
    else:
        cfg.pop("contrastive", None)
        cfg.pop("classification", None)
        cfg.pop("contrastive_temperature", None)
        cfg.pop("contact_overlap_contrastive", None)
    return cfg


def fast_loss_profile(loss_cfg, contrastive):
    cfg = copy.deepcopy(loss_cfg)
    keep = {
        "tactile_index_tip",
        "tactile_thumb_tip",
        "tactile_middle_tip",
        "tactile_ring_tip",
        "tactile_loss",
        "tactile_huber_beta",
        "tactile_mae_weight",
        "tactile_raw_l1_weight",
        "tactile_raw_huber_weight",
        "tactile_raw_contact_weight",
        "tactile_contact_mae_weight",
        "tactile_raw_contact_threshold",
        "tactile_raw_active_margin_weight",
        "tactile_raw_active_margin",
        "tactile_raw_topk_weight",
        "tactile_raw_error_loss_scale",
        "tactile_raw_error_clip",
        "tactile_raw_huber_beta",
        "tactile_raw_mean_loss_scale",
        "tactile_raw_timestep_mean_weight",
        "tactile_raw_taxel_mean_weight",
        "tactile_timestep_mean_weight",
        "tactile_contact_ratio",
        "tactile_inactive_weight",
        "tactile_raw_inactive_weight",
        "tactile_topk_count",
        "tactile_topk_ratio",
        "contrastive",
        "classification",
        "contrastive_temperature",
        "contact_overlap_contrastive",
    }
    fast = {key: copy.deepcopy(value) for key, value in cfg.items() if key in keep}
    fast["tactile_index_tip"] = 3.4
    fast["tactile_thumb_tip"] = 3.4
    fast["tactile_middle_tip"] = 2.4
    fast["tactile_ring_tip"] = 2.8
    fast["tactile_fast_loss"] = True
    fast["tactile_loss"] = "huber"
    fast["tactile_huber_beta"] = 0.03
    fast["tactile_mae_weight"] = 1.0
    fast["tactile_raw_l1_weight"] = 2.5
    fast["tactile_raw_huber_weight"] = 0.8
    fast["tactile_raw_contact_weight"] = 12.0
    fast["tactile_contact_mae_weight"] = 2.0
    fast["tactile_raw_contact_threshold"] = TACTILE_ACTIVE_TAXEL_THRESHOLD
    fast["tactile_raw_active_margin_weight"] = 4.0
    fast["tactile_raw_active_margin"] = TACTILE_ACCURACY_TOLERANCE
    fast["tactile_raw_topk_weight"] = 4.0
    fast["tactile_raw_error_loss_scale"] = 200.0
    fast["tactile_raw_error_clip"] = 0.0
    fast["tactile_raw_huber_beta"] = 120.0
    fast["tactile_raw_mean_loss_scale"] = 200.0
    fast["tactile_raw_timestep_mean_weight"] = 0.20
    fast["tactile_raw_taxel_mean_weight"] = 0.15
    fast["tactile_timestep_mean_weight"] = 0.04
    fast["tactile_contact_ratio"] = 0.35
    fast["tactile_inactive_weight"] = 0.4
    fast["tactile_raw_inactive_weight"] = 0.75
    fast["tactile_topk_count"] = 0
    fast["tactile_topk_ratio"] = 0.50
    if contrastive:
        fast["contrastive"] = 0.35
        fast["classification"] = 0.25
        fast["contrastive_temperature"] = 0.05
        fast["contact_overlap_contrastive"] = True
    else:
        fast.pop("contrastive", None)
        fast.pop("classification", None)
        fast.pop("contrastive_temperature", None)
        fast.pop("contact_overlap_contrastive", None)
    return fast


def build_config(template, spec):
    cfg = copy.deepcopy(template)
    name = spec["name"]
    contrastive = bool(spec.get("contrastive", False))
    model = copy.deepcopy(COMMON_MODEL)
    model.update(spec.get("extra", {}))
    history_enabled = bool(model.get("use_tactile_history", False))
    history_steps = int(model.get("tactile_history_steps", 0) or 0)
    if history_enabled:
        condition = f"pos_trq_tactile_history_{history_steps}step"
        history_tag = f"tactile-history-{history_steps}step"
        group = f"pos-trq-tactile-history-{history_steps}step"
    else:
        condition = "pos_trq_no_tactile_history"
        history_tag = "no-tactile-history"
        group = "pos-trq-no-tactile-history"
    model.update(
        {
            "model": name,
            "model_name": name,
            "architecture": spec["architecture"],
            "backbone": spec["backbone"],
            "contrastive_encoder": contrastive,
            "experiment_condition": condition,
        }
    )
    cfg["Model"] = model
    cfg.setdefault("Required", {})["controller_name"] = "RNN_controller"
    cfg["Required"]["model_name"] = name
    cfg["Dataset"]["sequence_length"] = SELFTOUCH_SEQUENCE_LENGTH
    cfg["Dataset"]["combinations"] = [
        "thumb-index",
        "thumb-middle",
        "index-middle",
        "middle-ring",
        "index-middle-ring",
        "thumb-index-middle",
    ]
    cfg["Dataset"]["add_selftouch_combo_condition"] = True
    cfg["Dataset"]["add_selftouch_phase_condition"] = False
    cfg["Dataset"]["tactile_accuracy_mode"] = TACTILE_ACCURACY_MODE
    cfg["Dataset"]["tactile_accuracy_tolerance"] = TACTILE_ACCURACY_TOLERANCE
    cfg["Dataset"]["tactile_active_taxel_threshold"] = TACTILE_ACTIVE_TAXEL_THRESHOLD
    cfg["Dataset"]["tactile_peak_taxel_ratio"] = TACTILE_PEAK_TAXEL_RATIO
    cfg["Dataset"]["tactile_raw_clip_value"] = TACTILE_RAW_CLIP_VALUE
    for tactile_key in TACTILE_MODALITY_KEYS:
        if tactile_key in cfg["Dataset"].get("modality", {}):
            cfg["Dataset"]["modality"][tactile_key][0] = "rn"

    cfg["Train"]["project"] = name
    cfg["Train"]["tags"] = [
        "pos-trq",
        history_tag,
    ]
    cfg["Train"]["group"] = group
    cfg["Train"]["batch_size"] = 64
    cfg["Train"]["num_epochs"] = 500
    cfg["Train"]["deterministic"] = False
    cfg["Train"]["model_save_iter"] = 0
    cfg["Train"]["save_initial_model"] = False
    cfg["Train"]["save_final_model"] = True
    cfg["Train"]["export_combo_paths"] = True
    cfg["Train"]["export_combo_paths_dir"] = "selftouch_paths"
    cfg["Train"]["eval_every"] = 50
    cfg["Train"]["plot_enabled"] = False
    cfg["Train"]["plot_every"] = 0
    cfg["Train"]["wandb_log_images"] = False
    cfg["Train"]["wandb_log_tactile_metrics"] = True
    cfg["Train"]["wandb_log_pca"] = True
    cfg["Train"]["wandb_log_tactile_profile"] = True
    cfg["Train"]["wandb_image_keys"] = [
        "tactile_profile",
        "combination_pca",
        "latent_combination_pca",
    ]
    cfg["Train"]["early_stop_enabled"] = False
    cfg["Train"]["early_stop_monitor"] = "total_loss"
    cfg["Train"]["early_stop_mode"] = "min"
    cfg["Train"]["early_stop_patience"] = 2
    cfg["Train"]["early_stop_min_delta"] = 0.0005
    cfg["Train"]["early_stop_min_epochs"] = 150
    cfg["Train"]["eval_batch_size"] = 64
    cfg["Train"]["eval_micro_batch_size"] = 0
    cfg["Train"]["train_micro_batch_size"] = 0
    cfg["Train"]["init_output_bias_from_tactile_mean"] = True
    cfg["Train"]["zero_output_head_on_bias_init"] = True
    cfg["Train"]["train_step_sleep_seconds"] = 0.0
    cfg["Train"]["empty_cache_after_save"] = False
    cfg["Train"]["empty_cache_after_eval"] = False
    cfg["Train"]["cuda_memory_fraction"] = None
    cfg["Train"]["adamw_foreach"] = True
    cfg["Train"]["max_train_batches_per_epoch"] = 0
    cfg["Train"]["loss_coef"] = fast_loss_profile(
        normalize_loss(cfg["Train"]["loss_coef"], contrastive),
        contrastive,
    )
    cfg["Test"]["loss_coef"] = fast_loss_profile(
        normalize_loss(cfg["Test"]["loss_coef"], contrastive),
        contrastive,
    )
    cfg["Test"]["model_load_path"] = str(
        ROOT / "model_weight" / name / "parameter_base" / "epoch499.pth"
    )

    cfg["Sweep"] = {
        "tune": build_tune(model),
        "method": "grid",
        "metric": {"name": "active_taxel_acc", "goal": "maximize"},
        "project": name,
        "wandb_entity": cfg["Train"].get("wandb_entity"),
        "use_wandb_sweep_api": False,
    }
    return cfg


def write_config(name, cfg):
    out_dir = ROOT / "parameter" / name / "parameter_base"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "parameter_base.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    (ROOT / "model_weight" / name / "parameter_base").mkdir(parents=True, exist_ok=True)


def main():
    template = load_template()
    for spec in MODEL_SPECS:
        cfg = build_config(template, spec)
        write_config(spec["name"], cfg)
        print(f"wrote parameter/{spec['name']}/parameter_base/parameter_base.yaml")


if __name__ == "__main__":
    main()
