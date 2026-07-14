"""Shared configuration and object-construction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from continuum_env import CableDrivenContinuumEnv
from forward_model import ForwardModelPredictor


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The YAML configuration must contain a dictionary at its root.")
    return config, path.parent


def resolve_path(base_dir: Path, value: str) -> str:
    return str((base_dir / value).expanduser().resolve())


def build_predictor(config: dict[str, Any], base_dir: Path) -> ForwardModelPredictor:
    path_cfg = config["paths"]
    model_cfg = config["forward_model"]
    return ForwardModelPredictor(
        checkpoint_path=resolve_path(base_dir, path_cfg["forward_checkpoint"]),
        checkpoint_key=model_cfg.get("checkpoint_key", "model_state_dict"),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        input_scaler_path=resolve_path(base_dir, path_cfg["input_scaler"]),
        output_scaler_path=resolve_path(base_dir, path_cfg["output_scaler"]),
        device=model_cfg.get("device", "cpu"),
    )


def build_environment(
    config: dict[str, Any],
    base_dir: Path,
    *,
    initial_state_mode: str | None = None,
) -> CableDrivenContinuumEnv:
    predictor = build_predictor(config, base_dir)
    path_cfg = config["paths"]
    data_cfg = config["dataset"]
    env_cfg = config["environment"]
    reward_cfg = env_cfg["reward"]

    return CableDrivenContinuumEnv(
        predictor=predictor,
        dataset_path=resolve_path(base_dir, path_cfg["dataset"]),
        dataset_has_header=bool(data_cfg.get("has_header", False)),
        dataset_delimiter=str(data_cfg.get("delimiter", ",")),
        column_names=list(data_cfg.get("column_names", ["a1", "a2", "a3", "x", "y", "z"])),
        cable_columns=list(data_cfg["cable_columns"]),
        position_columns=list(data_cfg["position_columns"]),
        cable_unit_to_mm=float(data_cfg["cable_unit_to_mm"]),
        max_active_cables=int(data_cfg.get("max_active_cables", 2)),
        active_threshold_units=float(data_cfg.get("active_threshold_units", 1e-6)),
        invalid_row_policy=data_cfg.get("invalid_row_policy", "error"),
        cable_min_units=list(env_cfg["cable_min_units"]),
        cable_max_units=list(env_cfg["cable_max_units"]),
        max_delta_units=float(env_cfg["max_delta_units"]),
        enforce_measured_cable_domain=bool(env_cfg.get("enforce_measured_cable_domain", True)),
        max_episode_steps=int(env_cfg["max_episode_steps"]),
        success_tolerance_mm=float(env_cfg["success_tolerance_mm"]),
        minimum_start_distance_mm=float(env_cfg["minimum_start_distance_mm"]),
        initial_state_mode=initial_state_mode or env_cfg["initial_state_mode"],
        progress_weight=float(reward_cfg["progress_weight"]),
        distance_weight=float(reward_cfg["distance_weight"]),
        action_weight=float(reward_cfg["action_weight"]),
        boundary_weight=float(reward_cfg["boundary_weight"]),
        success_bonus=float(reward_cfg["success_bonus"]),
    )
