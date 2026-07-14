"""Verify dataset, checkpoint, scalers and forward-model predictions before PPO."""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import build_environment, load_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config, base_dir = load_config(args.config)
    env = build_environment(config, base_dir)

    sample_count = min(max(args.samples, 1), len(env.dataset_cables))
    rng = np.random.default_rng(int(config.get("seed", 42)))
    indices = rng.choice(len(env.dataset_cables), size=sample_count, replace=False)

    predicted = env.predictor.predict(env.dataset_cables[indices])
    measured = env.dataset_positions[indices]
    axis_abs_error = np.abs(predicted - measured)
    euclidean_error = np.linalg.norm(predicted - measured, axis=1)

    observation, info = env.reset(seed=int(config.get("seed", 42)))
    next_observation, reward, terminated, truncated, step_info = env.step(
        env.action_space.sample()
    )

    summary = {
        "checkpoint": str(env.predictor.checkpoint_path),
        "forward_hidden_dim": env.predictor.hidden_dim,
        "forward_dropout": env.predictor.dropout,
        "dataset_samples": int(len(env.dataset_cables)),
        "checked_samples": int(sample_count),
        "mae_per_axis": axis_abs_error.mean(axis=0).tolist(),
        "euclidean_error_mean": float(euclidean_error.mean()),
        "euclidean_error_p95": float(np.percentile(euclidean_error, 95)),
        "observation_shape": list(observation.shape),
        "action_shape": list(env.action_space.shape),
        "test_step_reward": float(reward),
        "test_step_terminated": bool(terminated),
        "test_step_truncated": bool(truncated),
        "test_step_active_cables": int(step_info["active_cables"]),
    }
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
