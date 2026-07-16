"""Evaluate a trained PPO policy and save quantitative results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from common import build_environment, load_config, resolve_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--model",
        default=None,
        help="PPO model path. Defaults to the best model, then the final model.",
    )
    parser.add_argument(
        "--neutral-start",
        action="store_true",
        help="Evaluate every episode from the neutral cable configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config, base_dir = load_config(args.config)
    seed = int(config.get("seed", 42))

    output_dir = Path(resolve_path(base_dir, config["paths"]["output_dir"]))
    results_dir = output_dir / "final_evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.model is not None:
        model_path = Path(args.model).expanduser().resolve()
    else:
        best_model = output_dir / "models" / "best_model.zip"
        final_model = output_dir / "models" / "ppo_continuum_final.zip"
        model_path = best_model if best_model.is_file() else final_model

    if not model_path.is_file():
        raise FileNotFoundError(f"PPO model not found: {model_path}")

    initial_mode = "neutral" if args.neutral_start else None
    env = build_environment(config, base_dir, initial_state_mode=initial_mode)
    model = PPO.load(model_path)

    evaluation_cfg = config["evaluation"]
    episodes = int(evaluation_cfg["episodes"])
    deterministic = bool(evaluation_cfg.get("deterministic", True))

    episode_rows: list[dict] = []
    trajectory_rows: list[dict] = []

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0
        step = 0

        trajectory_rows.append(
            {
                "episode": episode,
                "step": 0,
                "reward": 0.0,
                "distance_mm": info["distance_mm"],
                "cable_1_units": info["cable_position_units"][0],
                "cable_2_units": info["cable_position_units"][1],
                "cable_3_units": info["cable_position_units"][2],
                "cable_1_mm": info["cable_position_mm"][0],
                "cable_2_mm": info["cable_position_mm"][1],
                "cable_3_mm": info["cable_position_mm"][2],
                "active_cables": info["active_cables"],
                "tip_x_mm": info["tip_position_mm"][0],
                "tip_y_mm": info["tip_position_mm"][1],
                "tip_z_mm": info["tip_position_mm"][2],
                "target_x_mm": info["target_position_mm"][0],
                "target_y_mm": info["target_position_mm"][1],
                "target_z_mm": info["target_position_mm"][2],
            }
        )

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
            episode_reward += float(reward)

            trajectory_rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "reward": reward,
                    "distance_mm": info["distance_mm"],
                    "cable_1_units": info["cable_position_units"][0],
                    "cable_2_units": info["cable_position_units"][1],
                    "cable_3_units": info["cable_position_units"][2],
                    "cable_1_mm": info["cable_position_mm"][0],
                    "cable_2_mm": info["cable_position_mm"][1],
                    "cable_3_mm": info["cable_position_mm"][2],
                    "active_cables": info["active_cables"],
                    "tip_x_mm": info["tip_position_mm"][0],
                    "tip_y_mm": info["tip_position_mm"][1],
                    "tip_z_mm": info["tip_position_mm"][2],
                    "target_x_mm": info["target_position_mm"][0],
                    "target_y_mm": info["target_position_mm"][1],
                    "target_z_mm": info["target_position_mm"][2],
                }
            )

        episode_rows.append(
            {
                "episode": episode,
                "success": bool(info["is_success"]),
                "steps": step,
                "episode_reward": episode_reward,
                "final_error_mm": float(info["distance_mm"]),
                "target_x_mm": float(info["target_position_mm"][0]),
                "target_y_mm": float(info["target_position_mm"][1]),
                "target_z_mm": float(info["target_position_mm"][2]),
                "final_cable_1_units": float(info["cable_position_units"][0]),
                "final_cable_2_units": float(info["cable_position_units"][1]),
                "final_cable_3_units": float(info["cable_position_units"][2]),
                "final_cable_1_mm": float(info["cable_position_mm"][0]),
                "final_cable_2_mm": float(info["cable_position_mm"][1]),
                "final_cable_3_mm": float(info["cable_position_mm"][2]),
                "final_active_cables": int(info["active_cables"]),
            }
        )

    env.close()

    episodes_frame = pd.DataFrame(episode_rows)
    trajectories_frame = pd.DataFrame(trajectory_rows)
    episodes_frame.to_csv(results_dir / "evaluation_episodes.csv", index=False)
    trajectories_frame.to_csv(results_dir / "evaluation_trajectories.csv", index=False)

    final_errors = episodes_frame["final_error_mm"].to_numpy(dtype=float)
    summary = {
        "model": str(model_path),
        "episodes": episodes,
        "success_rate": float(episodes_frame["success"].mean()),
        "mean_final_error_mm": float(np.mean(final_errors)),
        "median_final_error_mm": float(np.median(final_errors)),
        "p95_final_error_mm": float(np.percentile(final_errors, 95)),
        "maximum_final_error_mm": float(np.max(final_errors)),
        "mean_steps": float(episodes_frame["steps"].mean()),
        "mean_episode_reward": float(episodes_frame["episode_reward"].mean()),
        "initial_state_mode": initial_mode or config["environment"]["initial_state_mode"],
    }
    with (results_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Evaluation results saved to: {results_dir}")


if __name__ == "__main__":
    main()
