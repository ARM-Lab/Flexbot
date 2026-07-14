"""Create basic plots from PPO training and evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from common import load_config, resolve_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--rolling-window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config, base_dir = load_config(args.config)
    output_dir = Path(resolve_path(base_dir, config["paths"]["output_dir"]))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    monitor_dir = output_dir / "monitor"
    if monitor_dir.is_dir():
        monitor_data = load_results(str(monitor_dir))
        x, rewards = ts2xy(monitor_data, "timesteps")
        if len(rewards) > 0:
            rolling = pd.Series(rewards).rolling(
                window=args.rolling_window, min_periods=1
            ).mean()
            plt.figure(figsize=(8, 5))
            plt.plot(x, rolling)
            plt.xlabel("Training timesteps")
            plt.ylabel("Rolling mean episode reward")
            plt.tight_layout()
            plt.savefig(figure_dir / "training_reward.png", dpi=200)
            plt.close()

    episode_file = output_dir / "final_evaluation" / "evaluation_episodes.csv"
    if episode_file.is_file():
        episodes = pd.read_csv(episode_file)
        plt.figure(figsize=(8, 5))
        plt.hist(episodes["final_error_mm"], bins=20)
        plt.xlabel("Final Cartesian error [mm]")
        plt.ylabel("Number of episodes")
        plt.tight_layout()
        plt.savefig(figure_dir / "final_error_histogram.png", dpi=200)
        plt.close()

    print(f"Figures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
