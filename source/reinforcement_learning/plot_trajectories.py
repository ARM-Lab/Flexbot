"""Plot the Cartesian trajectory of one evaluated PPO episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/ppo_one_segment/"
            "final_evaluation/evaluation_trajectories.csv"
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode number to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ppo_one_segment/figures"),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Trajectory file not found: {args.input}\n"
            "Run evaluate_ppo.py first."
        )

    data = pd.read_csv(args.input)

    episode_data = (
        data[data["episode"] == args.episode]
        .sort_values("step")
        .reset_index(drop=True)
    )

    if episode_data.empty:
        available = sorted(data["episode"].unique().tolist())
        raise ValueError(
            f"Episode {args.episode} not found. "
            f"Available episodes: {available}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = episode_data["tip_x_mm"].to_numpy()
    y = episode_data["tip_y_mm"].to_numpy()
    z = episode_data["tip_z_mm"].to_numpy()

    target_x = float(episode_data["target_x_mm"].iloc[0])
    target_y = float(episode_data["target_y_mm"].iloc[0])
    target_z = float(episode_data["target_z_mm"].iloc[0])

    # 3D Cartesian trajectory
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")

    axis.plot(x, y, z, marker="o", markersize=3, label="PPO trajectory")
    axis.scatter(
        x[0],
        y[0],
        z[0],
        marker="s",
        s=80,
        label="Start",
    )
    axis.scatter(
        x[-1],
        y[-1],
        z[-1],
        marker="^",
        s=80,
        label="Final position",
    )
    axis.scatter(
        target_x,
        target_y,
        target_z,
        marker="*",
        s=180,
        label="Target",
    )

    axis.set_xlabel("X [mm]")
    axis.set_ylabel("Y [mm]")
    axis.set_zlabel("Z [mm]")
    axis.set_title(f"PPO Cartesian trajectory — episode {args.episode}")
    axis.legend()
    figure.tight_layout()

    trajectory_path = (
        args.output_dir / f"trajectory_3d_episode_{args.episode}.png"
    )
    figure.savefig(trajectory_path, dpi=300)

    if args.show:
        plt.show()

    plt.close(figure)

    # Cartesian error during the episode
    figure = plt.figure(figsize=(8, 5))
    plt.plot(
        episode_data["step"],
        episode_data["distance_mm"],
        marker="o",
        markersize=3,
    )
    plt.xlabel("Control step")
    plt.ylabel("Cartesian error [mm]")
    plt.title(f"Target error — episode {args.episode}")
    plt.grid(True)
    plt.tight_layout()

    error_path = (
        args.output_dir / f"error_episode_{args.episode}.png"
    )
    figure.savefig(error_path, dpi=300)

    if args.show:
        plt.show()

    plt.close(figure)

    # Cable trajectories
    figure = plt.figure(figsize=(8, 5))

    plt.plot(
        episode_data["step"],
        episode_data["cable_1_mm"],
        label="Cable 1",
    )
    plt.plot(
        episode_data["step"],
        episode_data["cable_2_mm"],
        label="Cable 2",
    )
    plt.plot(
        episode_data["step"],
        episode_data["cable_3_mm"],
        label="Cable 3",
    )

    plt.xlabel("Control step")
    plt.ylabel("Cable shortening [mm]")
    plt.title(f"Cable commands — episode {args.episode}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    cable_path = (
        args.output_dir / f"cables_episode_{args.episode}.png"
    )
    figure.savefig(cable_path, dpi=300)

    if args.show:
        plt.show()

    plt.close(figure)

    final_error = float(episode_data["distance_mm"].iloc[-1])

    print(f"Episode: {args.episode}")
    print(f"Steps: {int(episode_data['step'].iloc[-1])}")
    print(f"Final Cartesian error: {final_error:.4f} mm")
    print(f"Saved 3D trajectory to: {trajectory_path}")
    print(f"Saved error plot to: {error_path}")
    print(f"Saved cable plot to: {cable_path}")


if __name__ == "__main__":
    main()