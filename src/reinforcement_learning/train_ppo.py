"""Train PPO for one continuum segment actuated by three cables."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from common import build_environment, load_config, resolve_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps for a short test run.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config, base_dir = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_random_seed(seed)

    output_dir = Path(resolve_path(base_dir, config["paths"]["output_dir"]))
    model_dir = output_dir / "models"
    monitor_dir = output_dir / "monitor"
    checkpoint_dir = output_dir / "checkpoints"
    eval_dir = output_dir / "evaluation"
    tensorboard_dir = output_dir / "tensorboard"
    for directory in [model_dir, monitor_dir, checkpoint_dir, eval_dir, tensorboard_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Validate one non-vectorised environment before training.
    validation_env = build_environment(config, base_dir)
    check_env(validation_env, warn=True)
    validation_env.close()

    training_cfg = config["training"]
    n_envs = int(training_cfg.get("n_envs", 1))

    def make_training_env(rank: int):
        def _factory():
            env = build_environment(config, base_dir)
            env.reset(seed=seed + rank)
            return Monitor(env, filename=str(monitor_dir / f"env_{rank}"))

        return _factory

    train_env = DummyVecEnv([make_training_env(rank) for rank in range(n_envs)])

    def make_eval_env():
        env = build_environment(config, base_dir)
        env.reset(seed=seed + 10000)
        return Monitor(env)

    eval_env = DummyVecEnv([make_eval_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(training_cfg["checkpoint_frequency_steps"]) // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_continuum",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(eval_dir),
        eval_freq=max(int(training_cfg["eval_frequency_steps"]) // n_envs, 1),
        n_eval_episodes=int(training_cfg["evaluation_episodes"]),
        deterministic=True,
        render=False,
    )

    hidden_sizes = list(training_cfg.get("policy_hidden_sizes", [128, 128]))
    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {"pi": hidden_sizes, "vf": hidden_sizes},
    }

    tensorboard_log = None
    if importlib.util.find_spec("tensorboard") is not None:
        tensorboard_log = str(tensorboard_dir)
    else:
        print("TensorBoard is not installed; TensorBoard logging is disabled.")

    progress_bar = bool(training_cfg.get("progress_bar", True))
    if progress_bar and (
        importlib.util.find_spec("tqdm") is None
        or importlib.util.find_spec("rich") is None
    ):
        print("tqdm/rich is not installed; the training progress bar is disabled.")
        progress_bar = False

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(training_cfg["learning_rate"]),
        n_steps=int(training_cfg["n_steps"]),
        batch_size=int(training_cfg["batch_size"]),
        n_epochs=int(training_cfg["n_epochs"]),
        gamma=float(training_cfg["gamma"]),
        gae_lambda=float(training_cfg["gae_lambda"]),
        clip_range=float(training_cfg["clip_range"]),
        ent_coef=float(training_cfg["ent_coef"]),
        vf_coef=float(training_cfg["vf_coef"]),
        max_grad_norm=float(training_cfg["max_grad_norm"]),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=1,
        device="auto",
    )

    model.learn(
        total_timesteps=int(args.timesteps if args.timesteps is not None else training_cfg["total_timesteps"]),
        callback=[checkpoint_callback, eval_callback],
        progress_bar=progress_bar,
    )
    model.save(model_dir / "ppo_continuum_final")

    train_env.close()
    eval_env.close()
    print(f"Training complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
