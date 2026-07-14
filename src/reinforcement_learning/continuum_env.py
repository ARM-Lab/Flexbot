"""Gymnasium environment for one continuum segment driven by three cables."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from forward_model import ForwardModelPredictor


class CableDrivenContinuumEnv(gym.Env[np.ndarray, np.ndarray]):
    """Point-reaching environment using a learned forward model.

    The forward model receives the same cable units as the dataset. In the
    default configuration, cable values are in the range 0--100, where 100
    units correspond to 10 mm of shortening.

    To remain inside the measured domain, cable states are represented in a
    canonical differential form: after every action, the smallest of the three
    cable values is subtracted from all cables. At least one cable is therefore
    always zero, so no more than two cables are shortened simultaneously.

    Observation
    -----------
    - Three current cable shortenings, normalised from dataset units
    - Current tip position XYZ [mm]
    - Target tip position XYZ [mm]
    - Current Cartesian error target - tip [mm]

    Action
    ------
    A three-dimensional normalised action in [-1, 1]. Each component is
    converted to an increment in dataset cable units.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        predictor: ForwardModelPredictor,
        dataset_path: str,
        dataset_has_header: bool,
        dataset_delimiter: str,
        column_names: list[str],
        cable_columns: list[str],
        position_columns: list[str],
        cable_unit_to_mm: float,
        max_active_cables: int,
        active_threshold_units: float,
        invalid_row_policy: Literal["error", "drop"],
        cable_min_units: list[float],
        cable_max_units: list[float],
        max_delta_units: float = 2.5,
        enforce_measured_cable_domain: bool = True,
        max_episode_steps: int = 50,
        success_tolerance_mm: float = 2.0,
        minimum_start_distance_mm: float = 10.0,
        initial_state_mode: Literal["neutral", "random_dataset"] = "random_dataset",
        progress_weight: float = 10.0,
        distance_weight: float = 1.0,
        action_weight: float = 0.01,
        boundary_weight: float = 0.10,
        success_bonus: float = 10.0,
    ) -> None:
        super().__init__()

        self.predictor = predictor
        self.cable_columns = cable_columns
        self.position_columns = position_columns
        self.cable_unit_to_mm = float(cable_unit_to_mm)
        self.max_active_cables = int(max_active_cables)
        self.active_threshold_units = float(active_threshold_units)
        self.invalid_row_policy = invalid_row_policy
        self.cable_min = np.asarray(cable_min_units, dtype=np.float32)
        self.cable_max = np.asarray(cable_max_units, dtype=np.float32)
        self.max_delta_units = float(max_delta_units)
        self.enforce_measured_cable_domain = bool(enforce_measured_cable_domain)
        self.max_episode_steps = int(max_episode_steps)
        self.success_tolerance_mm = float(success_tolerance_mm)
        self.minimum_start_distance_mm = float(minimum_start_distance_mm)
        self.initial_state_mode = initial_state_mode

        self.progress_weight = float(progress_weight)
        self.distance_weight = float(distance_weight)
        self.action_weight = float(action_weight)
        self.boundary_weight = float(boundary_weight)
        self.success_bonus = float(success_bonus)

        if self.cable_min.shape != (3,) or self.cable_max.shape != (3,):
            raise ValueError(
                "cable_min_units and cable_max_units must each contain three values."
            )
        if np.any(self.cable_max <= self.cable_min):
            raise ValueError("Every cable maximum must be greater than its minimum.")
        if self.cable_unit_to_mm <= 0.0:
            raise ValueError("cable_unit_to_mm must be positive.")
        if self.max_delta_units <= 0.0:
            raise ValueError("max_delta_units must be positive.")
        if self.max_active_cables not in (1, 2, 3):
            raise ValueError("max_active_cables must be 1, 2, or 3.")
        if self.enforce_measured_cable_domain and self.max_active_cables != 2:
            raise ValueError(
                "The subtract-minimum projection implemented for the measured "
                "domain requires max_active_cables: 2."
            )
        if self.enforce_measured_cable_domain and not np.allclose(
            self.cable_min, self.cable_min[0]
        ):
            raise ValueError(
                "The measured-domain projection requires equal minimum values "
                "for all three cables."
            )
        if self.invalid_row_policy not in ("error", "drop"):
            raise ValueError("invalid_row_policy must be either 'error' or 'drop'.")

        if dataset_has_header:
            frame = pd.read_csv(dataset_path, sep=dataset_delimiter)
        else:
            frame = pd.read_csv(
                dataset_path,
                sep=dataset_delimiter,
                header=None,
                names=column_names,
            )

        required_columns = cable_columns + position_columns
        missing_columns = [name for name in required_columns if name not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Dataset is missing columns: {missing_columns}. "
                f"Available columns: {list(frame.columns)}"
            )

        numeric = frame[required_columns].apply(pd.to_numeric, errors="coerce").dropna()
        cable_values = numeric[cable_columns].to_numpy(dtype=np.float32)
        position_values = numeric[position_columns].to_numpy(dtype=np.float32)

        in_range = np.all(
            (cable_values >= self.cable_min - self.active_threshold_units)
            & (cable_values <= self.cable_max + self.active_threshold_units),
            axis=1,
        )
        active_counts = np.sum(
            cable_values > (self.cable_min + self.active_threshold_units), axis=1
        )
        valid_active_count = active_counts <= self.max_active_cables
        valid_rows = in_range & valid_active_count

        invalid_count = int(np.count_nonzero(~valid_rows))
        if invalid_count:
            message = (
                f"Dataset contains {invalid_count} invalid rows: cable values must "
                f"remain inside {self.cable_min.tolist()}--{self.cable_max.tolist()} "
                f"and at most {self.max_active_cables} cables may be active."
            )
            if self.invalid_row_policy == "error":
                raise ValueError(message)
            print(f"Warning: {message} Invalid rows were dropped.")
            cable_values = cable_values[valid_rows]
            position_values = position_values[valid_rows]

        self.dataset_cables = cable_values
        self.dataset_positions = position_values
        if len(self.dataset_positions) < 2:
            raise ValueError("The dataset must contain at least two valid samples.")

        self.workspace_low = self.dataset_positions.min(axis=0)
        self.workspace_high = self.dataset_positions.max(axis=0)
        self.workspace_span = np.maximum(self.workspace_high - self.workspace_low, 1e-6)
        self.workspace_diagonal = max(float(np.linalg.norm(self.workspace_span)), 1e-6)
        self.cable_span = self.cable_max - self.cable_min

        self.observation_space = spaces.Box(
            low=np.full(12, -5.0, dtype=np.float32),
            high=np.full(12, 5.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.cable_position = np.zeros(3, dtype=np.float32)
        self.tip_position = np.zeros(3, dtype=np.float32)
        self.target_position = np.zeros(3, dtype=np.float32)
        self.previous_distance = np.inf
        self.step_count = 0

    def cable_units_to_mm(self, values: np.ndarray) -> np.ndarray:
        """Convert dataset cable units to physical shortening in millimetres."""
        return np.asarray(values, dtype=np.float32) * self.cable_unit_to_mm

    def _normalise_cables(self, cables: np.ndarray) -> np.ndarray:
        return 2.0 * (cables - self.cable_min) / self.cable_span - 1.0

    def _normalise_position(self, position: np.ndarray) -> np.ndarray:
        return 2.0 * (position - self.workspace_low) / self.workspace_span - 1.0

    def _get_observation(self) -> np.ndarray:
        error = self.target_position - self.tip_position
        normalised_error = 2.0 * error / self.workspace_span
        observation = np.concatenate(
            [
                self._normalise_cables(self.cable_position),
                self._normalise_position(self.tip_position),
                self._normalise_position(self.target_position),
                normalised_error,
            ]
        )
        return observation.astype(np.float32)

    def _project_to_measured_domain(self, values: np.ndarray) -> np.ndarray:
        """Project a three-cable state into the dataset's valid actuation domain.

        The dataset contains only differential shortening combinations: zero,
        one, or two shortened cables. Subtracting the smallest cable value from
        all three preserves pairwise cable differences and guarantees that at
        least one cable is at its minimum.
        """
        clipped = np.clip(values, self.cable_min, self.cable_max).astype(np.float32)
        if not self.enforce_measured_cable_domain:
            return clipped

        common_shortening = float(np.min(clipped - self.cable_min))
        projected = clipped - common_shortening
        projected = np.clip(projected, self.cable_min, self.cable_max)
        return projected.astype(np.float32)

    def _sample_target(self) -> np.ndarray:
        best_target = self.dataset_positions[
            int(self.np_random.integers(len(self.dataset_positions)))
        ]
        for _ in range(100):
            index = int(self.np_random.integers(len(self.dataset_positions)))
            candidate = self.dataset_positions[index]
            best_target = candidate
            if np.linalg.norm(candidate - self.tip_position) >= self.minimum_start_distance_mm:
                break
        return best_target.copy()

    def _build_info(
        self,
        *,
        distance: float,
        is_success: bool,
        realised_delta: np.ndarray | None = None,
        blocked_fraction: float = 0.0,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "distance_mm": float(distance),
            "is_success": bool(is_success),
            "step_count": self.step_count,
            "cable_position_units": self.cable_position.copy(),
            "cable_position_mm": self.cable_units_to_mm(self.cable_position),
            "active_cables": int(
                np.count_nonzero(
                    self.cable_position
                    > (self.cable_min + self.active_threshold_units)
                )
            ),
            "tip_position_mm": self.tip_position.copy(),
            "target_position_mm": self.target_position.copy(),
            "blocked_fraction": float(blocked_fraction),
        }
        if realised_delta is not None:
            info["realised_delta_units"] = realised_delta.copy()
            info["realised_delta_mm"] = self.cable_units_to_mm(realised_delta)
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        if self.initial_state_mode == "neutral":
            self.cable_position = self.cable_min.copy()
        elif self.initial_state_mode == "random_dataset":
            index = int(self.np_random.integers(len(self.dataset_cables)))
            self.cable_position = self._project_to_measured_domain(
                self.dataset_cables[index]
            )
        else:
            raise ValueError(
                "initial_state_mode must be either 'neutral' or 'random_dataset'."
            )

        self.tip_position = self.predictor.predict(self.cable_position).astype(np.float32)
        self.target_position = self._sample_target().astype(np.float32)
        self.previous_distance = float(
            np.linalg.norm(self.target_position - self.tip_position)
        )

        info = self._build_info(
            distance=self.previous_distance,
            is_success=self.previous_distance <= self.success_tolerance_mm,
        )
        return self._get_observation(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1

        clipped_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        requested_delta = clipped_action * self.max_delta_units
        requested_cable_position = self.cable_position + requested_delta
        new_cable_position = self._project_to_measured_domain(
            requested_cable_position
        )
        realised_delta = new_cable_position - self.cable_position

        self.cable_position = new_cable_position
        self.tip_position = self.predictor.predict(self.cable_position).astype(np.float32)

        distance = float(np.linalg.norm(self.target_position - self.tip_position))
        progress = (self.previous_distance - distance) / self.workspace_diagonal
        normalised_distance = distance / self.workspace_diagonal
        action_cost = float(np.mean(np.square(clipped_action)))
        blocked_fraction = float(
            np.mean(np.abs(requested_delta - realised_delta) / self.max_delta_units)
        )

        reward = (
            self.progress_weight * progress
            - self.distance_weight * normalised_distance
            - self.action_weight * action_cost
            - self.boundary_weight * blocked_fraction
        )

        terminated = distance <= self.success_tolerance_mm
        truncated = self.step_count >= self.max_episode_steps
        if terminated:
            reward += self.success_bonus

        self.previous_distance = distance
        info = self._build_info(
            distance=distance,
            is_success=terminated,
            realised_delta=realised_delta,
            blocked_fraction=blocked_fraction,
        )

        return self._get_observation(), float(reward), terminated, truncated, info
