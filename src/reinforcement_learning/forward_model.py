"""Adapter for the exact forward model used by train_forward_model.py.

The training script uses:
    3 -> hidden_dim -> hidden_dim -> hidden_dim//2 -> 3
with ReLU, BatchNorm and Dropout in the first two hidden blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from torch import nn


class ForwardNet(nn.Module):
    """Exact architecture used by the supplied forward-model training script."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ForwardModelPredictor:
    """Load the forward checkpoint and its StandardScaler objects."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        checkpoint_key: Optional[str] = "model_state_dict",
        hidden_dim: int = 256,
        dropout: float = 0.05,
        input_scaler_path: str | Path | None = None,
        output_scaler_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Forward checkpoint not found: {checkpoint_path}")

        self.device = self._resolve_device(device)
        self.input_scaler = self._load_scaler(input_scaler_path, "input")
        self.output_scaler = self._load_scaler(output_scaler_path, "output")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict, metadata = self._extract_state_dict(checkpoint, checkpoint_key)

        # Prefer metadata saved by the supplied training script. For a raw
        # state_dict, infer hidden_dim from the first linear layer.
        resolved_hidden_dim = int(metadata.get("hidden_dim", self._infer_hidden_dim(state_dict, hidden_dim)))
        resolved_dropout = float(metadata.get("dropout", dropout))

        self.model = ForwardNet(
            hidden_dim=resolved_hidden_dim,
            dropout=resolved_dropout,
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.hidden_dim = resolved_hidden_dim
        self.dropout = resolved_dropout
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        requested = str(device).lower().strip()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the forward model but is unavailable.")
        return torch.device(requested)

    @staticmethod
    def _load_scaler(path: str | Path | None, name: str):
        if path is None:
            raise ValueError(
                f"The supplied forward model was trained with StandardScaler; "
                f"the {name} scaler path must be provided."
            )
        scaler_path = Path(path)
        if not scaler_path.is_file():
            raise FileNotFoundError(f"{name.capitalize()} scaler not found: {scaler_path}")
        scaler = joblib.load(scaler_path)
        if not hasattr(scaler, "transform"):
            raise TypeError(f"Loaded {name} scaler does not provide transform().")
        return scaler

    @staticmethod
    def _extract_state_dict(checkpoint, checkpoint_key: Optional[str]):
        if checkpoint_key is None:
            if not isinstance(checkpoint, dict):
                raise TypeError("A raw state_dict checkpoint must be a dictionary.")
            return checkpoint, {}

        if not isinstance(checkpoint, dict) or checkpoint_key not in checkpoint:
            available = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
            raise KeyError(
                f"Checkpoint key '{checkpoint_key}' not found. Available keys: {available}. "
                "Use checkpoint_key: null only with forward_model_best.pt."
            )
        metadata = {key: value for key, value in checkpoint.items() if key != checkpoint_key}
        return checkpoint[checkpoint_key], metadata

    @staticmethod
    def _infer_hidden_dim(state_dict: dict, fallback: int) -> int:
        first_weight = state_dict.get("network.0.weight")
        if isinstance(first_weight, torch.Tensor) and first_weight.ndim == 2:
            return int(first_weight.shape[0])
        return int(fallback)

    def predict(self, cable_values: np.ndarray) -> np.ndarray:
        """Predict XYZ for cable values in the original 0--100 dataset units."""
        values = np.asarray(cable_values, dtype=np.float32)
        single_sample = values.ndim == 1
        values = np.atleast_2d(values)

        if values.shape[1] != 3:
            raise ValueError(f"Expected cable input shape (N, 3), got {values.shape}.")

        scaled_input = self.input_scaler.transform(values).astype(np.float32)
        input_tensor = torch.from_numpy(scaled_input).to(self.device)

        with torch.no_grad():
            scaled_output = self.model(input_tensor).cpu().numpy()

        output = self.output_scaler.inverse_transform(scaled_output).astype(np.float32)
        return output[0] if single_sample else output
