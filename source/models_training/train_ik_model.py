"""Train an inverse-kinematics neural model from action-position data.

Expected dataset format, without header:
    a1,a2,a3,x,y,z

The model learns:
    input  = [x, y, z]
    output = [a1, a2, a3]

Example:
    python train_ik_model.py --data continuum_robot_dataset.txt --epochs 500 --hidden-dim 256
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainConfig:
    data: Path
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    dropout: float
    test_size: float
    seed: int


class IKNet(nn.Module):
    """Simple MLP inverse-kinematics model: XYZ position -> actuator actions."""

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


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train IK model: XYZ -> cable actions.")
    parser.add_argument("--data", type=Path, default=Path("continuum_robot_dataset.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("ik_model_output"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        data=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        test_size=args.test_size,
        seed=args.seed,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    columns = ["a1", "a2", "a3", "x", "y", "z"]
    data = pd.read_csv(path, header=None, names=columns)

    if data.isna().any().any():
        raise ValueError("Dataset contains missing values. Please clean the data first.")

    x_position = data[["x", "y", "z"]].to_numpy(dtype=np.float32)
    y_action = data[["a1", "a2", "a3"]].to_numpy(dtype=np.float32)
    return x_position, y_action


def create_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(
    model: IKNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


def evaluate_loss(
    model: IKNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


def predict(model: IKNet, x_scaled: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
        predictions = model(x_tensor).cpu().numpy()
    return predictions


def save_loss_plot(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    x_position, y_action = load_dataset(config.data)

    x_train, x_val, y_train, y_val = train_test_split(
        x_position,
        y_action,
        test_size=config.test_size,
        random_state=config.seed,
        shuffle=True,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_scaled = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val_scaled = x_scaler.transform(x_val).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val).astype(np.float32)

    train_loader, val_loader = create_dataloaders(
        x_train_scaled,
        y_train_scaled,
        x_val_scaled,
        y_val_scaled,
        config.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKNet(hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
    )

    best_val_loss = float("inf")
    best_model_path = config.output_dir / "ik_model_best.pt"
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:04d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    y_val_pred_scaled = predict(model, x_val_scaled, device)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)

    metrics = {
        "mae_per_action": mean_absolute_error(y_val, y_val_pred, multioutput="raw_values").tolist(),
        "mae_mean": float(mean_absolute_error(y_val, y_val_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
        "r2": float(r2_score(y_val, y_val_pred)),
        "best_val_loss_scaled_mse": float(best_val_loss),
        "num_samples": int(len(x_position)),
        "device": str(device),
        "config": {**asdict(config), "data": str(config.data), "output_dir": str(config.output_dir)},
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "metrics": metrics,
        },
        config.output_dir / "ik_model_checkpoint.pt",
    )
    joblib.dump(x_scaler, config.output_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, config.output_dir / "y_scaler.joblib")

    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_loss_plot(train_losses, val_losses, config.output_dir / "loss_curve.png")

    print("\nValidation metrics in original action units:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved outputs to: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
