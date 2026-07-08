"""Train a prototype binary tennis shot classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from .models import TennisShotClassifier
except ImportError:  # Allows direct execution: python src/train_prototype.py ...
    from models import TennisShotClassifier  # type: ignore


LABEL_COLUMNS = ("label", "target", "shot_label")


class TennisSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for fused sequence tensors exported as ``.npz`` or ``.parquet``.

    Each item returns:
        features: ``(sequence_length, feature_dimension)`` float tensor.
        label: ``(1,)`` float tensor for binary cross-entropy.
    """

    def __init__(
        self,
        data_path: str | Path,
        labels_path: str | Path | None = None,
        label_column: str | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path) if labels_path is not None else None
        self.label_column = label_column

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        self.features, self.labels = self._load_dataset()
        if self.features.ndim != 3:
            raise ValueError(
                "features must have shape (num_sequences, sequence_length, feature_dimension)."
            )
        if self.labels.ndim != 1:
            raise ValueError("labels must be a 1D array.")
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must contain the same number of samples.")

        self.features = np.nan_to_num(self.features.astype(np.float32), nan=0.0)
        self.labels = self.labels.astype(np.float32)
        unique_labels = set(np.unique(self.labels).tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError("labels must be binary values: 0 or 1.")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.features[index])
        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return features, label

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[2])

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        if self.data_path.suffix == ".npz":
            return self._load_npz()

        if self.data_path.suffix == ".parquet":
            return self._load_parquet()

        raise ValueError("data_path must end with .npz or .parquet.")

    def _load_npz(self) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.data_path, allow_pickle=True) as data:
            if "sequences" not in data:
                raise KeyError("NPZ dataset must contain a 'sequences' array.")

            features = np.asarray(data["sequences"], dtype=np.float32)
            if "labels" in data:
                labels = np.asarray(data["labels"], dtype=np.float32).reshape(-1)
            elif self.labels_path is not None:
                labels = self._load_external_labels()
            else:
                raise ValueError(
                    "Labels are required. Add a 'labels' array to the NPZ or pass --labels-path."
                )

        return features, labels

    def _load_parquet(self) -> tuple[np.ndarray, np.ndarray]:
        dataframe = pd.read_parquet(self.data_path)
        sequence_column = "sequence_idx"
        if sequence_column not in dataframe.columns:
            raise KeyError("Parquet dataset must contain a 'sequence_idx' column.")

        inferred_label_column = self._infer_label_column(dataframe.columns)
        label_column = self.label_column or inferred_label_column
        if label_column is None and self.labels_path is None:
            raise ValueError(
                "Labels are required. Include label/target/shot_label or pass --labels-path."
            )

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in {sequence_column, label_column, *LABEL_COLUMNS}
        ]
        if not feature_columns:
            raise ValueError("No feature columns found in parquet dataset.")

        sequences = []
        labels = []
        for _, group in dataframe.sort_values([sequence_column, "frame_idx"]).groupby(
            sequence_column
        ):
            sequences.append(group[feature_columns].to_numpy(dtype=np.float32))
            if label_column is not None:
                labels.append(float(group[label_column].iloc[0]))

        features = np.stack(sequences, axis=0)
        if self.labels_path is not None and label_column is None:
            labels_array = self._load_external_labels()
        else:
            labels_array = np.asarray(labels, dtype=np.float32)

        return features, labels_array

    def _load_external_labels(self) -> np.ndarray:
        if self.labels_path is None:
            raise ValueError("labels_path is not configured.")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        if self.labels_path.suffix == ".npy":
            return np.asarray(np.load(self.labels_path), dtype=np.float32).reshape(-1)

        if self.labels_path.suffix == ".csv":
            label_frame = pd.read_csv(self.labels_path)
            label_column = self.label_column or self._infer_label_column(label_frame.columns)
            if label_column is None:
                raise ValueError("CSV labels file must contain label, target, or shot_label.")
            return label_frame[label_column].to_numpy(dtype=np.float32).reshape(-1)

        raise ValueError("labels_path must end with .npy or .csv.")

    @staticmethod
    def _infer_label_column(columns: Iterable[str]) -> str | None:
        for column in LABEL_COLUMNS:
            if column in columns:
                return column
        return None


def select_device() -> torch.device:
    """Select the best local PyTorch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binary_accuracy(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary accuracy from probability outputs."""

    predictions = (probabilities >= 0.5).float()
    return float((predictions == labels).float().mean().item())


def train_epoch(
    model: TennisShotClassifier,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one optimization epoch and return average loss and accuracy."""

    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    for features, labels in data_loader:
        # Batch features shape: (B, T, F), labels shape: (B, 1).
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        probabilities = model(features)
        loss = criterion(probabilities, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_accuracy += binary_accuracy(probabilities.detach(), labels)
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Training loader produced zero batches.")

    return total_loss / total_batches, total_accuracy / total_batches


def evaluate_epoch(
    model: TennisShotClassifier,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one epoch without gradient tracking."""

    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    with torch.no_grad():
        for features, labels in data_loader:
            # Batch features shape: (B, T, F), labels shape: (B, 1).
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            probabilities = model(features)
            loss = criterion(probabilities, labels)

            total_loss += float(loss.item())
            total_accuracy += binary_accuracy(probabilities, labels)
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Validation loader produced zero batches.")

    return total_loss / total_batches, total_accuracy / total_batches


def build_loaders(
    dataset: TennisSequenceDataset,
    batch_size: int,
    validation_split: float,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Split a dataset into train/validation loaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")
    if len(dataset) < 2:
        raise ValueError("At least two labeled sequences are required for train/val split.")

    validation_size = max(1, int(round(len(dataset) * validation_split)))
    training_size = len(dataset) - validation_size
    if training_size <= 0:
        raise ValueError("Validation split leaves no training samples.")

    train_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def train(args: argparse.Namespace) -> Path:
    """Train the prototype classifier and save best weights."""

    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    dataset = TennisSequenceDataset(
        data_path=args.data_path,
        labels_path=args.labels_path,
        label_column=args.label_column,
    )
    train_loader, validation_loader = build_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )

    device = select_device()
    print(f"Using device: {device}")
    print(
        "Dataset: "
        f"{len(dataset)} sequences, feature_dimension={dataset.feature_dimension}"
    )

    model = TennisShotClassifier(
        feature_dimension=dataset.feature_dimension,
        hidden_dimension=args.hidden_dimension,
        backbone=args.backbone,
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "prototype_classifier.pt"
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dimension": dataset.feature_dimension,
                    "hidden_dimension": args.hidden_dimension,
                    "backbone": args.backbone,
                    "validation_loss": best_validation_loss,
                },
                best_path,
            )
            print(f"  saved best weights -> {best_path}")

    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prototype tennis shot classifier.")
    parser.add_argument("--data-path", required=True, help="Path to .npz or .parquet dataset.")
    parser.add_argument("--labels-path", default=None, help="Optional .npy or .csv labels file.")
    parser.add_argument("--label-column", default=None, help="Optional label column name.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--backbone", choices=("lstm", "transformer"), default="lstm")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


def main() -> None:
    try:
        best_path = train(parse_args())
        print(f"Training complete. Best checkpoint: {best_path}")
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise


if __name__ == "__main__":
    main()
