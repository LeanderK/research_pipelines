"""
Simple example: synthetic binary classification with Torch.

Pipeline shape:
train_split, val_split, test_split = create_classification_splits()
model = get_model()
train(model, train_split)
evaluate(model, test_split)

All runtime artifacts are persisted under data/torch_classification/.

This example demonstrates argument decoration using typing.Annotated:
arguments marked with Annotated[T, Ignore()] are excluded from traced configs
but can still be used in the function implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - runtime guidance only
    raise SystemExit(
        "This example requires the optional torch extra. Install it with: pip install '.[torch]'"
    ) from exc

from research_pipelines.backends.manager import get_backend, set_backend
from research_pipelines.backends.pickle_backend import PickleBackend
from research_pipelines.core import clear_traced_registry, Ignore
from research_pipelines.decorators import dataset, evaluation, model
from research_pipelines.dag import build_dag


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "torch_classification"
SPLIT_ROOT = DATA_ROOT / "splits"
CHECKPOINT_ROOT = DATA_ROOT / "checkpoints"
RESULT_ROOT = DATA_ROOT / "results"
TRACE_ROOT = DATA_ROOT / "traced_configs"


def ensure_directories() -> None:
    """Create the data directories used by the example."""
    for path in (SPLIT_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, TRACE_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    """Persist a small JSON summary artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_tensor_split(path: Path, features: torch.Tensor, labels: torch.Tensor, split: str) -> None:
    """Persist a tensor split to disk."""
    torch.save({"split": split, "features": features, "labels": labels}, path)


def load_tensor_split(path: Path) -> dict:
    """Load a tensor split from disk."""
    return torch.load(path, map_location="cpu")


def make_blob_data(samples_per_class: int, noise: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a tiny 2D binary classification dataset."""
    generator = torch.Generator().manual_seed(seed)
    class_zero = torch.randn(samples_per_class, 2, generator=generator) * noise + torch.tensor([-1.4, -1.4])
    class_one = torch.randn(samples_per_class, 2, generator=generator) * noise + torch.tensor([1.4, 1.4])
    features = torch.cat([class_zero, class_one], dim=0)
    labels = torch.cat(
        [torch.zeros(samples_per_class, dtype=torch.long), torch.ones(samples_per_class, dtype=torch.long)],
        dim=0,
    )
    permutation = torch.randperm(features.shape[0], generator=generator)
    return features[permutation], labels[permutation]


def split_data(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_fraction: float,
    val_fraction: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Split tensors into train/val/test partitions."""
    total = features.shape[0]
    train_end = int(total * train_fraction)
    val_end = int(total * (train_fraction + val_fraction))
    return (
        (features[:train_end], labels[:train_end]),
        (features[train_end:val_end], labels[train_end:val_end]),
        (features[val_end:], labels[val_end:]),
    )


@dataset()
def create_classification_splits(artifact_root: Annotated[str, Ignore()]):
    """Generate, persist, and return train/val/test splits as traced objects.
    
    The artifact_root parameter is marked with Annotated[str, Ignore()] to exclude
    it from the traced configuration, even though we use it for persisting splits.
    """
    ensure_directories()

    features, labels = make_blob_data(samples_per_class=80, noise=0.35, seed=7)
    train_split, val_split, test_split = split_data(features, labels, train_fraction=0.7, val_fraction=0.15)

    train_features, train_labels = train_split
    val_features, val_labels = val_split
    test_features, test_labels = test_split

    save_tensor_split(SPLIT_ROOT / "train.pt", train_features, train_labels, "train")
    save_tensor_split(SPLIT_ROOT / "val.pt", val_features, val_labels, "val")
    save_tensor_split(SPLIT_ROOT / "test.pt", test_features, test_labels, "test")

    return (
        {
            "split": "train",
            "path": str(SPLIT_ROOT / "train.pt"),
            "features": train_features,
            "labels": train_labels,
        },
        {
            "split": "val",
            "path": str(SPLIT_ROOT / "val.pt"),
            "features": val_features,
            "labels": val_labels,
        },
        {
            "split": "test",
            "path": str(SPLIT_ROOT / "test.pt"),
            "features": test_features,
            "labels": test_labels,
        },
    )


class SimpleClassifier(nn.Module):
    """Tiny MLP for binary classification."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@model()
def get_model(hidden_dim: int = 8) -> SimpleClassifier:
    """Create the initial untrained model."""
    ensure_directories()
    return SimpleClassifier(hidden_dim=hidden_dim)


def load_classifier_from_checkpoint(checkpoint_path: str) -> SimpleClassifier:
    """Rebuild a classifier from its persisted checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SimpleClassifier(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def accuracy_for_model(model: SimpleClassifier, features: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy for a classifier on a full tensor batch."""
    with torch.no_grad():
        logits = model(features)
        predictions = logits.argmax(dim=1)
        return (predictions == labels).float().mean().item()


@model()
def train(
    model: SimpleClassifier,
    train_split,
    artifact_root: Annotated[str, Ignore()],
    learning_rate: float = 0.05,
    epochs: int = 20,
) -> SimpleClassifier:
    """Train the classifier and persist a checkpoint.
    
    The artifact_root parameter is marked with Annotated[str, Ignore()] to exclude
    it from the traced configuration while still using it for checkpoint persistence.
    """
    checkpoint_path = Path(artifact_root) / "checkpoints" / "simple_classifier.pt"
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_split["features"])
        loss = loss_fn(logits, train_split["labels"])
        loss.backward()
        optimizer.step()

    model.eval()
    train_accuracy = accuracy_for_model(model, train_split["features"], train_split["labels"])

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": train_split["features"].shape[1],
            "hidden_dim": model.net[0].out_features,
        },
        checkpoint_path,
    )

    save_json(
        RESULT_ROOT / "training_summary.json",
        {
            "train_accuracy": round(train_accuracy, 4),
            "checkpoint_path": str(checkpoint_path),
        },
    )

    return model


@evaluation()
def evaluate(model: SimpleClassifier, test_split, artifact_root: Annotated[str, Ignore()]) -> dict:
    """Evaluate the trained classifier on the held-out test split.
    
    The artifact_root parameter is marked with Annotated[str, Ignore()] to exclude
    it from the traced configuration while still using it for results persistence.
    """
    test_accuracy = accuracy_for_model(model, test_split["features"], test_split["labels"])
    results = {
        "test_accuracy": round(test_accuracy, 4),
        "num_examples": int(test_split["labels"].shape[0]),
    }
    save_json(Path(artifact_root) / "results" / "evaluation.json", results)
    return results


def predict_sample(model: SimpleClassifier, feature_vector: torch.Tensor) -> dict:
    """Run a single prediction from the in-memory model."""
    with torch.no_grad():
        logits = model(feature_vector.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(probabilities.argmax().item())
    return {
        "predicted_class": predicted_class,
        "confidence": round(float(probabilities[predicted_class].item()), 4),
    }


def main():
    """Run the pipeline and print the DAG."""
    ensure_directories()
    set_backend(PickleBackend(directory=str(TRACE_ROOT)))
    clear_traced_registry()
    get_backend().clear()

    print("=" * 60)
    print("Research Pipelines - Torch Classification Example")
    print("=" * 60)
    print()

    print("Step 1: Creating synthetic splits...")
    train_split, val_split, test_split = create_classification_splits(artifact_root=str(DATA_ROOT))
    print(f"Train split size: {train_split['features'].shape[0]}")
    print(f"Validation split size: {val_split['features'].shape[0]}")
    print(f"Test split size: {test_split['features'].shape[0]}")
    print()

    print("Step 2: Building model...")
    model = get_model(hidden_dim=8)
    print()

    print("Step 3: Training model...")
    model = train(model, train_split, artifact_root=str(DATA_ROOT), learning_rate=0.05, epochs=20)
    print()

    print("Step 4: Evaluating model...")
    results = evaluate(model, test_split, artifact_root=str(DATA_ROOT))
    print(f"Test results: {results}")
    print()

    print("Step 5: Demo prediction...")
    prediction = predict_sample(model, torch.tensor([1.2, 1.0]))
    print(f"Prediction: {prediction}")
    print()

    save_json(
        RESULT_ROOT / "run_summary.json",
        {
            "evaluation": results,
            "prediction": prediction,
            "model_checkpoint": str(Path(DATA_ROOT) / "checkpoints" / "simple_classifier.pt"),
        },
    )

    print("=" * 60)
    print("DAG Structure:")
    print("=" * 60)
    dag = build_dag()
    for obj_id, obj_info in dag.items():
        print(f"\n{obj_info['type'].upper()}: {obj_id[:8]}...")
        print(f"  Config: {obj_info['config']}")
        if obj_info["dependencies"]:
            dep_strs = [f"{d[:8]}..." for d in obj_info["dependencies"]]
            print(f"  Depends on: {', '.join(dep_strs)}")

    print()
    print("=" * 60)
    print("Backend Persistence:")
    print("=" * 60)
    backend = get_backend()
    all_configs = backend.load_all()
    print(f"Objects persisted: {len(all_configs)}")
    for obj_id in all_configs:
        print(f"  - {obj_id[:12]}...")


if __name__ == "__main__":
    main()
