"""Configuration and persistence helpers for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Store shared hyperparameters for embedding experiments."""

    seed: int = 42
    corpus_token_limit: int = 20_000_000
    dimensions: tuple[int, ...] = (50, 100, 300)
    window_size: int = 5
    num_negatives: int = 5
    min_count: int = 5
    subsampling_threshold: float = 1e-5
    batch_size: int = 2_048
    epochs: int = 5
    learning_rate: float = 0.0025
    output_dir: Path = Path("artifacts")

    def __post_init__(self) -> None:
        positive_fields = {
            "corpus_token_limit": self.corpus_token_limit,
            "window_size": self.window_size,
            "num_negatives": self.num_negatives,
            "min_count": self.min_count,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not self.dimensions or any(value <= 0 for value in self.dimensions):
            raise ValueError("dimensions must contain positive integers")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if not 0 < self.subsampling_threshold < 1:
            raise ValueError("subsampling_threshold must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON compatible configuration."""

        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["dimensions"] = list(self.dimensions)
        return values


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_run_metadata(
    output_dir: str | Path,
    run_name: str,
    metadata: dict[str, Any],
) -> Path:
    """Persist run metadata in a dedicated directory."""

    if not run_name.strip():
        raise ValueError("run_name must not be empty")
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metadata.json"
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def count_skipgram_pairs(token_ids: Any, window_size: int) -> int:
    """Count directed context pairs without materializing them."""

    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    try:
        length = len(token_ids)
    except TypeError as error:
        raise TypeError("token_ids must provide a finite length") from error
    if length < 2:
        return 0
    distance_count = min(window_size, length - 1)
    return 2 * (
        distance_count * length - distance_count * (distance_count + 1) // 2
    )


def oov_rate(token_sequences: Any, vocabulary: dict[str, int]) -> float:
    """Return the fraction of token occurrences absent from a vocabulary."""

    total = 0
    missing = 0
    for sequence in token_sequences:
        for token in sequence:
            if not isinstance(token, str):
                raise TypeError("tokens must be strings")
            total += 1
            missing += token not in vocabulary
    return missing / total if total else 0.0


def count_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable model parameters."""

    parameters = list(model.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    }
