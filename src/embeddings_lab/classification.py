"""Stratified text preparation and trainable EmbeddingBag classifiers."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from numbers import Integral
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import Tensor, nn

from .preprocessing import word_tokenize


def stratified_split(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Reserve a reproducible stratified validation subset of training data."""
    if len(texts) != len(labels) or not len(texts):
        raise ValueError("texts and labels must have the same nonzero length")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    return tuple(train_test_split(
        list(texts), list(labels), test_size=validation_fraction,
        random_state=seed, stratify=labels,
    ))


def _valid_id(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def encode_document(
    text: str, vocabulary: Mapping[str, int], *, unknown_token: str = "<unk>"
) -> list[int]:
    """Normalize a document and use the unknown token for an empty result."""
    if unknown_token not in vocabulary or not _valid_id(vocabulary[unknown_token]):
        raise ValueError("vocabulary must contain a valid unknown token ID")
    unknown_id = int(vocabulary[unknown_token])
    ids = [vocabulary.get(token, unknown_id) for token in word_tokenize(text)]
    if not ids:
        ids = [unknown_id]
    if any(not _valid_id(index) for index in ids):
        raise ValueError("vocabulary IDs must be nonnegative integers")
    return ids


def embedding_bag_collate(
    batch: Sequence[tuple[Sequence[int] | Tensor, int]], *, unknown_id: int = 0
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten token lists and create one offset and label per document."""
    if not batch or not _valid_id(unknown_id):
        raise ValueError("batch must be nonempty and unknown_id must be valid")
    flat_ids: list[int] = []
    offsets: list[int] = []
    labels: list[int] = []
    for document, label in batch:
        ids = document.tolist() if isinstance(document, Tensor) else list(document)
        if any(not _valid_id(index) for index in ids) or not _valid_id(label):
            raise ValueError("token IDs and labels must be nonnegative integers")
        offsets.append(len(flat_ids))
        flat_ids.extend(ids or [unknown_id])
        labels.append(int(label))
    return (
        torch.tensor(flat_ids, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class EmbeddingBagClassifier(nn.Module):
    """Mean word embeddings followed by a linear head or a compact MLP."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int = 4,
        *,
        pretrained_embeddings: Tensor | np.ndarray | None = None,
        freeze: bool = False,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dimensions = [vocab_size, embedding_dim, num_classes]
        if hidden_dim is not None:
            dimensions.append(hidden_dim)
        if any(not _valid_id(size) or size == 0 for size in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be at least zero and below one")
        self.num_classes = num_classes
        if pretrained_embeddings is not None:
            weights = torch.as_tensor(pretrained_embeddings, dtype=torch.float32).detach().clone()
            if weights.shape != (vocab_size, embedding_dim) or not torch.isfinite(weights).all():
                raise ValueError("pretrained embeddings must have the specified finite shape")
            self.embedding = nn.EmbeddingBag.from_pretrained(weights, freeze=freeze, mode="mean")
        else:
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
            self.embedding.weight.requires_grad_(not freeze)
        self.classifier = (
            nn.Linear(embedding_dim, num_classes)
            if hidden_dim is None
            else nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        )

    def forward(self, token_ids: Tensor, offsets: Tensor) -> Tensor:
        """Return class logits for flattened documents and their offsets."""
        return self.classifier(self.embedding(token_ids, offsets))


def _run_epoch(
    model: EmbeddingBagClassifier,
    loader: Iterable[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | int]:
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    criterion = nn.CrossEntropyLoss()
    for token_ids, offsets, labels in loader:
        token_ids, offsets, labels = (tensor.to(device) for tensor in (token_ids, offsets, labels))
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(token_ids, offsets)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("classifier loss must be finite")
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    count = len(targets)
    if count == 0:
        return {
            "loss": 0.0, "accuracy": 0.0, "macro_precision": 0.0,
            "macro_recall": 0.0, "macro_f1": 0.0, "count": 0,
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=list(range(model.num_classes)),
        average="macro", zero_division=0,
    )
    return {
        "loss": total_loss / count,
        "accuracy": sum(actual == predicted for actual, predicted in zip(targets, predictions)) / count,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "count": count,
    }


def evaluate_classifier(
    model: EmbeddingBagClassifier,
    loader: Iterable[tuple[Tensor, Tensor, Tensor]],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Compute loss, accuracy and macro precision, recall and F1 without gradients."""
    target_device = torch.device(device)
    model.to(target_device)
    previous_mode = model.training
    model.train(False)
    try:
        with torch.no_grad():
            return _run_epoch(model, loader, target_device)
    finally:
        model.train(previous_mode)


def train_classifier(
    model: EmbeddingBagClassifier,
    train_loader: Iterable[tuple[Tensor, Tensor, Tensor]],
    validation_loader: Iterable[tuple[Tensor, Tensor, Tensor]] | None = None,
    *,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
) -> list[dict[str, float | int]]:
    """Train with Adam and report epoch losses and optional validation metrics."""
    if not _valid_id(epochs) or epochs == 0:
        raise ValueError("epochs must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if epochs > 1:
        for name, loader in (("training", train_loader), ("validation", validation_loader)):
            if isinstance(loader, Iterator):
                raise ValueError(f"{name} loader must be reiterable for multiple epochs")
    target_device = torch.device(device)
    model.to(target_device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        model.train()
        metrics = _run_epoch(model, train_loader, target_device, optimizer)
        if metrics["count"] == 0:
            raise ValueError("training loader must contain at least one document")
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": metrics["loss"],
            "train_accuracy": metrics["accuracy"],
            "train_macro_precision": metrics["macro_precision"],
            "train_macro_recall": metrics["macro_recall"],
            "train_macro_f1": metrics["macro_f1"],
        }
        if validation_loader is not None:
            validation = evaluate_classifier(model, validation_loader, device=target_device)
            if validation["count"] == 0:
                raise ValueError("validation loader must contain at least one document")
            row.update({f"validation_{name}": value for name, value in validation.items()})
        row["duration_seconds"] = perf_counter() - started
        history.append(row)
    return history
