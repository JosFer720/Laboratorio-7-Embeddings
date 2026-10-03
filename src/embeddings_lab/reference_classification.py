"""TF-IDF baseline, data fractions and one-shot test evaluation for AG News."""

from collections.abc import Iterable, Sequence
from time import perf_counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import Tensor, nn

from .classification import EmbeddingBagClassifier
from .preprocessing import word_tokenize


def stratified_fraction(
    texts: Sequence[str], labels: Sequence[int], fraction: float, *, seed: int = 42
) -> tuple[list[str], list[int]]:
    """Take a stratified fraction of the training data, or all of it when the fraction is one."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1.0:
        return list(texts), list(labels)
    subset_texts, _, subset_labels, _ = train_test_split(
        list(texts), list(labels), train_size=fraction, stratify=list(labels), random_state=seed,
    )
    return subset_texts, subset_labels


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], *, num_classes: int = 4
) -> dict[str, float | int]:
    """Accuracy and macro precision, recall and F1 for integer labels."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=list(range(num_classes)), average="macro", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "count": len(targets),
    }


def _word_and_bigram_features(tokens: Sequence[str]) -> list[str]:
    return list(tokens) + [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]


class TfidfBaseline:
    """TF-IDF over words and bigrams followed by multinomial logistic regression."""

    def __init__(self, *, regularization: float = 10.0, min_df: int = 2, seed: int = 42) -> None:
        self.regularization = regularization
        self.vectorizer = TfidfVectorizer(
            analyzer=_word_and_bigram_features, min_df=min_df, sublinear_tf=True,
        )
        self.classifier = LogisticRegression(C=regularization, max_iter=300, random_state=seed)
        self.training_seconds = 0.0

    @staticmethod
    def tokenize(texts: Iterable[str]) -> list[list[str]]:
        """Use the tokenizer of the embedding pipeline so every model sees the same tokens."""
        return [word_tokenize(text) for text in texts]

    def fit(self, tokenized_texts: Sequence[Sequence[str]], labels: Sequence[int]) -> "TfidfBaseline":
        started = perf_counter()
        features = self.vectorizer.fit_transform(tokenized_texts)
        self.classifier.fit(features, labels)
        self.training_seconds = perf_counter() - started
        return self

    def predict(self, tokenized_texts: Sequence[Sequence[str]]) -> np.ndarray:
        return self.classifier.predict(self.vectorizer.transform(tokenized_texts))

    def predict_proba(self, tokenized_texts: Sequence[Sequence[str]]) -> np.ndarray:
        return self.classifier.predict_proba(self.vectorizer.transform(tokenized_texts))

    def parameter_counts(self) -> dict[str, int]:
        """Weights and biases of the linear classifier; the vectorizer has no learned weights."""
        total = int(self.classifier.coef_.size + self.classifier.intercept_.size)
        return {"total": total, "trainable": total}


def evaluate_once(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor, Tensor]],
    device: str | torch.device,
    *,
    num_classes: int = 4,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Run a classifier over a loader and return its metrics and confusion matrix."""
    target_device = torch.device(device)
    model.to(target_device)
    model.train(False)
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for token_ids, offsets, labels in loader:
            logits = model(token_ids.to(target_device), offsets.to(target_device))
            total_loss += float(nn.functional.cross_entropy(logits, labels.to(target_device), reduction="sum"))
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = {"loss": total_loss / len(targets), **classification_metrics(targets, predictions, num_classes=num_classes)}
    return metrics, confusion_matrix(targets, predictions, labels=list(range(num_classes)))


def pretrained_classifier(
    vocabulary: dict[str, int],
    table: np.ndarray,
    *,
    freeze: bool,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> EmbeddingBagClassifier:
    """Build the mean embedding classifier initialised from a lookup table."""
    return EmbeddingBagClassifier(
        len(vocabulary), table.shape[1], pretrained_embeddings=table, freeze=freeze,
        hidden_dim=hidden_dim, dropout=dropout,
    )
