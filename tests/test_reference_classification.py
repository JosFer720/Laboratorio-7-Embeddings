from functools import partial

import numpy as np
import pytest
from torch.utils.data import DataLoader

from embeddings_lab.classification import embedding_bag_collate
from embeddings_lab.reference_classification import (
    TfidfBaseline,
    classification_metrics,
    evaluate_once,
    pretrained_classifier,
    stratified_fraction,
)

TEXTS = ["world leaders meet", "team wins match", "market reports profit", "new computer released"]


def test_fraction_is_stratified_reproducible_and_full_when_one():
    texts = [f"doc {i}" for i in range(80)]
    labels = [i // 20 for i in range(80)]
    subset, subset_labels = stratified_fraction(texts, labels, 0.25, seed=3)
    assert len(subset) == 20 and [subset_labels.count(k) for k in range(4)] == [5, 5, 5, 5]
    assert (subset, subset_labels) == stratified_fraction(texts, labels, 0.25, seed=3)
    assert stratified_fraction(texts, labels, 1.0)[0] == texts
    for bad in (0, 1.5, -1):
        with pytest.raises(ValueError):
            stratified_fraction(texts, labels, bad)


def test_metrics_include_classes_missing_from_predictions():
    metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 2])
    assert metrics["accuracy"] == 0.75 and metrics["count"] == 4
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["macro_precision"] == pytest.approx((1 + 1 + 0.5 + 0) / 4)


def test_tfidf_baseline_separates_toy_classes_and_counts_parameters():
    train = [text + f" extra{i}" for i in range(6) for text in TEXTS]
    labels = [k for _ in range(6) for k in range(4)]
    model = TfidfBaseline(regularization=10.0, min_df=1).fit(TfidfBaseline.tokenize(train), labels)
    assert list(model.predict(TfidfBaseline.tokenize(TEXTS))) == [0, 1, 2, 3]
    counts = model.parameter_counts()
    assert counts["total"] == counts["trainable"] == 4 * len(model.vectorizer.vocabulary_) + 4
    assert model.training_seconds > 0


def test_evaluate_once_returns_metrics_and_confusion_matrix():
    vocabulary = {"<unk>": 0, "a": 1, "b": 2}
    table = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    model = pretrained_classifier(vocabulary, table, freeze=True, hidden_dim=4, dropout=0.0)
    assert not model.embedding.weight.requires_grad
    data = [([1], 0), ([2], 1), ([1, 1], 0), ([2, 2], 1)]
    loader = DataLoader(data, batch_size=2, collate_fn=partial(embedding_bag_collate, unknown_id=0))
    metrics, matrix = evaluate_once(model, loader, "cpu")
    assert metrics["count"] == 4 and metrics["loss"] > 0
    assert matrix.shape == (4, 4) and matrix.sum() == 4
    assert not model.training
