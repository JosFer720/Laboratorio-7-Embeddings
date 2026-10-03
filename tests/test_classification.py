import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from embeddings_lab.classification import (
    EmbeddingBagClassifier,
    embedding_bag_collate,
    encode_document,
    evaluate_classifier,
    stratified_split,
    train_classifier,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_split_is_stratified_reproducible_and_disjoint():
    texts = [f"document {index}" for index in range(40)]
    labels = [index // 10 for index in range(40)]
    split = stratified_split(texts, labels, seed=7)
    train_texts, validation_texts, train_labels, validation_labels = split
    assert len(train_texts) == 36
    assert len(validation_texts) == 4
    assert [train_labels.count(label) for label in range(4)] == [9, 9, 9, 9]
    assert sorted(validation_labels) == [0, 1, 2, 3]
    assert not set(train_texts) & set(validation_texts)
    assert split == stratified_split(texts, labels, seed=7)
    assert dict(zip(train_texts + validation_texts, train_labels + validation_labels)) == dict(zip(texts, labels))


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        stratified_split(["a", "b"], [0, 0], validation_fraction=fraction)


def test_document_encoding_reuses_normalization_and_unknown_ids():
    vocabulary = {"<unk>": 0, "hello": 1, "world": 2}
    assert encode_document("  HELLO, World! new ", vocabulary) == [1, 2, 0]
    assert encode_document("", vocabulary) == [0]
    assert encode_document(" !!! ", vocabulary) == [0]
    with pytest.raises(ValueError):
        encode_document("hello", {"hello": 1})


def test_document_encoding_checks_only_ids_used_by_the_document():
    vocabulary = {"<unk>": 0, "used": 1, "unused": "invalid"}

    assert encode_document("used", vocabulary) == [1]
    with pytest.raises(ValueError, match="vocabulary IDs"):
        encode_document("unused", vocabulary)


def test_collate_flattens_documents_and_preserves_empty_documents():
    ids, offsets, labels = embedding_bag_collate([([1, 2], 0), ([], 1), ([3], 2)])
    assert ids.tolist() == [1, 2, 0, 3]
    assert offsets.tolist() == [0, 2, 3]
    assert labels.tolist() == [0, 1, 2]
    assert ids.dtype == offsets.dtype == labels.dtype == torch.long


@pytest.mark.parametrize("batch", [[], [([-1], 0)], [([1.5], 0)], [([1], -1)]])
def test_collate_rejects_invalid_ids_and_labels(batch):
    with pytest.raises(ValueError):
        embedding_bag_collate(batch)


@pytest.mark.parametrize("freeze", [True, False])
def test_training_respects_embedding_freeze_and_updates_the_head(freeze):
    torch.manual_seed(4)
    pretrained = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    model = EmbeddingBagClassifier(3, 2, 2, pretrained_embeddings=pretrained, freeze=freeze)
    before_embedding = model.embedding.weight.detach().clone()
    before_head = model.classifier.weight.detach().clone()
    loader = DataLoader([([1], 0), ([2], 1)], batch_size=2, collate_fn=embedding_bag_collate)
    train_classifier(model, loader, epochs=1, learning_rate=0.1)
    assert torch.equal(model.embedding.weight, before_embedding) == freeze
    assert not torch.equal(model.classifier.weight, before_head)
    assert torch.equal(pretrained, torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))


def test_metrics_match_hand_checked_predictions_and_restore_model_mode():
    model = EmbeddingBagClassifier(3, 2, 2, pretrained_embeddings=torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))
    with torch.no_grad():
        model.classifier.weight.copy_(torch.eye(2))
        model.classifier.bias.zero_()
    loader = DataLoader([([1], 0), ([2], 1), ([1], 1)], batch_size=2, collate_fn=embedding_bag_collate)
    model.train()
    result = evaluate_classifier(model, loader)
    assert result["count"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_precision"] == pytest.approx(0.75)
    assert result["macro_recall"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["loss"] == pytest.approx(0.646595, abs=1e-5)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_training_reduces_loss_and_reports_validation_metrics():
    torch.manual_seed(12)
    model = EmbeddingBagClassifier(3, 4, 2, hidden_dim=8)
    loader = DataLoader([([1, 1], 0), ([2, 2], 1)] * 4, batch_size=8, collate_fn=embedding_bag_collate)
    history = train_classifier(model, loader, loader, epochs=15, learning_rate=0.05)
    assert len(history) == 15
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["validation_accuracy"] == 1.0
    assert history[-1]["validation_macro_f1"] == 1.0
    assert history[-1]["validation_macro_precision"] == 1.0
    assert history[-1]["validation_macro_recall"] == 1.0
    assert history[-1]["train_macro_precision"] == 1.0
    assert history[-1]["train_macro_recall"] == 1.0
    assert history[-1]["epoch"] == 15
    assert all(np.isfinite(row["train_loss"]) for row in history)


def test_empty_evaluation_returns_finite_metrics():
    model = EmbeddingBagClassifier(2, 3, 2)
    assert evaluate_classifier(model, []) == {
        "loss": 0.0, "accuracy": 0.0, "macro_precision": 0.0,
        "macro_recall": 0.0, "macro_f1": 0.0, "count": 0,
    }


def test_macro_metrics_include_classes_absent_from_targets_and_predictions():
    model = EmbeddingBagClassifier(
        3, 2, 3, pretrained_embeddings=torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    )
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., -1.]]))
        model.classifier.bias.zero_()
    batch = embedding_bag_collate([([1], 0), ([2], 1), ([1], 1)])
    result = evaluate_classifier(model, [batch])
    assert result["macro_precision"] == pytest.approx(0.5)
    assert result["macro_recall"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(4 / 9)


@pytest.mark.parametrize("single_pass_loader", ["training", "validation"])
@pytest.mark.parametrize("iterator_type", [iter, lambda batches: (batch for batch in batches)])
def test_multi_epoch_training_rejects_single_pass_loaders_before_updating_weights(single_pass_loader, iterator_type):
    model = EmbeddingBagClassifier(3, 2, 2)
    batch = embedding_bag_collate([([1], 0), ([2], 1)])
    train_loader = iterator_type([batch]) if single_pass_loader == "training" else [batch]
    validation_loader = iterator_type([batch]) if single_pass_loader == "validation" else [batch]
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(ValueError, match=single_pass_loader):
        train_classifier(model, train_loader, validation_loader, epochs=2)
    for actual, expected in zip(model.parameters(), before):
        assert torch.equal(actual, expected)


def test_single_epoch_training_accepts_single_pass_train_and_validation_loaders():
    model = EmbeddingBagClassifier(3, 2, 2)
    batch = embedding_bag_collate([([1], 0), ([2], 1)])
    history = train_classifier(model, iter([batch]), iter([batch]), epochs=1)
    assert len(history) == 1
    assert history[0]["validation_count"] == 2


@pytest.mark.parametrize("validation_loader", [[], iter([])])
def test_training_rejects_empty_validation(validation_loader):
    model = EmbeddingBagClassifier(3, 2, 2)
    batch = embedding_bag_collate([([1], 0), ([2], 1)])
    with pytest.raises(ValueError, match="validation"):
        train_classifier(model, [batch], validation_loader, epochs=1)


def test_model_and_training_reject_invalid_configuration():
    with pytest.raises(ValueError):
        EmbeddingBagClassifier(3, 2, 2, pretrained_embeddings=np.ones((4, 2)))
    with pytest.raises(ValueError):
        EmbeddingBagClassifier(0, 2, 2)
    model = EmbeddingBagClassifier(2, 3, 2)
    with pytest.raises(ValueError):
        train_classifier(model, [], epochs=0)
    with pytest.raises(ValueError):
        train_classifier(model, [], epochs=1)
