import torch
import pytest

import embeddings_lab.sgns as sgns_module
from embeddings_lab.sgns import (
    SGNS,
    _cuda_deterministic_algorithms,
    negative_sampling_distribution,
    sample_negatives,
    sgns_loss,
    train_sgns,
)


def test_sgns_returns_center_and_context_embeddings_with_requested_shapes():
    model = SGNS(vocab_size=7, embedding_dim=4)
    centers = torch.tensor([0, 3, 6])
    contexts = torch.tensor([2, 5, 1])

    center_vectors, context_vectors = model(centers, contexts)

    assert center_vectors.shape == (3, 4)
    assert context_vectors.shape == (3, 4)
    assert isinstance(model.input_embeddings, torch.nn.Embedding)
    assert isinstance(model.output_embeddings, torch.nn.Embedding)


def test_negative_sampling_distribution_uses_three_quarters_power_and_can_exclude_ids():
    distribution = negative_sampling_distribution(
        torch.tensor([1, 16, 81, 256], dtype=torch.float32),
        power=0.75,
        reserved_ids=[0],
    )

    expected = torch.tensor([0.0, 8.0, 27.0, 64.0])
    expected = expected / expected.sum()
    torch.testing.assert_close(distribution, expected)


def test_sample_negatives_never_returns_requested_reserved_ids():
    distribution = negative_sampling_distribution(
        torch.tensor([2, 3, 5, 7]), reserved_ids=[0, 2]
    )

    negatives = sample_negatives(
        distribution,
        batch_size=12,
        num_negatives=5,
        generator=torch.Generator().manual_seed(17),
        reserved_ids=[0, 2],
    )

    assert negatives.shape == (12, 5)
    assert not torch.isin(negatives, torch.tensor([0, 2])).any()


def test_sgns_loss_is_finite_and_rewards_positive_pairs():
    centers = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positive_contexts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negatives = torch.tensor([[[0.0, 1.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]]])

    loss = sgns_loss(centers, positive_contexts, negatives)
    reversed_loss = sgns_loss(centers, negatives[:, 0], positive_contexts.unsqueeze(1))

    assert torch.isfinite(loss)
    assert loss.item() < reversed_loss.item()


def test_train_sgns_reduces_loss_on_a_tiny_repeating_corpus_and_reports_timings():
    pairs = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]] * 24)
    counts = torch.tensor([24, 24, 24, 24], dtype=torch.float32)
    torch.manual_seed(23)
    model = SGNS(vocab_size=4, embedding_dim=12)

    result = train_sgns(
        model,
        pairs,
        counts,
        epochs=12,
        batch_size=16,
        num_negatives=3,
        learning_rate=0.08,
        seed=31,
        device="cpu",
    )

    assert result["model"] is model
    assert len(result["epoch_losses"]) == 12
    assert result["epoch_losses"][-1] < result["epoch_losses"][0]
    assert len(result["epoch_times_seconds"]) == 12
    assert all(duration >= 0 for duration in result["epoch_times_seconds"])
    assert result["total_time_seconds"] >= 0
    assert result["peak_gpu_memory_bytes"] == 0


def test_train_sgns_records_epoch_callback_metrics():
    pairs = torch.tensor([[0, 1], [1, 0]] * 8)
    counts = torch.tensor([8, 8], dtype=torch.float32)
    model = SGNS(vocab_size=2, embedding_dim=4, seed=5)
    observed_epochs = []

    def callback(epoch, current_model, metrics):
        observed_epochs.append(epoch)
        assert current_model is model
        assert metrics["loss"] >= 0
        return {"analogy_accuracy": epoch / 10}

    result = train_sgns(
        model,
        pairs,
        counts,
        epochs=2,
        batch_size=4,
        seed=7,
        epoch_callback=callback,
    )

    assert observed_epochs == [1, 2]
    assert result["epoch_records"][0]["analogy_accuracy"] == 0.1
    assert result["epoch_records"][1]["analogy_accuracy"] == 0.2


def test_training_uses_a_buffered_negative_pool(monkeypatch):
    pairs = torch.tensor([[0, 1], [1, 0]] * 32)
    counts = torch.tensor([32, 32], dtype=torch.float32)
    model = SGNS(vocab_size=2, embedding_dim=4, seed=3)
    original = torch.multinomial
    calls = 0

    def counted_multinomial(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", counted_multinomial)
    train_sgns(
        model,
        pairs,
        counts,
        epochs=1,
        batch_size=4,
        num_negatives=2,
        negative_pool_size=64,
    )

    assert calls == 2


def test_train_sgns_rejects_one_shot_iterators_before_training_multiple_epochs():
    model = SGNS(vocab_size=2, embedding_dim=4, seed=3)
    initial_weights = model.input_embeddings.weight.detach().clone()

    with pytest.raises(ValueError, match="callable"):
        train_sgns(
            model,
            iter([(0, 1), (1, 0)]),
            [1, 1],
            epochs=2,
            device="cpu",
        )

    torch.testing.assert_close(model.input_embeddings.weight, initial_weights)


def test_sgns_embedding_tables_use_sparse_gradients():
    model = SGNS(vocab_size=5, embedding_dim=3)

    assert model.input_embeddings.sparse
    assert model.output_embeddings.sparse


def test_cuda_determinism_restores_enabled_and_warn_only_after_exception():
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)

        with pytest.raises(RuntimeError, match="sentinel"):
            with _cuda_deterministic_algorithms(torch.device("cuda")):
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                raise RuntimeError("sentinel")

        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


def test_training_samples_from_prepared_distribution_without_public_validation(monkeypatch):
    def reject_public_sampler(*args, **kwargs):
        raise AssertionError("training must not call the validating sampler per batch")

    monkeypatch.setattr(sgns_module, "sample_negatives", reject_public_sampler)
    model = SGNS(vocab_size=4, embedding_dim=5, seed=13)
    pairs = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]])

    result = train_sgns(
        model,
        pairs,
        [1, 1, 1, 1],
        epochs=1,
        batch_size=1,
        num_negatives=2,
        seed=29,
        device="cpu",
    )

    assert len(result["epoch_losses"]) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_activates_determinism_and_repeats_exactly(monkeypatch):
    deterministic_calls = []
    use_deterministic_algorithms = torch.use_deterministic_algorithms

    def record_determinism(enabled, *args, **kwargs):
        deterministic_calls.append(enabled)
        return use_deterministic_algorithms(enabled, *args, **kwargs)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", record_determinism)
    pairs = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]] * 8)
    counts = torch.tensor([16, 16, 16, 16], dtype=torch.float32)
    first = SGNS(vocab_size=4, embedding_dim=8, seed=5)
    second = SGNS(vocab_size=4, embedding_dim=8, seed=5)

    train_sgns(first, pairs, counts, epochs=3, batch_size=8, seed=19, device="cuda")
    train_sgns(second, pairs, counts, epochs=3, batch_size=8, seed=19, device="cuda")

    assert deterministic_calls[0] is True
    assert deterministic_calls[2] is True
    torch.testing.assert_close(
        first.input_embeddings.weight, second.input_embeddings.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        first.output_embeddings.weight, second.output_embeddings.weight, rtol=0, atol=0
    )
