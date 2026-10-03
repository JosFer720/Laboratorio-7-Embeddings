"""Skip-gram with negative sampling and compact training metrics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class SGNS(nn.Module):
    """Skip-gram model with separate center and context embedding tables."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        if seed is None:
            self.reset_parameters()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 0.5 / self.input_embeddings.embedding_dim
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.output_embeddings.weight, -bound, bound)

    def forward(
        self, centers: torch.Tensor, contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_embeddings(centers), self.output_embeddings(contexts)

    @property
    def embeddings(self) -> torch.Tensor:
        """Return center vectors, the usual representation for downstream use."""
        return self.input_embeddings.weight


def negative_sampling_distribution(
    counts: torch.Tensor | Sequence[float],
    power: float = 0.75,
    reserved_ids: Iterable[int] = (),
) -> torch.Tensor:
    """Build the normalized count-to-power distribution for negative sampling."""
    if not math.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    weights = torch.as_tensor(counts, dtype=torch.float64).clone()
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("counts must be a non-empty one-dimensional sequence")
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("counts must be finite and non-negative")

    reserved = tuple(reserved_ids)
    if any(not isinstance(index, int) or index < 0 or index >= weights.numel() for index in reserved):
        raise ValueError("reserved_ids must contain valid vocabulary indices")
    weights.pow_(power)
    if reserved:
        weights[list(reserved)] = 0
    total = weights.sum()
    if total <= 0:
        raise ValueError("at least one non-reserved token must have positive count")
    return (weights / total).to(dtype=torch.float32)


def sample_negatives(
    distribution: torch.Tensor,
    batch_size: int,
    num_negatives: int,
    *,
    generator: torch.Generator | None = None,
    reserved_ids: Iterable[int] = (),
) -> torch.Tensor:
    """Sample a batch of vocabulary IDs with replacement."""
    if batch_size <= 0 or num_negatives <= 0:
        raise ValueError("batch_size and num_negatives must be positive")
    probabilities = torch.as_tensor(distribution, dtype=torch.float32).clone()
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("distribution must be a non-empty one-dimensional tensor")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("distribution must be finite and non-negative")
    reserved = tuple(reserved_ids)
    if any(not isinstance(index, int) or index < 0 or index >= probabilities.numel() for index in reserved):
        raise ValueError("reserved_ids must contain valid vocabulary indices")
    if reserved:
        probabilities[list(reserved)] = 0
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("at least one non-reserved ID must have positive probability")
    probabilities /= total
    return _sample_negatives_prepared(
        probabilities, batch_size, num_negatives, generator=generator
    )


def _sample_negatives_prepared(
    probabilities: torch.Tensor,
    batch_size: int,
    num_negatives: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample directly from a validated, normalized distribution."""
    samples = torch.multinomial(
        probabilities,
        batch_size * num_negatives,
        replacement=True,
        generator=generator,
    )
    return samples.view(batch_size, num_negatives)


class _NegativeSamplePool:
    """Reuse buffered draws to avoid sampling from the full vocabulary per batch."""

    def __init__(
        self,
        probabilities: torch.Tensor,
        pool_size: int,
        generator: torch.Generator,
    ) -> None:
        self.probabilities = probabilities
        self.pool_size = pool_size
        self.generator = generator
        self.pool = torch.empty(0, dtype=torch.long, device=probabilities.device)
        self.cursor = 0

    def sample(self, batch_size: int, num_negatives: int) -> torch.Tensor:
        requested = batch_size * num_negatives
        remaining = self.pool.numel() - self.cursor
        if remaining < requested:
            draw_count = max(self.pool_size, requested)
            self.pool = torch.multinomial(
                self.probabilities,
                draw_count,
                replacement=True,
                generator=self.generator,
            )
            self.cursor = 0
        start = self.cursor
        self.cursor += requested
        return self.pool[start : self.cursor].view(batch_size, num_negatives)


def sgns_loss(
    center_vectors: torch.Tensor,
    positive_context_vectors: torch.Tensor,
    negative_context_vectors: torch.Tensor,
) -> torch.Tensor:
    """Compute mean negative log likelihood for positive and negative pairs."""
    if center_vectors.ndim != 2 or positive_context_vectors.shape != center_vectors.shape:
        raise ValueError("center and positive context vectors must have matching [batch, dim] shapes")
    if (
        negative_context_vectors.ndim != 3
        or negative_context_vectors.shape[0] != center_vectors.shape[0]
        or negative_context_vectors.shape[2] != center_vectors.shape[1]
        or negative_context_vectors.shape[1] == 0
    ):
        raise ValueError("negative context vectors must have shape [batch, negatives, dim]")
    positive_scores = (center_vectors * positive_context_vectors).sum(dim=1)
    negative_scores = torch.bmm(
        negative_context_vectors, center_vectors.unsqueeze(2)
    ).squeeze(2)
    return -(F.logsigmoid(positive_scores) + F.logsigmoid(-negative_scores).sum(dim=1)).mean()


def _pair_batches(
    pairs: torch.Tensor | Iterable[Sequence[int]] | Callable[[], Iterable[Sequence[int]]],
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
):
    source = pairs() if callable(pairs) else pairs
    if isinstance(source, torch.Tensor):
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("pairs must have shape [number of pairs, 2]")
        if source.shape[0] == 0:
            return
        order = torch.randperm(
            source.shape[0], generator=generator, device=device
        ).cpu()
        for start in range(0, source.shape[0], batch_size):
            batch = source[order[start : start + batch_size]]
            yield batch[:, 0].to(device=device, dtype=torch.long), batch[:, 1].to(
                device=device, dtype=torch.long
            )
        return

    center_buffer: list[int] = []
    context_buffer: list[int] = []
    for pair in source:
        if len(pair) != 2:
            raise ValueError("each pair must contain a center and a context ID")
        center_buffer.append(int(pair[0]))
        context_buffer.append(int(pair[1]))
        if len(center_buffer) == batch_size:
            yield torch.tensor(center_buffer, dtype=torch.long, device=device), torch.tensor(
                context_buffer, dtype=torch.long, device=device
            )
            center_buffer.clear()
            context_buffer.clear()
    if center_buffer:
        yield torch.tensor(center_buffer, dtype=torch.long, device=device), torch.tensor(
            context_buffer, dtype=torch.long, device=device
        )


@contextmanager
def _cuda_deterministic_algorithms(device: torch.device):
    if device.type != "cuda":
        yield
        return
    previous_setting = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_setting, warn_only=previous_warn_only
        )


def train_sgns(
    model: SGNS,
    pairs: torch.Tensor | Iterable[Sequence[int]] | Callable[[], Iterable[Sequence[int]]],
    counts: torch.Tensor | Sequence[float],
    *,
    epochs: int = 5,
    batch_size: int = 512,
    num_negatives: int = 5,
    negative_pool_size: int = 1_000_000,
    learning_rate: float = 0.025,
    seed: int = 0,
    device: str | torch.device = "cpu",
    reserved_ids: Iterable[int] = (),
    epoch_callback: Callable[
        [int, SGNS, dict[str, float]], Mapping[str, object] | None
    ]
    | None = None,
) -> dict[str, object]:
    """Train SGNS and return losses, timings, and peak GPU memory.

    Reusable iterables or callable factories support multiple epochs. A one-shot
    iterator is rejected before training when more than one epoch is requested.
    CUDA training enables deterministic algorithms for its duration. PyTorch can
    raise an error when an operation lacks a deterministic implementation, and
    results may vary across PyTorch, CUDA, cuDNN, and hardware versions. Older
    PyTorch versions may require CUBLAS_WORKSPACE_CONFIG to be set before CUDA
    initialization for deterministic bmm behavior. This function does not change
    environment variables.
    """
    if epochs <= 0 or batch_size <= 0 or num_negatives <= 0:
        raise ValueError("epochs, batch_size, and num_negatives must be positive")
    if (
        isinstance(negative_pool_size, bool)
        or not isinstance(negative_pool_size, int)
        or negative_pool_size <= 0
    ):
        raise ValueError("negative_pool_size must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if epochs > 1 and not callable(pairs) and iter(pairs) is pairs:
        raise ValueError("pairs must be a reusable iterable or callable factory for multiple epochs")
    reserved_ids = tuple(reserved_ids)

    training_device = torch.device(device)
    model.to(training_device)
    distribution = negative_sampling_distribution(counts, reserved_ids=reserved_ids).to(
        training_device
    )
    generator = torch.Generator(device=training_device).manual_seed(seed)
    negative_pool = _NegativeSamplePool(
        distribution, negative_pool_size, generator
    )
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    if training_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(training_device)

    start_time = time.perf_counter()
    epoch_losses: list[float] = []
    epoch_times: list[float] = []
    epoch_records: list[dict[str, object]] = []
    with _cuda_deterministic_algorithms(training_device):
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            weighted_loss = 0.0
            seen = 0
            model.train()
            for centers, contexts in _pair_batches(pairs, batch_size, training_device, generator):
                negative_ids = negative_pool.sample(centers.shape[0], num_negatives)
                center_vectors = model.input_embeddings(centers)
                context_vectors = model.output_embeddings(contexts)
                negative_vectors = model.output_embeddings(negative_ids)
                loss = sgns_loss(center_vectors, context_vectors, negative_vectors)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                weighted_loss += loss.detach().item() * centers.shape[0]
                seen += centers.shape[0]
            if seen == 0:
                raise ValueError("pairs must contain at least one pair")
            if training_device.type == "cuda":
                torch.cuda.synchronize(training_device)
            epoch_loss = weighted_loss / seen
            epoch_duration = time.perf_counter() - epoch_start
            epoch_losses.append(epoch_loss)
            epoch_times.append(epoch_duration)
            callback_metrics = {
                "loss": epoch_loss,
                "duration_seconds": epoch_duration,
            }
            record: dict[str, object] = {"epoch": epoch, **callback_metrics}
            if epoch_callback is not None:
                additional = epoch_callback(epoch, model, callback_metrics)
                if additional is not None:
                    record.update(additional)
            epoch_records.append(record)

    if training_device.type == "cuda":
        torch.cuda.synchronize(training_device)
        peak_gpu_memory = torch.cuda.max_memory_allocated(training_device)
    else:
        peak_gpu_memory = 0

    return {
        "model": model,
        "epoch_losses": epoch_losses,
        "epoch_times_seconds": epoch_times,
        "epoch_records": epoch_records,
        "total_time_seconds": time.perf_counter() - start_time,
        "peak_gpu_memory_bytes": peak_gpu_memory,
    }
