"""Cosine neighbors, word analogies, ranks and t-SNE coordinates."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.manifold import TSNE


def _matrix(vectors: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("vectors must be a nonempty two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError("vectors must contain finite values")
    return values


def _space(
    vectors: ArrayLike, vocabulary: Mapping[str, int]
) -> tuple[NDArray[np.float64], list[str], NDArray[np.int64]]:
    values = _matrix(vectors)
    words = list(vocabulary)
    indices = np.asarray(list(vocabulary.values()))
    if not words or indices.dtype.kind not in "iu":
        raise ValueError("vocabulary must map words to integer row indices")
    if np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("vocabulary indices must reference vector rows")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("vocabulary indices must be unique")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return normalized, words, indices.astype(np.int64)


def _top(
    scores: NDArray[np.float64], words: list[str], top_k: int, exclude: Iterable[str]
) -> list[tuple[str, float]]:
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    excluded = set(exclude)
    eligible = np.asarray([i for i, word in enumerate(words) if word not in excluded], dtype=int)
    count = min(top_k, len(eligible))
    if count == 0:
        return []
    if count < len(eligible):
        candidate_scores = scores[eligible]
        partition = np.argpartition(candidate_scores, len(eligible) - count)
        cutoff = candidate_scores[partition[-count]]
        better = eligible[candidate_scores > cutoff]
        tied = eligible[candidate_scores == cutoff][:count - len(better)]
        eligible = np.concatenate((better, tied))
    order = np.argsort(-scores[eligible], kind="stable")
    return [(words[index], float(scores[index])) for index in eligible[order]]


def most_similar(
    vectors: ArrayLike,
    vocabulary: Mapping[str, int],
    query: str | ArrayLike,
    *,
    top_k: int = 10,
    exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """Return cosine neighbors, excluding a word query itself by default."""
    normalized, words, indices = _space(vectors, vocabulary)
    excluded = set(exclude)
    if isinstance(query, str):
        vector = normalized[vocabulary[query]]
        excluded.add(query)
    else:
        vector = np.asarray(query, dtype=np.float64)
        if vector.shape != (normalized.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("query must be a finite vector with the embedding dimension")
        norm = np.linalg.norm(vector)
        vector = vector / norm if norm > 0 else np.zeros_like(vector)
    return _top(normalized[indices] @ vector, words, top_k, excluded)


def _analogy_scores(
    normalized: NDArray[np.float64],
    candidates: NDArray[np.float64],
    vocabulary: Mapping[str, int],
    a: str,
    b: str,
    c: str,
    method: str,
) -> NDArray[np.float64]:
    va, vb, vc = (normalized[vocabulary[word]] for word in (a, b, c))
    if method == "3cosadd":
        query = vb - va + vc
        norm = np.linalg.norm(query)
        return candidates @ (query / norm if norm > 0 else np.zeros_like(query))
    similarities = np.clip((candidates @ np.stack([va, vb, vc], axis=1) + 1) / 2, 0, 1)
    return similarities[:, 1] * similarities[:, 2] / (similarities[:, 0] + 1e-6)


def analogy_3cosadd(
    vectors: ArrayLike,
    vocabulary: Mapping[str, int],
    a: str,
    b: str,
    c: str,
    *,
    top_k: int = 10,
    exclude_inputs: bool = True,
    exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """Find d for a:b::c:d using normalized vector addition."""
    normalized, words, indices = _space(vectors, vocabulary)
    scores = _analogy_scores(normalized, normalized[indices], vocabulary, a, b, c, "3cosadd")
    excluded = set(exclude) | ({a, b, c} if exclude_inputs else set())
    return _top(scores, words, top_k, excluded)


def analogy_3cosmul(
    vectors: ArrayLike,
    vocabulary: Mapping[str, int],
    a: str,
    b: str,
    c: str,
    *,
    top_k: int = 10,
    exclude_inputs: bool = True,
    exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """Find d with shifted cosine products and a stable denominator."""
    normalized, words, indices = _space(vectors, vocabulary)
    scores = _analogy_scores(normalized, normalized[indices], vocabulary, a, b, c, "3cosmul")
    excluded = set(exclude) | ({a, b, c} if exclude_inputs else set())
    return _top(scores, words, top_k, excluded)


def answer_rank(candidates: Sequence[tuple[str, float]], answer: str) -> int | None:
    """Return a one-based rank or None when the answer is absent."""
    return next((rank for rank, (word, _) in enumerate(candidates, start=1) if word == answer), None)


def evaluate_analogies(
    vectors: ArrayLike,
    vocabulary: Mapping[str, int],
    analogies: Iterable[Sequence[str]],
    *,
    method: str = "3cosadd",
    exclude_inputs: bool = True,
) -> dict[str, int | float]:
    """Report coverage and rank metrics over fully in-vocabulary records."""
    if method not in {"3cosadd", "3cosmul"}:
        raise ValueError("method must be 3cosadd or 3cosmul")
    normalized, words, indices = _space(vectors, vocabulary)
    candidates = normalized[indices]
    positions = {word: position for position, word in enumerate(words)}
    total = evaluated = correct = 0
    ranks: list[int] = []
    reciprocal_sum = 0.0
    for record in analogies:
        if len(record) != 4:
            raise ValueError("each analogy must contain four words")
        total += 1
        if any(word not in vocabulary for word in record):
            continue
        a, b, c, answer = record
        evaluated += 1
        excluded = {a, b, c} if exclude_inputs else set()
        if answer in excluded:
            continue
        scores = _analogy_scores(normalized, candidates, vocabulary, a, b, c, method)
        position = positions[answer]
        eligible = np.ones(len(words), dtype=bool)
        for word in excluded:
            eligible[positions[word]] = False
        answer_score = scores[position]
        better = np.count_nonzero(eligible & (scores > answer_score))
        earlier_ties = np.count_nonzero(eligible[:position] & (scores[:position] == answer_score))
        rank = 1 + int(better + earlier_ties)
        ranks.append(rank)
        reciprocal_sum += 1 / rank
        correct += rank == 1
    return {
        "total": total,
        "evaluated": evaluated,
        "coverage": evaluated / total if total else 0.0,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "mrr": reciprocal_sum / evaluated if evaluated else 0.0,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
    }


def tsne_projection(
    vectors: ArrayLike,
    *,
    perplexity: float = 30.0,
    seed: int = 42,
    max_iter: int = 1000,
) -> NDArray[np.float32]:
    """Project vectors into two dimensions with a reproducible random state."""
    values = _matrix(vectors)
    if len(values) < 2:
        raise ValueError("t-SNE requires at least two rows")
    if not np.isfinite(perplexity) or perplexity <= 0:
        raise ValueError("perplexity must be positive and finite")
    if max_iter < 250:
        raise ValueError("max_iter must be at least 250")
    return TSNE(
        n_components=2,
        perplexity=min(float(perplexity), len(values) - 1),
        random_state=seed,
        init="random",
        learning_rate="auto",
        max_iter=max_iter,
    ).fit_transform(values)
