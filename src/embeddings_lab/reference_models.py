"""Reference embeddings: gensim skip-gram on the shared corpus and pretrained GloVe."""

from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from numpy.typing import NDArray

EpochEvaluator = Callable[[int, KeyedVectors], Mapping[str, object]]


def load_encoded_sentences(
    corpus_path: str | Path,
    vocabulary: Mapping[str, int],
    *,
    unknown_token: str = "<unk>",
    max_tokens: int | None = None,
) -> list[list[str]]:
    """Read the corpus lines and replace tokens outside the vocabulary by the unknown token.

    Strings are shared between sentences so the token lists stay small in memory.
    """
    canonical = {word: word for word in vocabulary}
    unknown = canonical[unknown_token]
    sentences: list[list[str]] = []
    total = 0
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            if max_tokens is not None:
                tokens = tokens[: max_tokens - total]
            sentences.append([canonical.get(token, unknown) for token in tokens])
            total += len(tokens)
            if max_tokens is not None and total >= max_tokens:
                break
    return sentences


def token_total(sentences: list[list[str]]) -> int:
    """Count the tokens of a list of sentences."""
    return sum(len(sentence) for sentence in sentences)


class _EpochRecorder(CallbackAny2Vec):
    """Measure the training time of each epoch and run an optional evaluation after it."""

    def __init__(self, evaluator: EpochEvaluator | None) -> None:
        self.evaluator = evaluator
        self.records: list[dict[str, object]] = []
        self._started = 0.0

    def on_epoch_begin(self, model: Word2Vec) -> None:
        self._started = perf_counter()

    def on_epoch_end(self, model: Word2Vec) -> None:
        record: dict[str, object] = {
            "epoch": len(self.records) + 1,
            "duration_seconds": perf_counter() - self._started,
        }
        if self.evaluator is not None:
            record.update(self.evaluator(record["epoch"], model.wv))
        self.records.append(record)


def train_gensim_skipgram(
    sentences: list[list[str]],
    *,
    dimension: int = 100,
    window: int = 5,
    negatives: int = 5,
    min_count: int = 5,
    sample: float = 1e-5,
    epochs: int = 5,
    alpha: float = 0.025,
    min_alpha: float = 0.0001,
    seed: int = 42,
    workers: int = 1,
    epoch_evaluator: EpochEvaluator | None = None,
) -> dict[str, object]:
    """Train skip-gram with negative sampling using a fixed window and the same corpus."""
    model = Word2Vec(
        vector_size=dimension, window=window, min_count=min_count, sample=sample,
        negative=negatives, ns_exponent=0.75, sg=1, hs=0, alpha=alpha, min_alpha=min_alpha,
        seed=seed, workers=workers, epochs=epochs, shrink_windows=False,
    )
    started = perf_counter()
    model.build_vocab(sentences)
    vocabulary_seconds = perf_counter() - started
    recorder = _EpochRecorder(epoch_evaluator)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs, callbacks=[recorder])
    return {
        "model": model,
        "epoch_records": recorder.records,
        "vocabulary_seconds": vocabulary_seconds,
        "total_time_seconds": sum(record["duration_seconds"] for record in recorder.records),
        "tokens": token_total(sentences),
        "peak_memory_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", 0)),
    }


def load_glove(name: str = "glove-wiki-gigaword-100") -> KeyedVectors:
    """Download or reuse the pretrained vectors without altering them."""
    import gensim.downloader as api

    return api.load(name)


def glove_embedding_table(
    keyed_vectors: KeyedVectors, *, unknown_token: str = "<unk>"
) -> tuple[dict[str, int], NDArray[np.float32]]:
    """Build a lookup table whose first row is an unknown vector and the rest is a copy of GloVe."""
    if unknown_token in keyed_vectors.key_to_index:
        raise ValueError("the unknown token is already part of the pretrained vocabulary")
    unknown_row = keyed_vectors.vectors.mean(axis=0, keepdims=True)
    table = np.concatenate([unknown_row, keyed_vectors.vectors], axis=0).astype(np.float32)
    vocabulary = {unknown_token: 0}
    vocabulary.update({word: index + 1 for word, index in keyed_vectors.key_to_index.items()})
    return vocabulary, table
