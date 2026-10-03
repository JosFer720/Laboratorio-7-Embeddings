"""Text normalization, vocabulary construction, and skip-gram preparation."""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import islice
from math import isfinite, sqrt
from typing import TypeAlias


TokenSequence: TypeAlias = Iterable[str]
Vocabulary: TypeAlias = Mapping[str, int]
_TOKEN_PATTERN = re.compile(
    r"(?<![\w<])<[A-Za-z][A-Za-z0-9_]*>(?![\w>])"
    r"|\d+(?:[,.]\d+)*"
    r"|[\w]+(?:['’][\w]+)*",
    flags=re.UNICODE,
)
_WIKITEXT_ARTICLE_HEADER = re.compile(r"^=\s+[^=].*?\s+=$")


def normalize_text(text: str) -> str:
    """Apply Unicode compatibility normalization, lowercase, and spacing."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def word_tokenize(text: str) -> list[str]:
    """Keep contractions, numbers, and isolated special tokens as single tokens.

    Hyphens and other punctuation separate tokens. Commas and periods inside
    numbers remain part of the numeric token.
    """
    return _TOKEN_PATTERN.findall(normalize_text(text))


def wikitext_row_statistics(
    rows: Iterable[str | Mapping[str, object]], token_limit: int | None = None
) -> dict[str, int]:
    """Count WikiText articles, lines, non-empty lines, and limited tokens.

    Rows may be text strings or mappings with a string ``text`` field. Line
    counters cover all rows while token counting stops at ``token_limit``.
    """
    if token_limit is not None and (
        not isinstance(token_limit, int)
        or isinstance(token_limit, bool)
        or token_limit < 0
    ):
        raise ValueError("token_limit must be a non-negative integer or None")

    articles = lines_count = non_empty_lines = tokens = 0
    for row in rows:
        if isinstance(row, str):
            text = row
        elif isinstance(row, Mapping) and isinstance(row.get("text"), str):
            text = row["text"]
        else:
            raise TypeError("each row must be a string or contain string text")

        lines = text.splitlines() or [text]
        for line in lines:
            lines_count += 1
            stripped = line.strip()
            if not stripped:
                continue
            non_empty_lines += 1
            if _WIKITEXT_ARTICLE_HEADER.fullmatch(stripped):
                articles += 1
            if token_limit is None or tokens < token_limit:
                for _ in word_tokenize(line):
                    if token_limit is not None and tokens >= token_limit:
                        break
                    tokens += 1

    return {
        "articles": articles,
        "lines": lines_count,
        "non_empty_lines": non_empty_lines,
        "tokens": tokens,
    }


def corpus_statistics(corpus: Iterable[TokenSequence]) -> dict[str, int | float]:
    """Count sentences, tokens, distinct tokens, and mean sentence length."""
    sentences = 0
    tokens = 0
    unique_tokens: set[str] = set()
    for sentence in corpus:
        sentences += 1
        for token in sentence:
            if not isinstance(token, str):
                raise TypeError("corpus tokens must be strings")
            tokens += 1
            unique_tokens.add(token)
    return {
        "sentences": sentences,
        "tokens": tokens,
        "unique_tokens": len(unique_tokens),
        "average_tokens_per_sentence": tokens / sentences if sentences else 0.0,
    }


def build_vocabulary(
    corpus: Iterable[TokenSequence],
    min_count: int = 1,
    unknown_token: str = "<unk>",
) -> dict[str, int]:
    """Build deterministic token IDs ordered by descending frequency."""
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if not isinstance(unknown_token, str) or not unknown_token:
        raise ValueError("unknown_token must be a non-empty string")

    counts: Counter[str] = Counter()
    for sentence in corpus:
        for token in sentence:
            if not isinstance(token, str):
                raise TypeError("corpus tokens must be strings")
            if token != unknown_token:
                counts[token] += 1

    ordered_tokens = sorted(
        (token for token, count in counts.items() if count >= min_count),
        key=lambda token: (-counts[token], token),
    )
    return {
        unknown_token: 0,
        **{token: index for index, token in enumerate(ordered_tokens, 1)},
    }


def encode_tokens(
    tokens: Iterable[str], vocabulary: Vocabulary, unknown_token: str = "<unk>"
) -> list[int]:
    """Encode tokens, mapping words absent from the vocabulary to unknown."""
    return list(encode_tokens_iter(tokens, vocabulary, unknown_token))


def encode_tokens_iter(
    tokens: Iterable[str], vocabulary: Vocabulary, unknown_token: str = "<unk>"
) -> Iterator[int]:
    """Stream token IDs while mapping words outside the vocabulary to unknown."""
    _validate_vocabulary(vocabulary)
    if unknown_token not in vocabulary:
        raise ValueError("vocabulary must contain the unknown token")
    unknown_id = vocabulary[unknown_token]
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("input tokens must be strings")
        yield vocabulary.get(token, unknown_id)


def _validate_vocabulary(vocabulary: Vocabulary) -> None:
    if not isinstance(vocabulary, Mapping):
        raise TypeError("vocabulary must be a mapping")
    ids: set[int] = set()
    for token, token_id in vocabulary.items():
        if not isinstance(token, str):
            raise TypeError("vocabulary tokens must be strings")
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("vocabulary IDs must be non-negative integers")
        if token_id in ids:
            raise ValueError("vocabulary IDs must be unique")
        ids.add(token_id)


def coverage_at_k(tokens: Iterable[str], k: int) -> float:
    """Return the fraction of token occurrences covered by the top k words."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a non-negative integer")
    counts: Counter[str] = Counter()
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("input tokens must be strings")
        counts[token] += 1
    total = sum(counts.values())
    if total == 0 or k == 0:
        return 0.0
    covered = sum(count for _, count in counts.most_common(k))
    return covered / total


def subsampling_keep_probabilities(
    token_counts: Mapping[str, int], threshold: float = 1e-5
) -> dict[str, float]:
    """Compute word2vec keep probabilities from corpus token counts."""
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be numeric")
    if not isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if any(not isinstance(token, str) for token in token_counts):
        raise TypeError("token count keys must be strings")
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in token_counts.values()
    ):
        raise ValueError("token counts must be non-negative integers")
    total = sum(token_counts.values())
    if total == 0:
        return {token: 1.0 for token in token_counts}
    return {
        token: min(1.0, sqrt(threshold / (count / total)) + threshold / (count / total))
        if count
        else 1.0
        for token, count in token_counts.items()
    }


def subsample_ids(
    token_ids: Iterable[int],
    keep_probabilities: Sequence[float] | Mapping[int, float],
    seed: int | None = None,
    rng: random.Random | None = None,
) -> Iterator[int]:
    """Yield retained IDs using either a seed or a shared random generator."""
    if seed is not None and rng is not None:
        raise ValueError("provide seed or rng, not both")
    if rng is not None and not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None")
    generator = rng if rng is not None else random.Random(seed)
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("token IDs must be non-negative integers")
        try:
            probability = keep_probabilities[token_id]
        except (IndexError, KeyError) as error:
            raise ValueError(f"no keep probability for token ID {token_id}") from error
        if not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
            raise ValueError("keep probabilities must be between zero and one")
        if generator.random() < probability:
            yield token_id


def iter_skipgram_pairs(
    token_ids: Iterable[int], window_size: int
) -> Iterator[tuple[int, int]]:
    """Stream directed center-context pairs within a fixed symmetric window."""
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        raise ValueError("window_size must be a positive integer")

    iterator = iter(_validated_token_ids(token_ids))
    sentinel = object()
    current = next(iterator, sentinel)
    if current is sentinel:
        return
    history: deque[int] = deque(maxlen=window_size)
    future: deque[int] = deque(islice(iterator, window_size))
    while True:
        for context in history:
            yield current, context
        for context in future:
            yield current, context

        history.append(current)
        if not future:
            break
        current = future.popleft()
        following = next(iterator, sentinel)
        if following is not sentinel:
            future.append(following)


def iter_skipgram_pairs_from_sequences(
    sequences: Iterable[Iterable[int]], window_size: int
) -> Iterator[tuple[int, int]]:
    """Stream skip-gram pairs without crossing sequence boundaries."""
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    for sequence in sequences:
        yield from iter_skipgram_pairs(sequence, window_size)


def _validated_token_ids(token_ids: Iterable[int]) -> Iterator[int]:
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("token IDs must be non-negative integers")
        yield token_id
