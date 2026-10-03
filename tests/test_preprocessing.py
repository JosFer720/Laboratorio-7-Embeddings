import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import random

from embeddings_lab.preprocessing import (
    build_vocabulary,
    corpus_statistics,
    coverage_at_k,
    encode_tokens,
    encode_tokens_iter,
    iter_skipgram_pairs,
    iter_skipgram_pairs_from_sequences,
    normalize_text,
    subsample_ids,
    subsampling_keep_probabilities,
    word_tokenize,
    wikitext_row_statistics,
)


def test_normalization_and_tokenization_are_stable():
    text = "  HELLO,\nWorld!  Café\t"

    normalized = normalize_text(text)

    assert normalized == "hello, world! café"
    assert word_tokenize(text) == ["hello", "world", "café"]


def test_tokenization_policy_for_contractions_numbers_hyphens_and_specials():
    assert word_tokenize("Can't wait: 1,234.5 state-of-the-art <unk>!") == [
        "can't",
        "wait",
        "1,234.5",
        "state",
        "of",
        "the",
        "art",
        "<unk>",
    ]


def test_wikitext_row_statistics_count_headers_blank_lines_and_token_limit():
    rows = [
        {"text": "= First Article =\nA small line.\n\n== Details =="},
        {"text": "= Second Article =\nAnother line"},
    ]

    assert wikitext_row_statistics(rows, token_limit=7) == {
        "articles": 2,
        "lines": 6,
        "non_empty_lines": 5,
        "tokens": 7,
    }


def test_wikitext_row_statistics_accepts_plain_lines_and_empty_input():
    assert wikitext_row_statistics(["= Title =", "", "body"]) == {
        "articles": 1,
        "lines": 3,
        "non_empty_lines": 2,
        "tokens": 2,
    }
    assert wikitext_row_statistics([]) == {
        "articles": 0,
        "lines": 0,
        "non_empty_lines": 0,
        "tokens": 0,
    }


def test_corpus_statistics_handle_empty_and_populated_corpora():
    assert corpus_statistics([]) == {
        "sentences": 0,
        "tokens": 0,
        "unique_tokens": 0,
        "average_tokens_per_sentence": 0.0,
    }
    assert corpus_statistics([["a", "b"], ["a"]]) == {
        "sentences": 2,
        "tokens": 3,
        "unique_tokens": 2,
        "average_tokens_per_sentence": 1.5,
    }


def test_vocabulary_is_frequency_ordered_with_unknown_at_zero():
    vocab = build_vocabulary([["pear", "apple", "pear"], ["plum", "apple"]])

    assert vocab == {"<unk>": 0, "apple": 1, "pear": 2, "plum": 3}
    assert encode_tokens(["pear", "missing", "apple"], vocab) == [2, 0, 1]


def test_vocabulary_min_count_drops_rare_words_to_unknown():
    vocab = build_vocabulary([["common", "common", "rare"]], min_count=2)

    assert vocab == {"<unk>": 0, "common": 1}
    assert encode_tokens(["rare"], vocab) == [0]


@pytest.mark.parametrize(
    "vocabulary",
    [
        {"<unk>": 0, "a": 0},
        {"<unk>": 0, "a": -1},
        {"<unk>": 0, "a": True},
        {"<unk>": 0, "a": 1.5},
    ],
)
def test_encoding_rejects_invalid_vocabulary_ids(vocabulary):
    with pytest.raises(ValueError):
        encode_tokens(["a"], vocabulary)


def test_encoding_rejects_non_string_tokens_and_has_streaming_variant():
    vocabulary = {"<unk>": 0, "known": 1}

    with pytest.raises(TypeError):
        encode_tokens(["known", 3], vocabulary)
    assert list(encode_tokens_iter((token for token in ["known", "missing"]), vocabulary)) == [
        1,
        0,
    ]
    with pytest.raises(TypeError):
        list(encode_tokens_iter([None], vocabulary))


def test_coverage_at_k_counts_token_mass_of_most_frequent_words():
    assert coverage_at_k(["a", "a", "a", "b", "b", "c"], k=2) == pytest.approx(5 / 6)
    assert coverage_at_k([], k=2) == 0.0
    assert coverage_at_k(["a", "b"], k=0) == 0.0


def test_token_count_inputs_reject_non_string_tokens():
    with pytest.raises(TypeError):
        coverage_at_k(["word", 3], k=2)
    with pytest.raises(TypeError):
        subsampling_keep_probabilities({3: 1}, threshold=0.1)


def test_subsampling_keep_probabilities_match_word2vec_formula():
    probabilities = subsampling_keep_probabilities(
        {"common": 8, "rare": 2}, threshold=0.1
    )

    assert probabilities["common"] == pytest.approx(0.4785533905932738)
    assert probabilities["rare"] == 1.0
    assert subsampling_keep_probabilities({}, threshold=0.1) == {}


def test_subsampling_rejects_non_finite_thresholds():
    with pytest.raises(ValueError):
        subsampling_keep_probabilities({"word": 1}, threshold=float("nan"))


def test_subsample_ids_is_reproducible_and_preserves_kept_order():
    ids = [0, 1, 2, 3, 4, 5]
    probabilities = [0.0, 0.25, 0.5, 0.75, 1.0, 0.25]

    assert list(subsample_ids(ids, probabilities, seed=21)) == [3, 4]
    assert list(subsample_ids(ids, probabilities, seed=21)) == [3, 4]


def test_subsample_ids_can_continue_using_a_shared_random_generator():
    probabilities = [0.5, 0.5, 0.5, 0.5]
    expected = list(subsample_ids([0, 1, 2, 3], probabilities, rng=random.Random(7)))
    generator = random.Random(7)
    actual = list(subsample_ids([0, 1], probabilities, rng=generator))
    actual.extend(subsample_ids([2, 3], probabilities, rng=generator))

    assert actual == expected


def test_subsample_ids_rejects_seed_and_rng_together():
    with pytest.raises(ValueError):
        list(subsample_ids([0], [1.0], seed=1, rng=random.Random(1)))


def test_skipgram_pairs_respect_window_and_sequence_edges():
    pairs = list(iter_skipgram_pairs([10, 20, 30, 40], window_size=1))

    assert pairs == [(10, 20), (20, 10), (20, 30), (30, 20), (30, 40), (40, 30)]


@pytest.mark.parametrize("token_id", [-1, True, 1.5, "1"])
def test_skipgram_pairs_reject_invalid_ids(token_id):
    with pytest.raises(ValueError):
        list(iter_skipgram_pairs([1, token_id], window_size=1))


def test_pairs_do_not_cross_sequence_boundaries():
    pairs = list(iter_skipgram_pairs_from_sequences([[1], [2, 3]], window_size=1))

    assert pairs == [(2, 3), (3, 2)]


def test_skipgram_pairs_stream_iterables_and_accept_empty_input():
    pairs = iter_skipgram_pairs((token for token in [1, 2, 3]), window_size=2)

    assert list(pairs) == [
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    ]
    assert list(iter_skipgram_pairs([], window_size=2)) == []


@pytest.mark.parametrize("window_size", [0, -1])
def test_skipgram_pairs_reject_nonpositive_window(window_size):
    with pytest.raises(ValueError):
        list(iter_skipgram_pairs([1, 2], window_size=window_size))
