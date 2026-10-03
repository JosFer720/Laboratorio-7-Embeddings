import numpy as np
import pytest

from embeddings_lab.analogies import evaluate_analogies
from embeddings_lab.benchmark import (
    difference_parallelism,
    epoch_evaluation,
    evaluate_by_category,
    load_analogy_records,
    load_similarity_pairs,
    restrict_space,
    select_thematic_words,
    shared_vocabulary,
    spearman_similarity,
)


def _toy_space():
    # Two exact parallel relations: gender (man to woman) and royalty (man to king).
    base = {"man": (1.0, 0.0, 0.0), "woman": (1.0, 1.0, 0.0), "king": (1.0, 0.0, 1.0), "queen": (1.0, 1.0, 1.0),
            "boy": (2.0, 0.0, 0.0), "girl": (2.0, 1.0, 0.0), "prince": (2.0, 0.0, 1.0), "princess": (2.0, 1.0, 1.0)}
    words = list(base)
    return np.asarray([base[word] for word in words]), {word: index for index, word in enumerate(words)}


RECORDS = [
    ("family", "man", "woman", "king", "queen"),
    ("family", "boy", "girl", "prince", "princess"),
    ("gram1-adjective-to-adverb", "man", "king", "woman", "queen"),
    ("gram1-adjective-to-adverb", "man", "ghost", "woman", "queen"),
]


def test_loaders_skip_comments_and_lowercase(tmp_path):
    pairs_file = tmp_path / "pairs.tsv"
    pairs_file.write_text("# header\nLove\tSex\t6.77\n\nTiger\tCat\t7.35\n", encoding="utf-8")
    analogy_file = tmp_path / "questions.txt"
    analogy_file.write_text(": capital\nAthens Greece Baghdad Iraq\n: family\nBoy Girl King Queen\n", encoding="utf-8")
    assert load_similarity_pairs(str(pairs_file)) == [("love", "sex", 6.77), ("tiger", "cat", 7.35)]
    assert load_analogy_records(str(analogy_file)) == [
        ("capital", "athens", "greece", "baghdad", "iraq"), ("family", "boy", "girl", "king", "queen"),
    ]


def test_spearman_ignores_missing_pairs_and_reports_coverage():
    vectors, vocabulary = _toy_space()
    pairs = [("man", "man", 10.0), ("man", "woman", 5.0), ("man", "queen", 1.0), ("man", "unknown", 5.0)]
    result = spearman_similarity(vectors, vocabulary, pairs)
    assert result["pairs"] == 3 and result["total"] == 4 and result["coverage"] == 0.75
    assert result["spearman"] == pytest.approx(1.0)


def test_shared_vocabulary_keeps_rank_order_and_drops_missing_words():
    ranked = ["<unk>", "the", "of", "rare", "and", "cat"]
    first = {"<unk>": 0, "the": 1, "of": 2, "and": 3, "cat": 4}
    second = {"the": 0, "and": 1, "cat": 2, "rare": 3}
    assert shared_vocabulary(ranked, first, second) == ["the", "and", "cat"]
    assert shared_vocabulary(ranked, first, second, size=2) == ["the", "and"]
    with pytest.raises(ValueError):
        shared_vocabulary(ranked, first, size=0)


def test_restrict_space_reindexes_rows():
    vectors, vocabulary = _toy_space()
    matrix, restricted = restrict_space(vectors, vocabulary, ["queen", "man"])
    assert restricted == {"queen": 0, "man": 1}
    assert np.array_equal(matrix[0], vectors[vocabulary["queen"]])


def test_category_table_matches_direct_evaluation_and_aggregates():
    vectors, vocabulary = _toy_space()
    table = evaluate_by_category(vectors, vocabulary, RECORDS, semantic_categories={"family"})
    add = table[table["method"] == "3cosadd"].set_index("category")
    direct = evaluate_analogies(vectors, vocabulary, [r[1:] for r in RECORDS[:2]])
    assert add.loc["family", "accuracy"] == direct["accuracy"] == 1.0
    assert add.loc["gram1-adjective-to-adverb", "total"] == 2
    assert add.loc["gram1-adjective-to-adverb", "evaluated"] == 1
    assert add.loc["gram1-adjective-to-adverb", "coverage"] == 0.5
    assert add.loc["semantic", "total"] == 2 and add.loc["syntactic", "total"] == 2
    assert add.loc["total", "evaluated"] == add.loc["semantic", "evaluated"] + add.loc["syntactic", "evaluated"]
    assert set(table["method"]) == {"3cosadd", "3cosmul"}


def test_epoch_evaluation_reports_expected_keys_and_neighbors():
    vectors, vocabulary = _toy_space()
    pairs = [("man", "woman", 9.0), ("man", "princess", 1.0), ("boy", "girl", 8.0)]
    result = epoch_evaluation(vectors, vocabulary, pairs, RECORDS, control_words=("man", "zzz"), neighbors=3)
    assert set(result) == {"wordsim_spearman", "semantic_accuracy", "syntactic_accuracy", "total_accuracy", "neighbors"}
    assert list(result["neighbors"]) == ["man"] and len(result["neighbors"]["man"]) == 3


def test_parallel_differences_score_higher_than_random_pairs():
    vectors, vocabulary = _toy_space()
    table = difference_parallelism(vectors, vocabulary, RECORDS[:2], max_pairs=10, seed=0)
    row = table.iloc[0]
    assert row["category"] == "family" and row["pairs"] == 4
    assert row["mean_cosine"] > 0.95
    assert row["random_mean_cosine"] < row["mean_cosine"]


def test_thematic_selection_limits_group_size_and_never_repeats_words():
    groups = {"a": ("x", "y", "z", "w"), "b": ("y", "u", "v")}
    words, labels = select_thematic_words({"x", "y", "z", "u", "v"}, groups, per_group=2)
    assert words == ["x", "y", "u", "v"] and labels == ["a", "a", "b", "b"]
