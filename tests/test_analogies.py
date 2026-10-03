import numpy as np
import pytest

import embeddings_lab.analogies as analogy_module
from embeddings_lab.analogies import (
    analogy_3cosadd,
    analogy_3cosmul,
    answer_rank,
    evaluate_analogies,
    most_similar,
    tsne_projection,
)


@pytest.fixture
def space():
    words = {"man": 0, "woman": 1, "king": 2, "queen": 3, "other": 4}
    vectors = np.array([[1, 0], [1, 1], [2, 0], [2, 1], [-1, -1]], dtype=float)
    return vectors, words


def test_neighbors_use_cosine_and_exclude_the_query(space):
    vectors, words = space
    result = most_similar(vectors, words, "woman", top_k=2)
    assert [word for word, _ in result] == ["queen", "man"]
    assert result[0][1] == pytest.approx(3 / np.sqrt(10))


def test_neighbors_accept_vectors_and_explicit_exclusions(space):
    vectors, words = space
    result = most_similar(vectors, words, [1, 0], top_k=20, exclude={"man"})
    assert [word for word, _ in result] == ["king", "queen", "woman", "other"]


@pytest.mark.parametrize("method", [analogy_3cosadd, analogy_3cosmul])
def test_analogy_recovers_queen_with_input_terms_excluded(space, method):
    vectors, words = space
    result = method(vectors, words, "man", "woman", "king", top_k=1)
    assert result[0][0] == "queen"
    assert np.isfinite(result[0][1])


def test_analogy_can_keep_input_terms(space):
    vectors, words = space
    result = analogy_3cosadd(
        vectors, words, "man", "woman", "king", top_k=5, exclude_inputs=False
    )
    assert {word for word, _ in result} == set(words)


@pytest.mark.parametrize(
    ("method", "scores"),
    [
        (analogy_3cosadd, [0.9486832980505138, 0.8944271909999159]),
        (analogy_3cosmul, [4.97483976382213, 1.999996000008]),
    ],
)
def test_analogy_formula_differs_from_direct_neighbors_with_literal_scores(method, scores):
    vocabulary = {"a": 0, "b": 1, "c": 2, "answer": 3, "neighbor": 4}
    vectors = np.array([[1., 0.], [0., 1.], [0., 1.], [-1., 1.], [0., 1.]])
    direct = most_similar(vectors, vocabulary, "c", top_k=1, exclude={"a", "b"})
    assert direct == [("neighbor", 1.0)]
    result = method(vectors, vocabulary, "a", "b", "c", top_k=2)
    assert [word for word, _ in result] == ["answer", "neighbor"]
    assert [score for _, score in result] == pytest.approx(scores)
    excluded = method(vectors, vocabulary, "a", "b", "c", top_k=2, exclude={"answer"})
    assert [word for word, _ in excluded] == ["neighbor"]
    assert excluded[0][1] == pytest.approx(scores[1])


def test_3cosmul_stabilizes_a_zero_shifted_cosine_denominator():
    vocabulary = {"a": 0, "b": 1, "c": 2, "opposite": 3, "near_opposite": 4}
    vectors = np.array([[1., 0.], [0., 1.], [0., 1.], [-1., 0.], [-1., 0.000002]])
    result = analogy_3cosmul(vectors, vocabulary, "a", "b", "c", top_k=2)
    assert [word for word, _ in result] == ["near_opposite", "opposite"]
    assert [score for _, score in result] == pytest.approx([250000.75000528045, 250000.0])
    assert all(np.isfinite(score) for _, score in result)


@pytest.mark.parametrize("method", ["3cosadd", "3cosmul"])
def test_evaluation_reports_a_rank_greater_than_one(method):
    vocabulary = {"a": 0, "b": 1, "c": 2, "answer": 3, "neighbor": 4}
    vectors = np.array([[1., 0.], [0., 1.], [0., 1.], [-1., 1.], [0., 1.]])
    metrics = evaluate_analogies(vectors, vocabulary, [("a", "b", "c", "neighbor")], method=method)
    assert metrics["accuracy"] == 0.0
    assert metrics["mean_rank"] == 2.0
    assert metrics["mrr"] == 0.5


@pytest.mark.parametrize("top_k", [0, 1, 2, 4, 9])
def test_top_k_preserves_vocabulary_order_across_ties_and_exclusions(top_k):
    vocabulary = {"first": 4, "excluded": 0, "best": 3, "second": 2, "third": 1}
    vectors = np.array([[0., 1.], [0.6, 0.8], [0.6, 0.8], [1., 0.], [0.6, 0.8]])
    result = most_similar(vectors, vocabulary, [1., 0.], top_k=top_k, exclude={"excluded"})
    assert result == [("best", 1.0), ("first", 0.6), ("second", 0.6), ("third", 0.6)][:top_k]


def test_top_k_only_sorts_the_selected_candidates(monkeypatch):
    vectors = np.tile([0.6, 0.8], (1000, 1))
    vocabulary = {f"word{index}": index for index in range(len(vectors))}
    original_argsort = np.argsort

    def selected_argsort(values, *args, **kwargs):
        assert np.asarray(values).size <= 3, "top-k must not sort the entire vocabulary"
        return original_argsort(values, *args, **kwargs)

    monkeypatch.setattr(np, "argsort", selected_argsort)
    result = most_similar(vectors, vocabulary, [1., 0.], top_k=3)
    assert [word for word, _ in result] == ["word0", "word1", "word2"]


@pytest.mark.parametrize("method", ["3cosadd", "3cosmul"])
def test_evaluation_reuses_candidate_matrix_and_computes_stable_ranks_without_sorting(monkeypatch, method):
    vectors = np.array([[1., 0.], [0., 1.], [0., 1.], [-1., 1.], [-1., 1.], [-1., 1.]])
    vocabulary = {"a": 0, "b": 1, "c": 2, "first": 5, "second": 3, "third": 4}
    original_space = analogy_module._space
    candidate_copies = []

    class TrackedMatrix(np.ndarray):
        def __getitem__(self, key):
            result = super().__getitem__(key)
            if isinstance(key, np.ndarray) and key.ndim == 1:
                candidate_copies.append(result)
            return result

    def tracked_space(*args):
        normalized, words, indices = original_space(*args)
        return normalized.view(TrackedMatrix), words, indices

    def reject_sort(*args, **kwargs):
        pytest.fail("rank evaluation must count candidates instead of sorting the vocabulary")

    monkeypatch.setattr(analogy_module, "_space", tracked_space)
    monkeypatch.setattr(np, "argsort", reject_sort)
    records = [("a", "b", "c", answer) for answer in ("first", "second", "third", "a")]
    metrics = evaluate_analogies(vectors, vocabulary, records, method=method)
    assert metrics["accuracy"] == 0.25
    assert metrics["mean_rank"] == 2.0
    assert metrics["mrr"] == pytest.approx(11 / 24)
    assert len(candidate_copies) == 1


def test_answer_rank_is_one_based_and_missing_answers_have_no_rank():
    candidates = [("queen", 1.0), ("woman", 0.8)]
    assert answer_rank(candidates, "queen") == 1
    assert answer_rank(candidates, "woman") == 2
    assert answer_rank(candidates, "absent") is None


@pytest.mark.parametrize("method", ["3cosadd", "3cosmul"])
def test_evaluation_reports_coverage_without_counting_oov_as_wrong(space, method):
    vectors, words = space
    records = [("man", "woman", "king", "queen"), ("missing", "woman", "king", "queen")]
    metrics = evaluate_analogies(vectors, words, records, method=method)
    assert metrics["total"] == 2
    assert metrics["evaluated"] == 1
    assert metrics["coverage"] == 0.5
    assert metrics["accuracy"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["mean_rank"] == 1.0


def test_empty_evaluation_produces_finite_metrics(space):
    vectors, words = space
    metrics = evaluate_analogies(vectors, words, [])
    assert metrics == {
        "total": 0, "evaluated": 0, "coverage": 0.0,
        "accuracy": 0.0, "mrr": 0.0, "mean_rank": 0.0,
    }


def test_zero_vectors_have_finite_similarity():
    result = most_similar(np.array([[0., 0.], [1., 0.]]), {"empty": 0, "word": 1}, [0, 0])
    assert [score for _, score in result] == [0.0, 0.0]


def test_invalid_search_inputs_fail_explicitly(space):
    vectors, words = space
    with pytest.raises(KeyError):
        most_similar(vectors, words, "missing")
    with pytest.raises(ValueError):
        most_similar(vectors, words, "man", top_k=-1)
    with pytest.raises(ValueError):
        most_similar(vectors, {"man": 99}, "man")
    with pytest.raises(ValueError):
        most_similar(np.array([[np.nan, 0]]), {"word": 0}, "word")
    with pytest.raises(ValueError):
        evaluate_analogies(vectors, words, [], method="unknown")


def test_tsne_returns_finite_reproducible_two_dimensional_coordinates(space):
    vectors, _ = space
    first = tsne_projection(vectors, perplexity=2, seed=7, max_iter=250)
    second = tsne_projection(vectors, perplexity=2, seed=7, max_iter=250)
    assert first.shape == (5, 2)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


def test_tsne_rejects_too_few_rows():
    with pytest.raises(ValueError):
        tsne_projection(np.ones((1, 2)))
