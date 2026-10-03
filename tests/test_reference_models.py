import numpy as np
import pytest
from gensim.models import KeyedVectors

from embeddings_lab.reference_models import (
    glove_embedding_table,
    load_encoded_sentences,
    token_total,
    train_gensim_skipgram,
)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("the king and the queen\nthe man and a rare word\n" * 30, encoding="utf-8")
    vocabulary = {"<unk>": 0, "the": 1, "and": 2, "king": 3, "queen": 4, "man": 5}
    return path, vocabulary


def test_sentences_use_unknown_token_and_respect_token_limit(corpus):
    path, vocabulary = corpus
    sentences = load_encoded_sentences(path, vocabulary)
    assert sentences[1] == ["the", "man", "and", "<unk>", "<unk>", "<unk>"]
    assert token_total(sentences) == 330
    limited = load_encoded_sentences(path, vocabulary, max_tokens=7)
    assert token_total(limited) == 7 and len(limited) == 2


def test_gensim_training_is_reproducible_and_records_each_epoch(corpus):
    path, vocabulary = corpus
    sentences = load_encoded_sentences(path, vocabulary)
    calls = []
    first = train_gensim_skipgram(
        sentences, dimension=8, min_count=1, sample=0.0, epochs=3, seed=1,
        epoch_evaluator=lambda epoch, vectors: calls.append(epoch) or {"vocabulary": len(vectors)},
    )
    second = train_gensim_skipgram(sentences, dimension=8, min_count=1, sample=0.0, epochs=3, seed=1)
    assert calls == [1, 2, 3]
    assert [r["epoch"] for r in first["epoch_records"]] == [1, 2, 3]
    assert first["epoch_records"][0]["vocabulary"] == 6
    assert first["tokens"] == 330 and first["total_time_seconds"] > 0
    assert np.array_equal(first["model"].wv.vectors, second["model"].wv.vectors)


def test_glove_table_keeps_pretrained_rows_unchanged_and_prepends_unknown():
    keyed = KeyedVectors(vector_size=3)
    keyed.add_vectors(["a", "b"], np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.float32))
    original = keyed.vectors.copy()
    vocabulary, table = glove_embedding_table(keyed)
    assert vocabulary == {"<unk>": 0, "a": 1, "b": 2}
    assert np.array_equal(table[1:], original) and np.array_equal(keyed.vectors, original)
    assert np.allclose(table[0], [2, 2, 2])


def test_glove_table_rejects_a_vocabulary_that_already_has_the_unknown_token():
    keyed = KeyedVectors(vector_size=2)
    keyed.add_vectors(["<unk>"], np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        glove_embedding_table(keyed)
