from pathlib import Path

import json
import pytest

from embeddings_lab.experiments import (
    ExperimentConfig,
    count_model_parameters,
    count_skipgram_pairs,
    oov_rate,
    save_run_metadata,
    seed_everything,
)


def test_default_configuration_matches_required_sweep() -> None:
    config = ExperimentConfig()

    assert config.corpus_token_limit == 20_000_000
    assert config.dimensions == (50, 100, 300)
    assert 100 in config.dimensions


def test_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="corpus_token_limit"):
        ExperimentConfig(corpus_token_limit=0)

    with pytest.raises(ValueError, match="dimensions"):
        ExperimentConfig(dimensions=())


def test_metadata_is_saved_as_json(tmp_path: Path) -> None:
    path = save_run_metadata(
        output_dir=tmp_path,
        run_name="smoke",
        metadata={"seed": 17, "tokens": 100},
    )

    assert path == tmp_path / "smoke" / "metadata.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "seed": 17,
        "tokens": 100,
    }


def test_seed_everything_repeats_numpy_and_torch_values() -> None:
    import numpy as np
    import torch

    seed_everything(9)
    first_numpy = np.random.random(3)
    first_torch = torch.rand(3)

    seed_everything(9)
    second_numpy = np.random.random(3)
    second_torch = torch.rand(3)

    assert np.array_equal(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)


def test_pair_count_matches_directed_symmetric_windows() -> None:
    assert count_skipgram_pairs([0, 1, 2, 3, 4], window_size=2) == 14
    assert count_skipgram_pairs([], window_size=2) == 0
    assert count_skipgram_pairs([1], window_size=2) == 0


def test_oov_rate_counts_token_occurrences() -> None:
    vocabulary = {"<unk>": 0, "known": 1}

    assert oov_rate([["known", "missing"], ["missing"]], vocabulary) == pytest.approx(2 / 3)
    assert oov_rate([], vocabulary) == 0.0


def test_parameter_counts_distinguish_frozen_weights() -> None:
    import torch

    model = torch.nn.Sequential(torch.nn.Embedding(4, 3), torch.nn.Linear(3, 2))
    model[0].weight.requires_grad_(False)

    assert count_model_parameters(model) == {"total": 20, "trainable": 8}
