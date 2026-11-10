import json

import pytest

from mini_llm.config import Config


def test_load_reads_values_and_keeps_defaults(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"model": "bigram", "batch_size": 4}), encoding="utf-8")
    config = Config.load(path)
    assert config.batch_size == 4
    assert config.block_size == Config().block_size


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        Config.from_dict({"batch_szie": 4})


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        Config(model="lstm")


def test_round_trip_through_dict() -> None:
    config = Config(model="gpt", n_layer=2)
    assert Config.from_dict(config.to_dict()) == config


def test_shipped_configs_are_valid() -> None:
    from pathlib import Path

    configs = sorted(Path(__file__).resolve().parent.parent.glob("configs/*.json"))
    assert configs, "no config found in configs/"
    for path in configs:
        Config.load(path)
