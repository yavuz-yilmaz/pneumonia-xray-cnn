from pathlib import Path

import pytest
from src.core.config import ConfigFileError, ProjectConfig, load_config


def test_load_config_reads_default_project_configuration() -> None:
    config = load_config("configs/config.yaml")

    assert isinstance(config, ProjectConfig)
    assert config.paths.data_root == Path("data/raw/chest_xray")
    assert config.data.image_size == 224
    assert config.data.batch_size == 24
    assert config.data.use_stratified_validation_split is True
    assert config.data.use_group_disjoint_split is True
    assert config.data.validation_split_fraction == 0.20
    assert config.paths.processed_data_dir == Path("data/processed/clean_v1")
    assert config.training.epochs >= 1
    assert config.training.learning_rate > 0.0
    assert config.model.num_classes == 2


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    missing_config_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigFileError, match="Config file not found"):
        load_config(missing_config_path)


def test_load_config_rejects_invalid_yaml_shape(tmp_path: Path) -> None:
    invalid_config_path = tmp_path / "invalid.yaml"
    invalid_config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigFileError, match="must contain a YAML mapping"):
        load_config(invalid_config_path)
