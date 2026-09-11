"""Validated application configuration loading from YAML files."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigFileError(RuntimeError):
    """Raised when the project configuration file cannot be loaded or validated."""


class PathSettings(BaseModel):
    """Filesystem locations used by data, training, evaluation, and serving workflows."""

    model_config = ConfigDict(extra="forbid")

    data_root: Path = Field(default=Path("data/raw/chest_xray"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    reports_dir: Path = Field(default=Path("reports"))
    figures_dir: Path = Field(default=Path("reports/figures"))
    metrics_dir: Path = Field(default=Path("reports/metrics"))
    model_dir: Path = Field(default=Path("models"))
    best_model_path: Path = Field(default=Path("models/best_model.pt"))


class DataSettings(BaseModel):
    """Image and DataLoader parameters shared by model workflows."""

    model_config = ConfigDict(extra="forbid")

    image_size: int = Field(default=224, ge=32, le=2048)
    batch_size: int = Field(default=32, ge=1, le=1024)
    num_workers: int = Field(default=0, ge=0, le=32)
    pin_memory: bool = False
    use_stratified_validation_split: bool = True
    validation_split_fraction: float = Field(default=0.15, gt=0.0, lt=0.5)


class TrainingSettings(BaseModel):
    """Training hyperparameters and runtime behavior switches."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=5, ge=1, le=1000)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    use_weighted_loss: bool = True
    use_weighted_sampler: bool = False
    fine_tune_epochs: int = Field(default=1, ge=0, le=1000)
    fine_tune_learning_rate: float = Field(default=1e-5, gt=0.0, le=1.0)
    early_stopping_patience: int = Field(default=3, ge=1, le=1000)


class ModelSettings(BaseModel):
    """Model architecture selection and transfer-learning options."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["simple_cnn", "resnet18", "efficientnet_b0", "mobilenet_v3_small"] = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True
    num_classes: int = Field(default=2, ge=2)


class ProjectConfig(BaseModel):
    """Top-level project configuration loaded from `configs/config.yaml`."""

    model_config = ConfigDict(extra="forbid")

    paths: PathSettings = Field(default_factory=PathSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)

    @field_validator("paths")
    @classmethod
    def validate_output_paths(cls, paths: PathSettings) -> PathSettings:
        """Validate that configured output directories are distinct from raw data."""
        output_directories = {
            paths.processed_data_dir,
            paths.reports_dir,
            paths.figures_dir,
            paths.metrics_dir,
            paths.model_dir,
        }
        if paths.data_root in output_directories:
            message = "Raw data directory must not be reused as an output directory."
            raise ValueError(message)
        return paths

    def ensure_output_directories(self) -> None:
        """Create output directories required by project scripts.

        Raises:
            OSError: If any configured output directory cannot be created.
        """
        directories = (
            self.paths.processed_data_dir,
            self.paths.reports_dir,
            self.paths.figures_dir,
            self.paths.metrics_dir,
            self.paths.model_dir,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path = "configs/config.yaml") -> ProjectConfig:
    """Load and validate project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A validated `ProjectConfig` instance.

    Raises:
        ConfigFileError: If the file is missing, unreadable, malformed, or invalid.
    """
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_file():
        message = f"Config file not found: {resolved_config_path}"
        raise ConfigFileError(message)

    try:
        raw_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    except OSError as error:
        message = f"Could not read config file '{resolved_config_path}': {error}"
        raise ConfigFileError(message) from error
    except yaml.YAMLError as error:
        message = f"Config file '{resolved_config_path}' is not valid YAML: {error}"
        raise ConfigFileError(message) from error

    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        message = (
            f"Config file '{resolved_config_path}' must contain a YAML mapping at the top level."
        )
        raise ConfigFileError(message)

    try:
        return ProjectConfig.model_validate(raw_config)
    except ValidationError as error:
        message = f"Config file '{resolved_config_path}' failed validation: {error}"
        raise ConfigFileError(message) from error


def load_and_prepare_config(config_path: str | Path = "configs/config.yaml") -> ProjectConfig:
    """Load project configuration and create configured output directories.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A validated `ProjectConfig` with output directories created.

    Raises:
        ConfigFileError: If configuration loading or validation fails.
        OSError: If required output directories cannot be created.
    """
    config = load_config(config_path)
    config.ensure_output_directories()
    return config
