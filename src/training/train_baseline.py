"""CLI wrapper for quick baseline CNN training."""

from __future__ import annotations

import argparse

from src.core.config import load_and_prepare_config
from src.training.train import train_model


def parse_args() -> argparse.Namespace:
    """Parse baseline training command line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Train the SimpleCNN baseline model.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the SimpleCNN baseline using the project configuration."""
    args = parse_args()
    config = load_and_prepare_config(args.config)
    result = train_model(
        config,
        model_name="simple_cnn",
        checkpoint_filename="baseline_cnn.pt",
        history_filename="baseline_history.json",
        figure_filename="baseline_training_curves.png",
    )
    print(
        "Baseline CNN ready. "
        f"best_epoch={result.best_epoch}, "
        f"validation_accuracy={result.best_metrics.validation_accuracy:.4f}, "
        f"validation_precision={result.best_metrics.validation_precision:.4f}, "
        f"validation_recall={result.best_metrics.validation_recall:.4f}, "
        f"validation_f1={result.best_metrics.validation_f1:.4f}"
    )


if __name__ == "__main__":
    main()
