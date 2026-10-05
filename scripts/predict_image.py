"""Command line entry point for single-image pneumonia inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.core.config import ConfigFileError, load_config
from src.inference.predict import InferenceError, load_model, predict_image, preprocess_image


def parse_args() -> argparse.Namespace:
    """Parse single-image prediction arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict the class of a single chest X-ray using a trained model."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the .jpeg, .jpg, or .png image to predict.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Checkpoint path. Defaults to the path in configs/config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    """Run single-image prediction and print the result as JSON.

    Returns:
        Process exit code. `0` indicates success; `1` indicates a user-facing failure.
    """
    args = parse_args()
    try:
        model_path = Path(args.model) if args.model else load_config().paths.best_model_path
        loaded_model = load_model(model_path)
        image_tensor = preprocess_image(
            Path(args.image),
            image_size=loaded_model.image_size,
            normalization_mean=loaded_model.normalization_mean,
            normalization_std=loaded_model.normalization_std,
            preprocessing=loaded_model.preprocessing,
        )
        prediction = predict_image(loaded_model, image_tensor)
    except (FileNotFoundError, InferenceError, ConfigFileError) as error:
        print(f"Could not generate a prediction: {error}", file=sys.stderr)
        return 1

    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
