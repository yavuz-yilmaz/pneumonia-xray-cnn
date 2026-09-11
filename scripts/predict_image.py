"""Command line entry point for single-image pneumonia inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.inference.predict import InferenceError, load_model, predict_image, preprocess_image


def parse_args() -> argparse.Namespace:
    """Parse single-image prediction arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Eğitilmiş model ile tek bir akciğer röntgeni görüntüsü için tahmin üret."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Tahmin yapılacak .jpeg, .jpg veya .png görüntü yolu.",
    )
    parser.add_argument(
        "--model",
        default="models/best_model.pt",
        help="Eğitilmiş model checkpoint yolu.",
    )
    return parser.parse_args()


def main() -> int:
    """Run single-image prediction and print the result as JSON.

    Returns:
        Process exit code. `0` indicates success; `1` indicates a user-facing failure.
    """
    args = parse_args()
    try:
        loaded_model = load_model(Path(args.model))
        image_tensor = preprocess_image(
            Path(args.image),
            image_size=loaded_model.image_size,
            normalization_mean=loaded_model.normalization_mean,
            normalization_std=loaded_model.normalization_std,
        )
        prediction = predict_image(loaded_model, image_tensor)
    except (FileNotFoundError, InferenceError) as error:
        print(f"Tahmin üretilemedi: {error}", file=sys.stderr)
        return 1

    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
