"""Generate script-based exploratory data analysis reports for the X-ray dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.core.config import ConfigFileError, ProjectConfig, load_and_prepare_config
from src.data.validate_dataset import (
    EXPECTED_CLASSES,
    EXPECTED_SPLITS,
    DatasetValidationError,
    collect_image_paths,
)

matplotlib.use("Agg")

LABEL_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}
HISTOGRAM_BINS = 256
SAMPLE_GRID_SIZE = 9

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ImageEdaRecord:
    """EDA measurements for one readable image."""

    filepath: str
    split: str
    label: str
    label_id: int
    width: int
    height: int
    aspect_ratio: float
    mean_intensity: float
    std_intensity: float


@dataclass(frozen=True)
class CorruptedImageRecord:
    """Unreadable image discovered during EDA."""

    filepath: str
    split: str
    label: str
    error: str


@dataclass
class EdaComputation:
    """In-memory numerical EDA outputs used by plotting and report writing."""

    records: list[ImageEdaRecord]
    corrupted_images: list[CorruptedImageRecord]
    pixel_histogram: np.ndarray
    average_pixel_maps: dict[str, np.ndarray]
    sample_images: dict[str, list[Path]]


class EdaError(RuntimeError):
    """Raised when EDA cannot be completed."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Veri seti için EDA grafik ve rapor üretir.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="YAML config dosyası yolu.",
    )
    return parser.parse_args()


def read_grayscale_image(filepath: Path) -> tuple[Image.Image, np.ndarray]:
    """Read an image as grayscale and return the Pillow image and pixel array.

    Args:
        filepath: Image file path.

    Returns:
        Grayscale Pillow image and a two-dimensional uint8 pixel array.

    Raises:
        OSError: If the image cannot be read or decoded.
        UnidentifiedImageError: If Pillow cannot identify the image format.
    """
    with Image.open(filepath) as image:
        grayscale_image = image.convert("L")
        pixel_array = np.asarray(grayscale_image, dtype=np.uint8)
        return grayscale_image.copy(), pixel_array


def resize_for_average(image: Image.Image, image_size: int) -> np.ndarray:
    """Resize a grayscale image to the configured model size for average maps.

    Args:
        image: Grayscale Pillow image.
        image_size: Target square size in pixels.

    Returns:
        Float32 image array with values in the 0-255 range.
    """
    resized_image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(resized_image, dtype=np.float32)


def collect_eda_measurements(config: ProjectConfig) -> EdaComputation:
    """Collect image measurements, pixel histogram, average maps, and sample paths.

    Args:
        config: Loaded project configuration.

    Returns:
        EDA computation output for figures, JSON, and Markdown reports.

    Raises:
        DatasetValidationError: If required dataset directories are missing.
        EdaError: If no readable image is found.
    """
    image_paths = collect_image_paths(config.paths.data_root)
    records: list[ImageEdaRecord] = []
    corrupted_images: list[CorruptedImageRecord] = []
    pixel_histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    average_sums = {
        label: np.zeros((config.data.image_size, config.data.image_size), dtype=np.float64)
        for label in EXPECTED_CLASSES
    }
    average_counts = dict.fromkeys(EXPECTED_CLASSES, 0)
    sample_images: dict[str, list[Path]] = {label: [] for label in EXPECTED_CLASSES}

    for split in EXPECTED_SPLITS:
        for label in EXPECTED_CLASSES:
            for filepath in image_paths[split][label]:
                try:
                    grayscale_image, pixel_array = read_grayscale_image(filepath)
                except (OSError, UnidentifiedImageError) as error:
                    corrupted_images.append(
                        CorruptedImageRecord(
                            filepath=str(filepath),
                            split=split,
                            label=label,
                            error=str(error),
                        )
                    )
                    continue

                height, width = pixel_array.shape
                if width <= 0 or height <= 0:
                    corrupted_images.append(
                        CorruptedImageRecord(
                            filepath=str(filepath),
                            split=split,
                            label=label,
                            error="Image has invalid dimensions.",
                        )
                    )
                    continue

                records.append(
                    ImageEdaRecord(
                        filepath=str(filepath),
                        split=split,
                        label=label,
                        label_id=LABEL_TO_ID[label],
                        width=width,
                        height=height,
                        aspect_ratio=width / height,
                        mean_intensity=float(np.mean(pixel_array)),
                        std_intensity=float(np.std(pixel_array)),
                    )
                )
                image_histogram, _ = np.histogram(
                    pixel_array,
                    bins=HISTOGRAM_BINS,
                    range=(0, 256),
                )
                pixel_histogram += image_histogram.astype(np.int64)
                average_sums[label] += resize_for_average(
                    grayscale_image,
                    config.data.image_size,
                )
                average_counts[label] += 1
                if len(sample_images[label]) < SAMPLE_GRID_SIZE:
                    sample_images[label].append(filepath)

    if not records:
        message = "EDA üretilemedi; veri setinde okunabilir görüntü bulunamadı."
        raise EdaError(message)

    average_pixel_maps: dict[str, np.ndarray] = {}
    for label in EXPECTED_CLASSES:
        if average_counts[label] == 0:
            average_pixel_maps[label] = np.zeros(
                (config.data.image_size, config.data.image_size),
                dtype=np.float64,
            )
            continue
        average_pixel_maps[label] = average_sums[label] / average_counts[label]

    return EdaComputation(
        records=records,
        corrupted_images=corrupted_images,
        pixel_histogram=pixel_histogram,
        average_pixel_maps=average_pixel_maps,
        sample_images=sample_images,
    )


def summarize_numeric_values(values: list[float]) -> dict[str, JsonValue]:
    """Summarize a list of numeric values with robust descriptive statistics.

    Args:
        values: Numeric values to summarize.

    Returns:
        JSON-serializable statistics dictionary.
    """
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    value_array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(value_array.size),
        "min": float(np.min(value_array)),
        "max": float(np.max(value_array)),
        "mean": float(np.mean(value_array)),
        "median": float(np.median(value_array)),
        "std": float(np.std(value_array)),
    }


def build_class_distribution(records: list[ImageEdaRecord]) -> dict[str, dict[str, int]]:
    """Build split-by-class counts from EDA records.

    Args:
        records: Readable image EDA records.

    Returns:
        Nested split and class count dictionary.
    """
    distribution = {
        split: dict.fromkeys(EXPECTED_CLASSES, 0) | {"total": 0} for split in EXPECTED_SPLITS
    }
    for record in records:
        distribution[record.split][record.label] += 1
        distribution[record.split]["total"] += 1
    return distribution


def build_imbalance_summary(
    class_distribution: dict[str, dict[str, int]],
) -> dict[str, dict[str, JsonValue]]:
    """Build numerical class imbalance summary for each split and overall.

    Args:
        class_distribution: Nested split and class count dictionary.

    Returns:
        Split-level imbalance metrics.
    """
    imbalance_summary: dict[str, dict[str, JsonValue]] = {}
    normal_total = 0
    pneumonia_total = 0
    for split in EXPECTED_SPLITS:
        normal_count = class_distribution[split]["NORMAL"]
        pneumonia_count = class_distribution[split]["PNEUMONIA"]
        total_count = normal_count + pneumonia_count
        normal_total += normal_count
        pneumonia_total += pneumonia_count
        imbalance_summary[split] = calculate_imbalance_metrics(normal_count, pneumonia_count)
        imbalance_summary[split]["total"] = total_count

    imbalance_summary["overall"] = calculate_imbalance_metrics(normal_total, pneumonia_total)
    imbalance_summary["overall"]["total"] = normal_total + pneumonia_total
    return imbalance_summary


def calculate_imbalance_metrics(normal_count: int, pneumonia_count: int) -> dict[str, JsonValue]:
    """Calculate class imbalance metrics from two class counts.

    Args:
        normal_count: Number of NORMAL images.
        pneumonia_count: Number of PNEUMONIA images.

    Returns:
        JSON-serializable imbalance metrics.
    """
    total_count = normal_count + pneumonia_count
    minority_count = min(normal_count, pneumonia_count)
    majority_count = max(normal_count, pneumonia_count)
    return {
        "NORMAL": normal_count,
        "PNEUMONIA": pneumonia_count,
        "normal_fraction": normal_count / total_count if total_count else None,
        "pneumonia_fraction": pneumonia_count / total_count if total_count else None,
        "majority_to_minority_ratio": majority_count / minority_count if minority_count else None,
    }


def build_dimension_summary(records: list[ImageEdaRecord]) -> dict[str, dict[str, JsonValue]]:
    """Build descriptive statistics for width, height, aspect ratio, and intensity.

    Args:
        records: Readable image EDA records.

    Returns:
        Dimension and intensity statistics grouped by split and class.
    """
    summary: dict[str, dict[str, JsonValue]] = {}
    for split in (*EXPECTED_SPLITS, "overall"):
        split_records = (
            records
            if split == "overall"
            else [record for record in records if record.split == split]
        )
        summary[split] = {
            "width": summarize_numeric_values([float(record.width) for record in split_records]),
            "height": summarize_numeric_values([float(record.height) for record in split_records]),
            "aspect_ratio": summarize_numeric_values(
                [record.aspect_ratio for record in split_records]
            ),
            "mean_intensity": summarize_numeric_values(
                [record.mean_intensity for record in split_records]
            ),
            "std_intensity": summarize_numeric_values(
                [record.std_intensity for record in split_records]
            ),
        }

    for label in EXPECTED_CLASSES:
        label_records = [record for record in records if record.label == label]
        summary[f"class_{label}"] = {
            "width": summarize_numeric_values([float(record.width) for record in label_records]),
            "height": summarize_numeric_values([float(record.height) for record in label_records]),
            "aspect_ratio": summarize_numeric_values(
                [record.aspect_ratio for record in label_records]
            ),
            "mean_intensity": summarize_numeric_values(
                [record.mean_intensity for record in label_records]
            ),
            "std_intensity": summarize_numeric_values(
                [record.std_intensity for record in label_records]
            ),
        }
    return summary


def build_observations(
    imbalance_summary: dict[str, dict[str, JsonValue]],
    dimension_summary: dict[str, dict[str, JsonValue]],
    corrupted_count: int,
) -> list[str]:
    """Create human-readable observations for the Markdown report.

    Args:
        imbalance_summary: Split and overall class imbalance metrics.
        dimension_summary: Dimension and intensity descriptive statistics.
        corrupted_count: Number of unreadable images.

    Returns:
        Observed issues and noteworthy facts.
    """
    observations: list[str] = []
    overall_ratio = imbalance_summary["overall"]["majority_to_minority_ratio"]
    if isinstance(overall_ratio, float) and overall_ratio >= 1.5:
        observations.append(
            "Sınıf dağılımı dengesiz görünüyor; eğitimde class weight veya sampler "
            "kullanılması önerilir."
        )
    else:
        observations.append("Genel sınıf dağılımı ağır bir dengesizlik göstermiyor.")

    aspect_ratio_std = dimension_summary["overall"]["aspect_ratio"]["std"]
    if isinstance(aspect_ratio_std, float) and aspect_ratio_std > 0.25:
        observations.append(
            "Aspect ratio dağılımı geniş; resize işleminde oran bozulmasının etkisi izlenmeli."
        )
    else:
        observations.append("Aspect ratio dağılımı görece kontrollü görünüyor.")

    if corrupted_count:
        observations.append(
            f"EDA sırasında {corrupted_count} okunamayan görüntü bulundu; eğitimden önce "
            "bu dosyalar incelenmelidir."
        )
    else:
        observations.append("EDA sırasında okunamayan görüntüyle karşılaşılmadı.")
    return observations


def build_recommendations(
    imbalance_summary: dict[str, dict[str, JsonValue]],
) -> list[str]:
    """Create modeling recommendations from EDA metrics.

    Args:
        imbalance_summary: Split and overall class imbalance metrics.

    Returns:
        Recommended modeling actions.
    """
    recommendations = [
        "Görüntüler sabit boyuta getirilirken eğitim ve inference aşamalarında aynı "
        "normalizasyon kullanılmalıdır.",
        "Tıbbi anlamı bozabilecek agresif augmentasyonlardan kaçınılmalıdır.",
        "Model seçiminde accuracy ile birlikte recall, precision, F1-score ve confusion matrix "
        "mutlaka raporlanmalıdır.",
    ]
    overall_ratio = imbalance_summary["overall"]["majority_to_minority_ratio"]
    if isinstance(overall_ratio, float) and overall_ratio >= 1.5:
        recommendations.insert(
            0,
            "PNEUMONIA ve NORMAL sınıfları arasındaki dengesizlik için weighted loss "
            "veya WeightedRandomSampler denenmelidir.",
        )
    return recommendations


def build_eda_summary(computation: EdaComputation) -> dict[str, JsonValue]:
    """Build JSON-serializable EDA summary.

    Args:
        computation: EDA computation output.

    Returns:
        JSON-compatible summary payload.
    """
    class_distribution = build_class_distribution(computation.records)
    imbalance_summary = build_imbalance_summary(class_distribution)
    dimension_summary = build_dimension_summary(computation.records)
    observations = build_observations(
        imbalance_summary,
        dimension_summary,
        len(computation.corrupted_images),
    )
    recommendations = build_recommendations(imbalance_summary)
    return {
        "total_readable_images": len(computation.records),
        "corrupted_image_count": len(computation.corrupted_images),
        "corrupted_images": [asdict(record) for record in computation.corrupted_images],
        "class_distribution": class_distribution,
        "imbalance_summary": imbalance_summary,
        "dimension_summary": dimension_summary,
        "pixel_intensity_histogram": {
            "bins": list(range(HISTOGRAM_BINS)),
            "counts": [int(value) for value in computation.pixel_histogram.tolist()],
        },
        "observations": observations,
        "recommendations": recommendations,
    }


def save_json(filepath: Path, payload: dict[str, JsonValue]) -> None:
    """Write a JSON payload using UTF-8 encoding.

    Args:
        filepath: Output file path.
        payload: JSON-serializable payload.

    Raises:
        OSError: If the file cannot be written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def records_to_arrays(records: list[ImageEdaRecord]) -> dict[str, np.ndarray]:
    """Convert record fields to arrays for plotting.

    Args:
        records: Readable image EDA records.

    Returns:
        Mapping of numeric field names to NumPy arrays.
    """
    return {
        "width": np.asarray([record.width for record in records], dtype=np.float64),
        "height": np.asarray([record.height for record in records], dtype=np.float64),
        "aspect_ratio": np.asarray([record.aspect_ratio for record in records], dtype=np.float64),
        "mean_intensity": np.asarray(
            [record.mean_intensity for record in records],
            dtype=np.float64,
        ),
    }


def save_class_distribution_plot(
    records: list[ImageEdaRecord],
    output_path: Path,
) -> None:
    """Save train/validation/test class distribution chart.

    Args:
        records: Readable image EDA records.
        output_path: Figure output path.

    Raises:
        OSError: If the figure cannot be written.
    """
    distribution = build_class_distribution(records)
    x_positions = np.arange(len(EXPECTED_SPLITS))
    bar_width = 0.35

    plt.figure(figsize=(9, 5))
    for label_index, label in enumerate(EXPECTED_CLASSES):
        counts = [distribution[split][label] for split in EXPECTED_SPLITS]
        offset = (label_index - 0.5) * bar_width
        plt.bar(x_positions + offset, counts, width=bar_width, label=label)

    plt.xticks(x_positions, EXPECTED_SPLITS)
    plt.xlabel("Split")
    plt.ylabel("Görüntü sayısı")
    plt.title("Train/Val/Test Sınıf Dağılımı")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_sample_grid(label: str, sample_paths: list[Path], output_path: Path) -> None:
    """Save a grid of example images for one class.

    Args:
        label: Class label for the grid title.
        sample_paths: Image paths to include.
        output_path: Figure output path.

    Raises:
        OSError: If the figure cannot be written.
    """
    if not sample_paths:
        return

    columns = 3
    rows = int(np.ceil(len(sample_paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9, 3 * rows))
    axes_array = np.asarray(axes).reshape(-1)

    for axis in axes_array:
        axis.axis("off")

    for index, filepath in enumerate(sample_paths):
        grayscale_image, _ = read_grayscale_image(filepath)
        axes_array[index].imshow(grayscale_image, cmap="gray")
        axes_array[index].set_title(filepath.name, fontsize=8)
        axes_array[index].axis("off")

    fig.suptitle(f"{label} Örnek Görüntüleri", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def save_image_size_distribution(records: list[ImageEdaRecord], output_path: Path) -> None:
    """Save width and height distribution plots.

    Args:
        records: Readable image EDA records.
        output_path: Figure output path.

    Raises:
        OSError: If the figure cannot be written.
    """
    arrays = records_to_arrays(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(arrays["width"], arrays["height"], alpha=0.35, s=12)
    axes[0].set_xlabel("Genişlik")
    axes[0].set_ylabel("Yükseklik")
    axes[0].set_title("Genişlik/Yükseklik Dağılımı")

    axes[1].hist(arrays["width"], bins=40, alpha=0.7, label="Genişlik")
    axes[1].hist(arrays["height"], bins=40, alpha=0.7, label="Yükseklik")
    axes[1].set_xlabel("Piksel")
    axes[1].set_ylabel("Görüntü sayısı")
    axes[1].set_title("Boyut Histogramları")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def save_aspect_ratio_distribution(records: list[ImageEdaRecord], output_path: Path) -> None:
    """Save aspect ratio histogram.

    Args:
        records: Readable image EDA records.
        output_path: Figure output path.

    Raises:
        OSError: If the figure cannot be written.
    """
    arrays = records_to_arrays(records)
    plt.figure(figsize=(9, 5))
    plt.hist(arrays["aspect_ratio"], bins=50, color="#4C78A8", alpha=0.85)
    plt.xlabel("Aspect ratio (genişlik / yükseklik)")
    plt.ylabel("Görüntü sayısı")
    plt.title("Aspect Ratio Dağılımı")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_pixel_intensity_histogram(pixel_histogram: np.ndarray, output_path: Path) -> None:
    """Save aggregate grayscale pixel intensity histogram.

    Args:
        pixel_histogram: Histogram counts for intensity values 0-255.
        output_path: Figure output path.

    Raises:
        OSError: If the figure cannot be written.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(HISTOGRAM_BINS), pixel_histogram, color="#2F855A", linewidth=1.5)
    plt.xlabel("Piksel yoğunluğu")
    plt.ylabel("Piksel sayısı")
    plt.title("Toplam Piksel Yoğunluğu Histogramı")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_average_pixel_maps(
    average_pixel_maps: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save per-class average grayscale pixel maps.

    Args:
        average_pixel_maps: Class labels mapped to average image arrays.
        output_path: Figure output path.

    Raises:
        OSError: If the figure cannot be written.
    """
    fig, axes = plt.subplots(1, len(EXPECTED_CLASSES), figsize=(10, 5))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, label in zip(axes_array, EXPECTED_CLASSES, strict=True):
        image = axis.imshow(average_pixel_maps[label], cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"{label} Ortalama Piksel Haritası")
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def save_figures(config: ProjectConfig, computation: EdaComputation) -> list[Path]:
    """Save all EDA figures.

    Args:
        config: Loaded project configuration.
        computation: EDA computation output.

    Returns:
        Paths of generated figures.

    Raises:
        OSError: If any figure cannot be written.
    """
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        config.paths.figures_dir / "eda_class_distribution.png",
        config.paths.figures_dir / "eda_sample_grid_NORMAL.png",
        config.paths.figures_dir / "eda_sample_grid_PNEUMONIA.png",
        config.paths.figures_dir / "eda_image_size_distribution.png",
        config.paths.figures_dir / "eda_aspect_ratio_distribution.png",
        config.paths.figures_dir / "eda_pixel_intensity_histogram.png",
        config.paths.figures_dir / "eda_average_pixel_maps.png",
    ]
    save_class_distribution_plot(computation.records, figure_paths[0])
    save_sample_grid("NORMAL", computation.sample_images["NORMAL"], figure_paths[1])
    save_sample_grid("PNEUMONIA", computation.sample_images["PNEUMONIA"], figure_paths[2])
    save_image_size_distribution(computation.records, figure_paths[3])
    save_aspect_ratio_distribution(computation.records, figure_paths[4])
    save_pixel_intensity_histogram(computation.pixel_histogram, figure_paths[5])
    save_average_pixel_maps(computation.average_pixel_maps, figure_paths[6])
    return [path for path in figure_paths if path.is_file()]


def format_optional_number(value: JsonValue, precision: int = 3) -> str:
    """Format a JSON value as a human-readable number.

    Args:
        value: JSON value that may contain a numeric value.
        precision: Number of decimal places for floats.

    Returns:
        Formatted value or `yok` for missing values.
    """
    if value is None:
        return "yok"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def build_markdown_report(summary: dict[str, JsonValue], figure_paths: list[Path]) -> str:
    """Build the Turkish EDA Markdown summary.

    Args:
        summary: EDA summary payload.
        figure_paths: Generated EDA figure paths.

    Returns:
        Markdown report content.
    """
    class_distribution = summary["class_distribution"]
    imbalance_summary = summary["imbalance_summary"]
    dimension_summary = summary["dimension_summary"]
    if not isinstance(class_distribution, dict):
        raise EdaError("EDA özeti class_distribution alanı geçersiz.")
    if not isinstance(imbalance_summary, dict):
        raise EdaError("EDA özeti imbalance_summary alanı geçersiz.")
    if not isinstance(dimension_summary, dict):
        raise EdaError("EDA özeti dimension_summary alanı geçersiz.")

    lines = [
        "# EDA Özeti",
        "",
        "Bu doküman `python -m src.eda.run_eda --config configs/config.yaml` komutu ile "
        "otomatik oluşturulmuştur.",
        "",
        "## Veri Seti Özeti",
        "",
        f"- Okunabilir görüntü sayısı: {summary['total_readable_images']}",
        f"- Okunamayan görüntü sayısı: {summary['corrupted_image_count']}",
        "",
        "## Sınıf Dağılımı",
        "",
        "| Split | NORMAL | PNEUMONIA | Toplam |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in EXPECTED_SPLITS:
        split_distribution = class_distribution[split]
        if not isinstance(split_distribution, dict):
            raise EdaError(f"EDA özeti {split} dağılımı geçersiz.")
        lines.append(
            f"| {split} | {split_distribution['NORMAL']} | "
            f"{split_distribution['PNEUMONIA']} | {split_distribution['total']} |"
        )

    overall_imbalance = imbalance_summary["overall"]
    overall_dimensions = dimension_summary["overall"]
    if not isinstance(overall_imbalance, dict):
        raise EdaError("EDA özeti overall imbalance alanı geçersiz.")
    if not isinstance(overall_dimensions, dict):
        raise EdaError("EDA özeti overall dimension alanı geçersiz.")

    lines.extend(
        [
            "",
            "## Sayısal Özet",
            "",
            "- Çoğunluk/azınlık sınıf oranı: "
            f"{format_optional_number(overall_imbalance['majority_to_minority_ratio'])}",
            "- Ortalama genişlik: "
            f"{format_nested_stat(overall_dimensions, 'width', 'mean')} piksel",
            "- Ortalama yükseklik: "
            f"{format_nested_stat(overall_dimensions, 'height', 'mean')} piksel",
            "- Ortalama aspect ratio: "
            f"{format_nested_stat(overall_dimensions, 'aspect_ratio', 'mean')}",
            "- Ortalama piksel yoğunluğu: "
            f"{format_nested_stat(overall_dimensions, 'mean_intensity', 'mean')}",
            "",
            "## Gözlenen Olası Problemler",
            "",
        ]
    )
    observations = summary["observations"]
    if isinstance(observations, list):
        lines.extend(f"- {observation}" for observation in observations)

    lines.extend(["", "## Modelleme İçin Öneriler", ""])
    recommendations = summary["recommendations"]
    if isinstance(recommendations, list):
        lines.extend(f"- {recommendation}" for recommendation in recommendations)

    lines.extend(["", "## Üretilen Görseller", ""])
    lines.extend(f"- `{path.as_posix()}`" for path in figure_paths)
    lines.append("")
    return "\n".join(lines)


def format_nested_stat(
    summary: dict[str, JsonValue],
    group_name: str,
    stat_name: str,
) -> str:
    """Format one nested statistic from the EDA summary.

    Args:
        summary: Grouped summary dictionary.
        group_name: Parent group name.
        stat_name: Statistic name under the parent group.

    Returns:
        Formatted statistic.
    """
    group = summary[group_name]
    if not isinstance(group, dict):
        return "yok"
    return format_optional_number(group[stat_name])


def save_markdown_report(
    docs_dir: Path,
    summary: dict[str, JsonValue],
    figure_paths: list[Path],
) -> Path:
    """Write the Markdown EDA report.

    Args:
        docs_dir: Documentation directory.
        summary: EDA summary payload.
        figure_paths: Generated figure paths.

    Returns:
        Markdown file path.

    Raises:
        OSError: If the file cannot be written.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_path = docs_dir / "eda_summary.md"
    output_path.write_text(
        build_markdown_report(summary, figure_paths),
        encoding="utf-8",
    )
    return output_path


def run_eda(config: ProjectConfig) -> tuple[dict[str, JsonValue], list[Path], Path]:
    """Run the full EDA workflow and write all requested outputs.

    Args:
        config: Loaded project configuration.

    Returns:
        EDA summary, generated figure paths, and Markdown report path.

    Raises:
        DatasetValidationError: If the dataset directory structure is invalid.
        EdaError: If EDA cannot be completed.
        OSError: If output files cannot be written.
    """
    computation = collect_eda_measurements(config)
    summary = build_eda_summary(computation)
    summary_path = config.paths.metrics_dir / "eda_summary.json"
    save_json(summary_path, summary)
    figure_paths = save_figures(config, computation)
    docs_dir = Path("docs")
    markdown_path = save_markdown_report(docs_dir, summary, figure_paths)
    return summary, figure_paths + [summary_path], markdown_path


def main() -> int:
    """Run the EDA command-line workflow.

    Returns:
        Process exit code.
    """
    args = parse_args()
    try:
        config = load_and_prepare_config(args.config)
        summary, output_paths, markdown_path = run_eda(config)
    except (ConfigFileError, DatasetValidationError, EdaError, OSError) as error:
        print(f"HATA: {error}")
        return 1

    print("EDA tamamlandı.")
    print(f"Okunabilir görüntü sayısı: {summary['total_readable_images']}")
    print(f"Okunamayan görüntü sayısı: {summary['corrupted_image_count']}")
    print("Üretilen dosyalar:")
    for output_path in output_paths:
        print(f"- {output_path}")
    print(f"- {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
