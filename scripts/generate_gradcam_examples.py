"""Generate Grad-CAM example figures for the trained pneumonia classifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, UnidentifiedImageError
from src.core.config import ConfigFileError, ProjectConfig, load_and_prepare_config
from src.data.dataloader import get_manifest_paths, require_manifest_files
from src.data.dataset import ID_TO_LABEL, LABEL_TO_ID, ChestXRayDataset
from src.data.transforms import build_eval_transforms
from src.evaluation.evaluate import load_checkpoint_model
from src.explainability.gradcam import GradCam, resolve_target_layer

MAX_CORRECT_EXAMPLES_PER_CLASS = 4
MAX_MISCLASSIFIED_EXAMPLES = 6
MEDICAL_WARNING = "Bu proje eğitim amaçlıdır; çıktılar tıbbi teşhis amacıyla kullanılmamalıdır."


@dataclass(frozen=True)
class GradCamCandidate:
    """Candidate sample selected for Grad-CAM visualization.

    Attributes:
        dataset_index: Sample index in the test dataset.
        true_label_id: Ground-truth numeric label.
        predicted_label_id: Predicted numeric label.
        pneumonia_probability: Softmax probability for the PNEUMONIA class.
    """

    dataset_index: int
    true_label_id: int
    predicted_label_id: int
    pneumonia_probability: float


@dataclass(frozen=True)
class GradCamOutputs:
    """Grad-CAM generation output paths and counts.

    Attributes:
        figure_paths: Generated figure paths.
        summary_path: Markdown summary path.
        correct_normal_count: Number of correct NORMAL examples visualized.
        correct_pneumonia_count: Number of correct PNEUMONIA examples visualized.
        misclassified_count: Number of misclassified examples visualized.
    """

    figure_paths: list[Path]
    summary_path: Path
    correct_normal_count: int
    correct_pneumonia_count: int
    misclassified_count: int


def resolve_gradcam_device(config: ProjectConfig) -> torch.device:
    """Resolve a runtime device and fall back to CPU when the requested accelerator is unavailable.

    Args:
        config: Loaded project configuration.

    Returns:
        PyTorch device for Grad-CAM inference.
    """
    requested_device = config.training.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Config device='cuda' seçilmiş ancak CUDA yok; Grad-CAM CPU üzerinde çalıştırılıyor.")
        return torch.device("cpu")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        print("Config device='mps' seçilmiş ancak MPS yok; Grad-CAM CPU üzerinde çalıştırılıyor.")
        return torch.device("cpu")
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)


def collect_gradcam_candidates(
    *,
    model: torch.nn.Module,
    dataset: ChestXRayDataset,
    device: torch.device,
) -> list[GradCamCandidate]:
    """Run test-set inference and collect samples eligible for Grad-CAM.

    Args:
        model: Loaded classifier.
        dataset: Test dataset with deterministic transforms.
        device: Runtime device.

    Returns:
        Prediction metadata for each test sample.

    Raises:
        XRayDatasetError: If the dataset cannot produce tensors.
    """
    candidates: list[GradCamCandidate] = []
    model.eval()
    with torch.no_grad():
        for dataset_index in range(len(dataset)):
            image_tensor, label_id = dataset[dataset_index]
            logits = model(image_tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
            predicted_label_id = int(torch.argmax(probabilities).item())
            candidates.append(
                GradCamCandidate(
                    dataset_index=dataset_index,
                    true_label_id=int(label_id),
                    predicted_label_id=predicted_label_id,
                    pneumonia_probability=float(probabilities[LABEL_TO_ID["PNEUMONIA"]].item()),
                )
            )
    return candidates


def select_examples(candidates: list[GradCamCandidate]) -> dict[str, list[GradCamCandidate]]:
    """Select representative correct and incorrect Grad-CAM examples.

    Args:
        candidates: Prediction metadata for the test dataset.

    Returns:
        Named groups of selected candidates.
    """
    correct_normal = [
        candidate
        for candidate in candidates
        if candidate.true_label_id == LABEL_TO_ID["NORMAL"]
        and candidate.predicted_label_id == LABEL_TO_ID["NORMAL"]
    ][:MAX_CORRECT_EXAMPLES_PER_CLASS]
    correct_pneumonia = [
        candidate
        for candidate in candidates
        if candidate.true_label_id == LABEL_TO_ID["PNEUMONIA"]
        and candidate.predicted_label_id == LABEL_TO_ID["PNEUMONIA"]
    ][:MAX_CORRECT_EXAMPLES_PER_CLASS]
    misclassified = [
        candidate
        for candidate in candidates
        if candidate.true_label_id != candidate.predicted_label_id
    ][:MAX_MISCLASSIFIED_EXAMPLES]
    return {
        "correct_normal": correct_normal,
        "correct_pneumonia": correct_pneumonia,
        "misclassified": misclassified,
    }


def save_gradcam_figure(
    *,
    figure_path: Path,
    dataset: ChestXRayDataset,
    candidate: GradCamCandidate,
    gradcam: GradCam,
) -> None:
    """Save an original image plus Grad-CAM overlay figure.

    Args:
        figure_path: Destination PNG path.
        dataset: Test dataset used to resolve the original image path.
        candidate: Selected prediction sample.
        gradcam: Initialized Grad-CAM generator.

    Raises:
        OSError: If the source image or output path cannot be used.
        GradCamError: If Grad-CAM generation fails.
    """
    sample = dataset.get_sample(candidate.dataset_index)
    try:
        with Image.open(sample.filepath) as image:
            original_image = image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        message = f"Grad-CAM kaynak görseli okunamadı: {sample.filepath}. Hata: {error}"
        raise OSError(message) from error

    image_tensor, _label_id = dataset[candidate.dataset_index]
    gradcam_result = gradcam.generate(
        image_tensor=image_tensor,
        original_image=original_image,
        target_class_id=candidate.predicted_label_id,
    )

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(original_image.convert("L"), cmap="gray")
    axes[0].set_title(f"Original\ntrue={ID_TO_LABEL[candidate.true_label_id]}")
    axes[0].axis("off")
    axes[1].imshow(gradcam_result.overlay)
    axes[1].set_title(
        "Grad-CAM Overlay\n"
        f"pred={ID_TO_LABEL[candidate.predicted_label_id]}, "
        f"p(PNEUMONIA)={candidate.pneumonia_probability:.3f}"
    )
    axes[1].axis("off")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def generate_gradcam_examples(config: ProjectConfig) -> GradCamOutputs:
    """Generate Grad-CAM figures and Markdown documentation.

    Args:
        config: Validated project configuration.

    Returns:
        Generated figure and summary paths.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        RuntimeError: If checkpoint loading fails.
        XRayDatasetError: If test manifest data is unavailable.
        GradCamError: If Grad-CAM target layer resolution or generation fails.
    """
    require_manifest_files(config.paths.processed_data_dir)
    device = resolve_gradcam_device(config)
    checkpoint_bundle = load_checkpoint_model(config.paths.best_model_path, device)
    model_name = str(checkpoint_bundle.metadata["model_name"])
    target_layer = resolve_target_layer(checkpoint_bundle.model, model_name)
    test_manifest_path = get_manifest_paths(config.paths.processed_data_dir)["test"]
    test_dataset = ChestXRayDataset(
        test_manifest_path,
        transform=build_eval_transforms(int(checkpoint_bundle.metadata["image_size"])),
    )
    candidates = collect_gradcam_candidates(
        model=checkpoint_bundle.model,
        dataset=test_dataset,
        device=device,
    )
    selected_examples = select_examples(candidates)

    figure_paths: list[Path] = []
    gradcam = GradCam(checkpoint_bundle.model, target_layer, device)
    try:
        for group_name, group_candidates in selected_examples.items():
            for position, candidate in enumerate(group_candidates, start=1):
                figure_path = config.paths.figures_dir / f"gradcam_{group_name}_{position:02d}.png"
                save_gradcam_figure(
                    figure_path=figure_path,
                    dataset=test_dataset,
                    candidate=candidate,
                    gradcam=gradcam,
                )
                figure_paths.append(figure_path)
    finally:
        gradcam.close()

    summary_path = Path("docs/gradcam_summary.md")
    save_gradcam_summary(
        summary_path=summary_path,
        model_name=model_name,
        device=device,
        outputs=figure_paths,
        selected_examples=selected_examples,
    )
    return GradCamOutputs(
        figure_paths=figure_paths,
        summary_path=summary_path,
        correct_normal_count=len(selected_examples["correct_normal"]),
        correct_pneumonia_count=len(selected_examples["correct_pneumonia"]),
        misclassified_count=len(selected_examples["misclassified"]),
    )


def save_gradcam_summary(
    *,
    summary_path: Path,
    model_name: str,
    device: torch.device,
    outputs: list[Path],
    selected_examples: dict[str, list[GradCamCandidate]],
) -> None:
    """Write a Markdown summary explaining the generated Grad-CAM outputs.

    Args:
        summary_path: Destination Markdown file.
        model_name: Explained model architecture.
        device: Runtime device used for generation.
        outputs: Generated Grad-CAM figure paths.
        selected_examples: Candidate groups included in the report.

    Raises:
        OSError: If the summary file cannot be written.
    """
    output_lines = (
        "\n".join(f"- `{path}`" for path in outputs) if outputs else "- Görsel üretilemedi."
    )
    correct_normal_count = len(selected_examples["correct_normal"])
    correct_pneumonia_count = len(selected_examples["correct_pneumonia"])
    misclassified_count = len(selected_examples["misclassified"])
    misclassified_note = (
        f"{misclassified_count} hatalı sınıflandırılmış örnek için görsel üretildi."
        if misclassified_count
        else "Test seçiminde hatalı sınıflandırılmış örnek bulunmadığı için bu grup boş kaldı."
    )
    summary = f"""# Grad-CAM Açıklanabilirlik Özeti

## Grad-CAM Nedir?

Grad-CAM, bir CNN modelinin belirli bir sınıf kararını verirken son evrişimsel özellik
haritalarında hangi bölgelerin daha etkili olduğunu yaklaşık olarak görselleştiren bir
açıklanabilirlik yöntemidir. Üretilen ısı haritası, modelin karar skoruna katkısı yüksek
olan bölgeleri sıcak renklerle gösterir.

## Bu Projede Nasıl Kullanıldı?

- Açıklanan model: `{model_name}`
- Çalıştırma cihazı: `{device}`
- Hedef katman: seçilen mimarinin son evrişimsel özellik katmanı
- Açıklanan sınıf: modelin tahmin ettiği sınıf
- Görsel formatı: orijinal röntgen ve Grad-CAM heatmap overlay

Seçilen örnekler:

- Doğru sınıflandırılmış NORMAL: `{correct_normal_count}`
- Doğru sınıflandırılmış PNEUMONIA: `{correct_pneumonia_count}`
- Hatalı sınıflandırılmış örnek: `{misclassified_count}`

## Model Hangi Alanlara Odaklanıyor Gibi Görünüyor?

Üretilen overlay görselleri, modelin kararını görüntünün belirli akciğer bölgelerinde
yoğunlaşan aktivasyonlarla ilişkilendirdiğini incelemek için kullanılabilir. Özellikle
PNEUMONIA tahminlerinde sıcak bölgelerin akciğer alanları üzerinde kalıp kalmadığı kontrol
edilmelidir. Eğer ısı haritası görüntü kenarları, yazılar veya akciğer dışı alanlara
yoğunlaşıyorsa bu durum modelin klinik olarak anlamlı olmayan ipuçlarını öğrenmiş
olabileceğine işaret eder.

## Sınırlılıklar

Grad-CAM nedensel bir açıklama değildir; yalnızca modelin son evrişimsel özellikleri
üzerinden yaklaşık bir görsel yorum sağlar. Isı haritası yüksek çözünürlüklü patoloji
lokalizasyonu olarak değerlendirilmemelidir. Bu proje eğitim amaçlıdır ve üretilen
tahminler veya açıklamalar tıbbi teşhis amacıyla kullanılmamalıdır.

## Üretilen Görseller

{output_lines}

## Not

{misclassified_note}

## Uyarı

{MEDICAL_WARNING}
"""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse Grad-CAM generation command line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Eğitilmiş model için Grad-CAM açıklanabilirlik görselleri üret."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="YAML config dosyası yolu.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Grad-CAM example generation from the command line.

    Raises:
        ConfigFileError: If the config file is invalid.
        FileNotFoundError: If the trained checkpoint is unavailable.
        RuntimeError: If checkpoint loading fails.
        XRayDatasetError: If test data cannot be loaded.
        GradCamError: If Grad-CAM generation fails.
    """
    args = parse_args()
    try:
        config = load_and_prepare_config(args.config)
    except ConfigFileError:
        raise

    outputs = generate_gradcam_examples(config)
    print("Grad-CAM görselleri üretildi.")
    print(f"Correct NORMAL examples: {outputs.correct_normal_count}")
    print(f"Correct PNEUMONIA examples: {outputs.correct_pneumonia_count}")
    print(f"Misclassified examples: {outputs.misclassified_count}")
    print(f"Summary: {outputs.summary_path}")
    for figure_path in outputs.figure_paths:
        print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
