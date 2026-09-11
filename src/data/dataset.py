"""PyTorch dataset implementation for chest X-ray classification manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset


LABEL_TO_ID: dict[str, int] = {"NORMAL": 0, "PNEUMONIA": 1}
ID_TO_LABEL: dict[int, str] = {label_id: label for label, label_id in LABEL_TO_ID.items()}
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpeg", ".jpg", ".png"}


class XRayDatasetError(RuntimeError):
    """Raised when an X-ray dataset manifest or image cannot be used."""


@dataclass(frozen=True)
class XRaySample:
    """One manifest-backed chest X-ray sample.

    Attributes:
        filepath: Path to the image file.
        split: Dataset split name.
        label: Human-readable class label.
        label_id: Numeric class identifier.
    """

    filepath: Path
    split: str
    label: str
    label_id: int


class ChestXRayDataset(Dataset[tuple[Tensor, int]]):
    """Load chest X-ray images and labels from a standardized manifest CSV.

    Args:
        manifest_path: CSV file with `filepath`, `split`, `label`, and `label_id` columns.
        transform: Optional callable applied to each PIL RGB image.

    Raises:
        XRayDatasetError: If the manifest is missing, malformed, empty, or contains invalid
            labels, label identifiers, paths, or unsupported image extensions.
    """

    required_columns = frozenset({"filepath", "split", "label", "label_id"})

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.samples = self._load_manifest(self.manifest_path)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of manifest rows.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load and transform one image sample.

        Args:
            index: Zero-based sample index.

        Returns:
            Tuple of image tensor and numeric class label.

        Raises:
            XRayDatasetError: If the image cannot be opened, decoded, or transformed.
            IndexError: If the sample index is outside the dataset bounds.
        """
        sample = self.samples[index]
        try:
            with Image.open(sample.filepath) as image:
                rgb_image = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            message = (
                f"Görüntü okunamadı: {sample.filepath}. "
                f"Split='{sample.split}', label='{sample.label}'. Hata: {error}"
            )
            raise XRayDatasetError(message) from error

        if self.transform is None:
            message = "Dataset transform tanımlı değil; görüntüyü tensöre çevirmek için transform gerekir."
            raise XRayDatasetError(message)

        try:
            image_tensor = self.transform(rgb_image)
        except (RuntimeError, ValueError, TypeError) as error:
            message = f"Görüntü transform aşamasında işlenemedi: {sample.filepath}. Hata: {error}"
            raise XRayDatasetError(message) from error

        return image_tensor, sample.label_id

    def get_sample(self, index: int) -> XRaySample:
        """Return manifest metadata for one sample without loading the image.

        Args:
            index: Zero-based sample index.

        Returns:
            The immutable sample metadata.

        Raises:
            IndexError: If the sample index is outside the dataset bounds.
        """
        return self.samples[index]

    @classmethod
    def _load_manifest(cls, manifest_path: Path) -> list[XRaySample]:
        if not manifest_path.is_file():
            message = f"Manifest dosyası bulunamadı: {manifest_path}"
            raise XRayDatasetError(message)

        try:
            with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
                reader = csv.DictReader(manifest_file)
                if reader.fieldnames is None:
                    message = f"Manifest boş veya başlık satırı yok: {manifest_path}"
                    raise XRayDatasetError(message)

                missing_columns = cls.required_columns.difference(reader.fieldnames)
                if missing_columns:
                    formatted_columns = ", ".join(sorted(missing_columns))
                    message = (
                        f"Manifest gerekli kolonları içermiyor: {manifest_path}. "
                        f"Eksik kolonlar: {formatted_columns}"
                    )
                    raise XRayDatasetError(message)

                samples = [
                    cls._parse_manifest_row(row=row, row_number=row_number, manifest_path=manifest_path)
                    for row_number, row in enumerate(reader, start=2)
                ]
        except OSError as error:
            message = f"Manifest dosyası okunamadı: {manifest_path}. Hata: {error}"
            raise XRayDatasetError(message) from error

        if not samples:
            message = f"Manifest içinde örnek bulunamadı: {manifest_path}"
            raise XRayDatasetError(message)

        return samples

    @staticmethod
    def _parse_manifest_row(
        row: dict[str, str],
        row_number: int,
        manifest_path: Path,
    ) -> XRaySample:
        filepath_value = row.get("filepath", "").strip()
        split = row.get("split", "").strip()
        label = row.get("label", "").strip()
        label_id_value = row.get("label_id", "").strip()

        if not filepath_value:
            message = f"Manifest satırında filepath boş: {manifest_path}:{row_number}"
            raise XRayDatasetError(message)

        filepath = Path(filepath_value)
        if not filepath.is_file():
            message = f"Manifest görsel dosyası bulunamadı: {filepath} ({manifest_path}:{row_number})"
            raise XRayDatasetError(message)

        if filepath.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            message = (
                f"Desteklenmeyen görsel uzantısı: {filepath.suffix} ({filepath}). "
                f"Desteklenen uzantılar: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )
            raise XRayDatasetError(message)

        if label not in LABEL_TO_ID:
            message = (
                f"Geçersiz label değeri: '{label}' ({manifest_path}:{row_number}). "
                f"Beklenen değerler: {sorted(LABEL_TO_ID)}"
            )
            raise XRayDatasetError(message)

        try:
            label_id = int(label_id_value)
        except ValueError as error:
            message = f"label_id tam sayı olmalı: '{label_id_value}' ({manifest_path}:{row_number})"
            raise XRayDatasetError(message) from error

        expected_label_id = LABEL_TO_ID[label]
        if label_id != expected_label_id:
            message = (
                f"Label mapping tutarsız: label='{label}' için label_id={expected_label_id} "
                f"bekleniyordu, {label_id} alındı ({manifest_path}:{row_number})."
            )
            raise XRayDatasetError(message)

        return XRaySample(filepath=filepath, split=split, label=label, label_id=label_id)
