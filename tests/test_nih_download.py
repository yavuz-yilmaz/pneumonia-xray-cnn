"""Protect data downloads against corrupt resume and unsafe output names."""

import io
import json
from pathlib import Path

import pytest
from src.data import download_nih


def record(size: int = 8) -> dict:
    return {
        "name": "images_001.tar.gz",
        "url": "https://nihcc.box.com/shared/static/example.gz",
        "expected_bytes": size,
    }


@pytest.mark.parametrize("name", ["../images_001.tar.gz", "C:/test.tar.gz", "images_001.zip"])
def test_reject_unsafe_name(name: str) -> None:
    item = dict(record(), name=name)
    with pytest.raises(ValueError, match="archive name"):
        download_nih.validate_record(item)


def test_reject_wrong_range() -> None:
    with pytest.raises(ValueError, match="Content-Range"):
        download_nih.validate_range("bytes 0-7/8", 4, 7, 8)


class Response(io.BytesIO):
    """Minimal streaming response with an explicit HTTP range."""

    def __init__(self, data: bytes, content_range: str, status: int = 206) -> None:
        super().__init__(data)
        self.headers = {"Content-Range": content_range}
        self.status = status


def test_resume_verifies_saved_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"\x1f\x8babcdef"
    partial = tmp_path / "images_001.tar.gz.partial"
    partial.write_bytes(payload[:4])
    monkeypatch.setattr(
        download_nih, "urlopen", lambda *a, **k: Response(payload, "bytes 0-7/8")
    )
    download_nih.download_archive(record(), tmp_path)
    final = tmp_path / "images_001.tar.gz"
    assert final.read_bytes() == payload
    assert not partial.exists()
    provenance = json.loads((tmp_path / "images_001.tar.gz.provenance.json").read_bytes())
    assert provenance["sha256"] == download_nih.file_sha256(final)
    assert provenance["publisher_checksum_verified"] is False


def test_changed_remote_tail_preserves_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = tmp_path / "images_001.tar.gz.partial"
    partial.write_bytes(b"\x1f\x8bxx")
    monkeypatch.setattr(
        download_nih, "urlopen", lambda *a, **k: Response(b"\x1f\x8babcdef", "bytes 0-7/8")
    )
    with pytest.raises(ValueError, match="no longer matches"):
        download_nih.download_archive(record(), tmp_path)
    assert partial.read_bytes() == b"\x1f\x8bxx"
    assert not (tmp_path / "images_001.tar.gz").exists()


def test_ignored_range_does_not_create_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        download_nih, "urlopen", lambda *a, **k: Response(b"\x1f\x8babcdef", "bytes 0-7/8", 200)
    )
    with pytest.raises(ValueError, match="honor"):
        download_nih.download_archive(record(), tmp_path)
    assert not list(tmp_path.iterdir())
