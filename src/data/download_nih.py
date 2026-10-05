"""Download public NIH archives from a locally verified official-link manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def validate_record(record: dict) -> None:
    """Restrict names and source URLs before making requests or creating files."""
    url = urlparse(record["url"])
    if (
        url.scheme != "https"
        or url.hostname != "nihcc.box.com"
        or not re.fullmatch(r"/shared/static/[a-z0-9]+\.gz", url.path)
        or url.query
        or url.fragment
        or url.username
        or url.password
        or url.port
    ):
        raise ValueError("Expected an official NIH public static URL")
    if not re.fullmatch(r"images_\d{3}\.tar\.gz", record["name"]):
        raise ValueError("Invalid archive name")
    if not isinstance(record["expected_bytes"], int) or record["expected_bytes"] <= 0:
        raise ValueError("Invalid expected archive size")


def validate_range(header: str | None, start: int, end: int, total: int) -> None:
    """Reject ignored ranges or a different remote object before appending bytes."""
    if header != f"bytes {start}-{end}/{total}":
        raise ValueError(f"Unexpected Content-Range: {header!r}")


def file_sha256(path: Path) -> str:
    """Hash a large file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(record: dict, output: Path) -> None:
    """Fetch validated byte ranges, resume partial files, and preserve provenance."""
    validate_record(record)
    destination = output / record["name"]
    provenance = destination.with_suffix(destination.suffix + ".provenance.json")
    if destination.exists():
        if not provenance.exists():
            raise FileExistsError(f"Unverified existing archive: {destination}")
        previous = json.loads(provenance.read_bytes())
        if (
            previous["source_url"] != record["url"]
            or destination.stat().st_size != record["expected_bytes"]
            or file_sha256(destination) != previous["sha256"]
        ):
            raise ValueError("Previously downloaded archive failed verification")
        print(f"Verified existing {destination.name}", flush=True)
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    total = record["expected_bytes"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise ValueError("Partial file exceeds expected archive size")
    failures = 0
    last_report = 0.0
    while offset < total:
        # Recheck the saved tail on every range to detect an inconsistent resume.
        overlap = min(offset, 4096)
        prefix = b""
        if overlap:
            with partial.open("rb") as stream:
                stream.seek(offset - overlap)
                prefix = stream.read(overlap)
        start = offset - overlap
        end = min(total - 1, offset + 64 * 1024 * 1024 - 1)
        request = Request(  # noqa: S310 - validate_record restricts this to official HTTPS URLs.
            record["url"],
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        )
        try:
            # The source host and path are validated above; redirects are Box's download route.
            with urlopen(request, timeout=45) as response:  # noqa: S310
                if response.status != 206:
                    raise ValueError("Server did not honor the requested byte range")
                validate_range(response.headers.get("Content-Range"), start, end, total)
                payload = response.read(end - start + 2)
            if len(payload) != end - start + 1:
                raise OSError("Incomplete byte range")
            if payload[:overlap] != prefix:
                raise ValueError("Remote archive no longer matches partial file")
            if offset == 0 and payload[:2] != b"\x1f\x8b":
                raise ValueError("Response is not a gzip archive")
            with partial.open("ab") as stream:
                stream.write(payload[overlap:])
            offset = partial.stat().st_size
            failures = 0
        except OSError as error:
            failures += 1
            if failures >= 5:
                raise
            print(f"Retry {record['name']}: {type(error).__name__}", flush=True)
            time.sleep(min(2**failures, 30))
            continue
        now = time.monotonic()
        if now - last_report >= 30 or offset == total:
            print(
                f"{record['name']}: {offset}/{total} bytes ({100 * offset / total:.1f}%)",
                flush=True,
            )
            last_report = now
    digest = file_sha256(partial)
    # Never overwrite existing archive data, including a concurrently completed download.
    with destination.open("xb") as target, partial.open("rb") as source:
        shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
    provenance.write_text(
        json.dumps(
            {
                "source_url": record["url"],
                "bytes": total,
                "sha256": digest,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "verification": (
                    "Exact official listing size and validated HTTP ranges; local SHA256"
                ),
                "publisher_checksum_verified": False,
                "archive_integrity": (
                    "Gzip CRC and image decoding must be checked during extraction"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # This file was created only by this downloader and has just been copied and hashed.
    partial.unlink()
    print(f"Completed {destination.name}: SHA256 {digest}", flush=True)


def main() -> None:
    """Download listed archives without executing the publisher's Python script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/nih_archives"))
    parser.add_argument("--start", type=int, default=1, help="First archive index, starting at 1")
    parser.add_argument("--limit", type=int, default=0, help="At most N archives; 0 means all")
    parser.add_argument(
        "--indices", type=int, nargs="+", help="Specific archive indices, starting at 1"
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    manifest = json.loads(args.manifest.read_bytes())
    if not 1 <= args.start <= len(manifest["archives"]):
        parser.error("--start is outside the manifest")
    if args.indices is not None:
        if args.start != 1 or args.limit:
            parser.error("--indices cannot be combined with --start or --limit")
        if len(set(args.indices)) != len(args.indices) or any(
            not 1 <= index <= len(manifest["archives"]) for index in args.indices
        ):
            parser.error("Archive indices must be unique and inside the manifest")
        records = [manifest["archives"][index - 1] for index in args.indices]
    else:
        records = manifest["archives"][args.start - 1 :][: args.limit or None]
    for record in records:
        validate_record(record)
    names = [record["name"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("Manifest repeats archive names")
    args.output.mkdir(parents=True, exist_ok=True)
    required = sum(record["expected_bytes"] for record in records)
    if shutil.disk_usage(args.output).free < 2 * required:
        raise OSError("Insufficient free space for archives and partial-file completion")
    print(f"Downloading {len(records)} official NIH archives", flush=True)
    for record in records:
        download_archive(record, args.output)


if __name__ == "__main__":
    main()
