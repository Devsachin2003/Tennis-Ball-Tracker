"""Download runtime assets required by the validation spike.

This script intentionally uses only Python's standard library so it can run
before project dependencies are fully installed.
"""

from __future__ import annotations

import logging
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger("download_assets")

USER_AGENT = (
    "Ball-Tracker-Validation-Spike/1.0 "
    "(asset bootstrap; urllib.request; +https://github.com/)"
)
CHUNK_SIZE_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Asset:
    """Downloadable binary asset."""

    name: str
    url: str
    destination: Path
    fallback_urls: tuple[str, ...] = ()


ASSETS = (
    Asset(
        name="TrackNetV2 weights",
        url="https://raw.githubusercontent.com/ChgygLin/TrackNetV2-pytorch/master/tf2torch/track.pt",
        destination=Path("models/tracknetv2.pt"),
    ),
    Asset(
        name="Sample rally video",
        url="https://huggingface.co/datasets/hf-internal-testing/fixtures_video/resolve/main/tennis.mp4",
        destination=Path("test_data/sample_rally.mp4"),
        fallback_urls=("https://samplelib.com/mp4/sample-10s.mp4",),
    ),
)


def configure_logging() -> None:
    """Configure readable console logs."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directories() -> None:
    """Create asset directories if they are missing."""

    for directory in (Path("models"), Path("test_data")):
        directory.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Verified directory: %s", directory)


def build_request(url: str) -> urllib.request.Request:
    """Build a urllib request with headers accepted by GitHub raw/CDN hosts."""

    return urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream,*/*;q=0.8",
        },
    )


def download_asset(asset: Asset) -> None:
    """Download one asset unless it already exists."""

    if asset.destination.exists() and asset.destination.stat().st_size > 0:
        LOGGER.info(
            "Skipping %s; already exists at %s (%.2f MB).",
            asset.name,
            asset.destination,
            asset.destination.stat().st_size / (1024 * 1024),
        )
        return

    errors: list[str] = []
    for url in (asset.url, *asset.fallback_urls):
        try:
            download_asset_from_url(asset, url)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
            LOGGER.warning("%s", exc)

    raise RuntimeError(
        f"Unable to download {asset.name} after {1 + len(asset.fallback_urls)} "
        f"attempt(s): {' | '.join(errors)}"
    )


def download_asset_from_url(asset: Asset, url: str) -> None:
    """Download one asset from a specific URL."""

    temp_path = asset.destination.with_suffix(asset.destination.suffix + ".download")
    if temp_path.exists():
        LOGGER.info("Removing stale partial download: %s", temp_path)
        temp_path.unlink()

    LOGGER.info("Downloading %s", asset.name)
    LOGGER.info("Source: %s", url)
    LOGGER.info("Target: %s", asset.destination)

    request = build_request(url)
    context = ssl._create_unverified_context()
    started_at = time.monotonic()

    try:
        with urllib.request.urlopen(
            request,
            context=context,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            total_bytes = int(response.headers.get("Content-Length") or 0)
            downloaded_bytes = 0
            next_log_threshold = 0

            with temp_path.open("wb") as output_file:
                while True:
                    chunk = response.read(CHUNK_SIZE_BYTES)
                    if not chunk:
                        break

                    output_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    next_log_threshold = log_progress(
                        downloaded_bytes=downloaded_bytes,
                        total_bytes=total_bytes,
                        next_log_threshold=next_log_threshold,
                    )

        shutil.move(str(temp_path), str(asset.destination))
        elapsed = max(time.monotonic() - started_at, 0.001)
        LOGGER.info(
            "Finished %s: %.2f MB in %.1fs (%.2f MB/s).",
            asset.name,
            asset.destination.stat().st_size / (1024 * 1024),
            elapsed,
            (asset.destination.stat().st_size / (1024 * 1024)) / elapsed,
        )
    except TimeoutError as exc:
        cleanup_partial(temp_path)
        raise RuntimeError(f"Timed out while downloading {asset.name}: {exc}") from exc
    except urllib.error.URLError as exc:
        cleanup_partial(temp_path)
        raise RuntimeError(f"Network error while downloading {asset.name}: {exc}") from exc
    except OSError as exc:
        cleanup_partial(temp_path)
        raise RuntimeError(f"Filesystem error while saving {asset.name}: {exc}") from exc


def log_progress(
    downloaded_bytes: int,
    total_bytes: int,
    next_log_threshold: int,
) -> int:
    """Log download progress in stable increments."""

    if total_bytes > 0:
        percent = int((downloaded_bytes / total_bytes) * 100)
        if percent >= next_log_threshold:
            LOGGER.info(
                "Progress: %s%% (%.2f / %.2f MB)",
                min(percent, 100),
                downloaded_bytes / (1024 * 1024),
                total_bytes / (1024 * 1024),
            )
            return next_log_threshold + 10

        return next_log_threshold

    if downloaded_bytes // (10 * CHUNK_SIZE_BYTES) >= next_log_threshold:
        LOGGER.info("Progress: %.2f MB downloaded", downloaded_bytes / (1024 * 1024))
        return next_log_threshold + 1

    return next_log_threshold


def cleanup_partial(temp_path: Path) -> None:
    """Delete a partial download file if present."""

    if temp_path.exists():
        temp_path.unlink()
        LOGGER.info("Removed partial download: %s", temp_path)


def main() -> int:
    configure_logging()
    ensure_directories()

    try:
        for asset in ASSETS:
            download_asset(asset)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("All requested assets are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
