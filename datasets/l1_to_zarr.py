"""
Convert ``L1/himawari_L1_aws`` PNG frames to the common tiled Zarr.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from satpy.area import get_area_def

from zarr_common import (
    TILE_PIXELS,
    initialise_or_validate_store,
    parse_utc,
    process_frames,
    regular_times,
    sample_tiles,
    target_lon_lat,
    unix_seconds,
)


DEFAULT_PATH = Path("himawari_L1_aws")
DEFAULT_TILE = 100
DEFAULT_OUTPUT = Path("himawari_l1.zarr")
DEFAULT_START = "2023-01-01 00:00"
DEFAULT_END = "2023-01-31 23:50"
DEFAULT_CHANNELS = ("B03", "B07", "B10", "B13", "B15")
SOURCE_AREA_NAME = "himawari_ahi_fes_2km"
SOURCE_SHAPE = (5_500, 5_500)

PNG_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{4})\.png$", re.IGNORECASE)


# PNG discovery
# =============

@dataclass(frozen=True)
class PngFrameSource:
    timestamp: datetime
    paths: tuple[Path, ...]

    def path_by_channel(self, channels: Sequence[str]) -> dict[str, Path]:
        if len(channels) != len(self.paths):
            raise RuntimeError("PNG source/channel count changed after discovery")
        return dict(zip(channels, self.paths))

    def __str__(self) -> str:
        return self.timestamp.strftime("PNG frame %Y-%m-%d %H:%M")


def datetime_from_filename(path: Path) -> datetime | None:
    match = PNG_RE.match(path.name)
    if match is None:
        return None
    return datetime.strptime(match["date"] + match["time"], "%Y%m%d%H%M")


def expected_png_path(root: Path, channel: str, timestamp: datetime) -> Path:
    return root / channel / f"{timestamp:%Y%m%d_%H%M}.png"


def scan_source_files(
    root: Path,
    timeline: Sequence[datetime],
    channels: Sequence[str],
) -> tuple[dict[datetime, PngFrameSource], int]:
    """Index timestamps for which every requested band PNG exists."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"input root does not exist or is not a directory: {root}"
        )
    missing_directories = [
        channel for channel in channels if not (root / channel).is_dir()
    ]
    if missing_directories:
        raise FileNotFoundError(
            f"input root lacks channel directories: {missing_directories}"
        )

    sources: dict[datetime, PngFrameSource] = {}
    incomplete = 0
    for timestamp in timeline:
        paths = tuple(
            expected_png_path(root, channel, timestamp) for channel in channels
        )
        present = tuple(path.is_file() for path in paths)
        if all(present):
            sources[timestamp] = PngFrameSource(timestamp, paths)
        elif any(present):
            incomplete += 1
    return sources, incomplete


def validate_source_pngs(
    sources: dict[datetime, PngFrameSource],
    channels: Sequence[str],
) -> None:
    """Verify dimensions and mode using one complete source frame."""
    if not sources:
        return
    source = sources[min(sources)]
    for channel, path in source.path_by_channel(channels).items():
        with Image.open(path) as image:
            if image.size != (SOURCE_SHAPE[1], SOURCE_SHAPE[0]):
                raise RuntimeError(
                    f"{path} size is {image.size}; expected "
                    f"{(SOURCE_SHAPE[1], SOURCE_SHAPE[0])}"
                )
            if image.mode != "L":
                raise RuntimeError(
                    f"{path} mode is {image.mode!r}; {channel} must be "
                    "8-bit grayscale L"
                )


# Standard AHI 2 km full-disk mapping
# ===================================

def source_area():
    area = get_area_def(SOURCE_AREA_NAME)
    if tuple(area.shape) != SOURCE_SHAPE:
        raise RuntimeError(
            f"Satpy area {SOURCE_AREA_NAME!r} has shape {area.shape}, "
            f"expected {SOURCE_SHAPE}"
        )
    return area


def mapping_for_center(
    center_lat: float,
    center_lon: float,
    area,
) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = target_lon_lat(center_lat, center_lon)
    target_u, target_v = area.get_array_indices_from_lonlat(longitude, latitude)
    masked = np.ma.getmaskarray(target_u) | np.ma.getmaskarray(target_v)
    target_u = np.asarray(np.ma.filled(target_u, -1), dtype=np.int32)
    target_v = np.asarray(np.ma.filled(target_v, -1), dtype=np.int32)
    invalid = masked | (target_u < 0) | (target_v < 0)
    invalid |= target_u >= area.width
    invalid |= target_v >= area.height
    if invalid.any():
        raise ValueError("target footprint extends outside the AHI full-disk grid")
    return target_u.astype(np.int16), target_v.astype(np.int16)


def build_tile_mappings(
    n_tile: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    area = source_area()

    def mapper(center_lat: float, center_lon: float):
        return mapping_for_center(center_lat, center_lon, area)

    return sample_tiles(n_tile, seed, mapper)


# PNG value reading
# =================

def read_frame_once(
    source: PngFrameSource,
    channels: Sequence[str],
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read each band once and gather all requested tile pixels."""
    n_tile = target_u.shape[0]
    result = np.empty(
        (n_tile, len(channels), TILE_PIXELS, TILE_PIXELS), dtype=np.int16
    )
    # Every successfully mapped pixel in a readable image is marked as valid
    # regardless of its grayscale value, including 0.
    valid_result = np.ones(result.shape, dtype=np.bool_)

    u_min, u_max = int(target_u.min()), int(target_u.max())
    v_min, v_max = int(target_v.min()), int(target_v.max())
    if (
        u_min < 0
        or v_min < 0
        or u_max >= SOURCE_SHAPE[1]
        or v_max >= SOURCE_SHAPE[0]
    ):
        raise RuntimeError("stored target mapping is outside the PNG grid")
    local_u = target_u.astype(np.int64) - u_min
    local_v = target_v.astype(np.int64) - v_min

    for channel_index, (channel, path) in enumerate(
        source.path_by_channel(channels).items()
    ):
        with Image.open(path) as image:
            if image.size != (SOURCE_SHAPE[1], SOURCE_SHAPE[0]) or image.mode != "L":
                raise RuntimeError(
                    f"{path} must be a {SOURCE_SHAPE[1]} x {SOURCE_SHAPE[0]} "
                    "8-bit grayscale PNG"
                )
            window = np.asarray(
                image.crop((u_min, v_min, u_max + 1, v_max + 1)),
                dtype=np.uint8,
            )
        sampled = window[local_v, local_u]
        result[:, channel_index] = sampled.astype(np.int16, copy=False)
    return result, valid_result


# Command-line and Python entry points
# ====================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop 5,500 x 5,500 Himawari AWS PNG bands into the common tiled Zarr."
        )
    )
    parser.add_argument(
        "--start",
        type=parse_utc,
        default=parse_utc(DEFAULT_START),
        help=f"inclusive UTC start (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        type=parse_utc,
        default=parse_utc(DEFAULT_END),
        help=f"inclusive UTC end (default: {DEFAULT_END})",
    )
    parser.add_argument("--n-tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def convert(
    start: str | datetime = DEFAULT_START,
    end: str | datetime = DEFAULT_END,
    n_tile: int = DEFAULT_TILE,
    path: str | Path = DEFAULT_PATH,
    output: str | Path = DEFAULT_OUTPUT,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    seed: int = 42,
    fail_fast: bool = False,
) -> tuple[int, int, int]:
    """Create/resume an L1 PNG store and return (written, already_valid, failed)."""
    start_time = parse_utc(start) if isinstance(start, str) else start
    end_time = parse_utc(end) if isinstance(end, str) else end
    if start_time.tzinfo is not None:
        start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)
    if end_time.tzinfo is not None:
        end_time = end_time.astimezone(timezone.utc).replace(tzinfo=None)
    if end_time < start_time:
        raise ValueError("end must be greater than or equal to start")
    if n_tile <= 0:
        raise ValueError("n_tile must be positive")

    normalized_channels = tuple(str(channel).upper() for channel in channels)
    if not normalized_channels or len(set(normalized_channels)) != len(
        normalized_channels
    ):
        raise ValueError("channels must contain one or more unique names")
    unknown = sorted(set(normalized_channels) - set(DEFAULT_CHANNELS))
    if unknown:
        raise ValueError(
            f"unsupported PNG channels {unknown}; available: {list(DEFAULT_CHANNELS)}"
        )

    input_root = Path(path).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = regular_times(start_time, end_time)
    timestamps = unix_seconds(timeline)

    print(f"Input:  {input_root}")
    print(f"Output: {output_path}")
    print(
        f"Range:  {start_time.isoformat(sep=' ')} to "
        f"{end_time.isoformat(sep=' ')} UTC ({len(timeline)} slots)"
    )
    print(f"Tiles:  {n_tile}; channels: {', '.join(normalized_channels)}")
    print(f"Source grid assumption: Satpy {SOURCE_AREA_NAME}")

    source_files, incomplete = scan_source_files(
        input_root, timeline, normalized_channels
    )
    validate_source_pngs(source_files, normalized_channels)
    print(
        f"Found {len(source_files)} complete PNG frames inside the requested range"
        + (f"; {incomplete} timestamps were incomplete" if incomplete else "")
    )

    root = initialise_or_validate_store(
        output_path,
        timestamps,
        normalized_channels,
        n_tile,
        lambda: build_tile_mappings(n_tile, seed),
        root_attributes={
            "product_level": "L1 PNG",
            "source_format": "5500 x 5500 uint8 grayscale PNG",
            "source_projection": (
                "assumed Satpy himawari_ahi_fes_2km geostationary full-disk grid"
            ),
            "source_value_meaning": (
                "exact PNG grayscale levels 0..255; not calibrated physical values"
            ),
            "pixel_valid_rule": (
                "all successfully mapped pixels in a readable PNG are valid; "
                "PNG contains no QA/fill metadata"
            ),
            "resampling": "nearest source pixel",
        },
    )
    result = process_frames(
        root,
        timeline,
        source_files,
        normalized_channels,
        read_frame_once,
        fail_fast,
    )
    print(f"Done: written={result[0]}, already_valid={result[1]}, failed={result[2]}")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _, _, failed = convert(
            start=args.start,
            end=args.end,
            n_tile=args.n_tile,
            path=args.data_path,
            output=args.output,
            channels=args.channels,
            seed=args.seed,
            fail_fast=args.fail_fast,
        )
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
