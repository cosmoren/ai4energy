"""
Convert Himawari L2 cloud-property NetCDF frames to the common tiled Zarr.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

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


DEFAULT_PATH = Path("data")
DEFAULT_TILE = 100
DEFAULT_OUTPUT = Path("himawari_l2.zarr")
DEFAULT_START = "2024-01-01 00:00"
DEFAULT_END = "2024-01-31 11:50"
DEFAULT_CHANNELS = ("CLOT",)  # CLOT, CLTT, CLTH, CLER_23, CLTYPE

FILENAME_RE = re.compile(
    r"^NC_[^_]+_(?P<date>\d{8})_(?P<time>\d{4})_.*\.nc$",
    re.IGNORECASE,
)


# L2 NetCDF discovery
# ===================

def datetime_from_filename(path: Path) -> datetime | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    return datetime.strptime(match["date"] + match["time"], "%Y%m%d%H%M")


def scan_source_files(
    root: Path,
    start: datetime,
    end: datetime,
) -> tuple[dict[datetime, Path], Path | None]:
    """Index one NetCDF per timestamp and return a grid reference file."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"input root does not exist or is not a directory: {root}"
        )

    indexed: dict[datetime, Path] = {}
    representative: Path | None = None
    for year in range(start.year, end.year + 1):
        year_directory = root / f"{year:04d}"
        if not year_directory.is_dir():
            continue
        for path in year_directory.rglob("*.nc"):
            timestamp = datetime_from_filename(path)
            if timestamp is None:
                continue
            if representative is None:
                representative = path
            if not start <= timestamp <= end:
                continue
            previous = indexed.get(timestamp)
            if previous is not None:
                raise RuntimeError(
                    f"multiple source files resolve to {timestamp}: "
                    f"{previous} and {path}"
                )
            indexed[timestamp] = path

    if representative is None:
        for path in root.rglob("*.nc"):
            if datetime_from_filename(path) is not None:
                representative = path
                break
    return indexed, representative


# L2 regular-lat/lon mapping
# ==========================

def read_source_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as source:
        try:
            latitude = np.asarray(source["latitude"][:], dtype=np.float64)
            longitude = np.asarray(source["longitude"][:], dtype=np.float64)
        except KeyError as error:
            raise RuntimeError(f"{path} lacks latitude/longitude datasets") from error
    return latitude, longitude


def nearest_coordinate_indices(
    coordinates: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Return nearest indices in a strictly monotonic 1-D coordinate."""
    coordinate = np.asarray(coordinates, dtype=np.float64)
    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError("source coordinate must be a one-dimensional array")

    delta = np.diff(coordinate)
    if np.all(delta > 0):
        ordered_coordinate = coordinate
        ordered_index = np.arange(coordinate.size, dtype=np.float64)
    elif np.all(delta < 0):
        ordered_coordinate = coordinate[::-1]
        ordered_index = np.arange(
            coordinate.size - 1, -1, -1, dtype=np.float64
        )
    else:
        raise ValueError("source coordinate must be strictly monotonic")

    if np.any(values < ordered_coordinate[0]) or np.any(
        values > ordered_coordinate[-1]
    ):
        raise ValueError("target footprint extends outside the source coordinate range")
    fractional = np.interp(values, ordered_coordinate, ordered_index)
    return np.rint(fractional).astype(np.int16)


def mapping_for_center(
    center_lat: float,
    center_lon: float,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = target_lon_lat(center_lat, center_lon)
    target_u = nearest_coordinate_indices(source_longitude, longitude)
    target_v = nearest_coordinate_indices(source_latitude, latitude)
    return target_u, target_v


def build_tile_mappings(
    n_tile: int,
    seed: int,
    representative: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if representative is None:
        raise RuntimeError(
            "no readable NetCDF was found; one source file is needed to map tiles"
        )
    source_latitude, source_longitude = read_source_coordinates(representative)

    def mapper(center_lat: float, center_lon: float):
        return mapping_for_center(
            center_lat,
            center_lon,
            source_latitude,
            source_longitude,
        )

    return sample_tiles(n_tile, seed, mapper)


# L2 packed-value reading and validity
# ====================================

def scalar_attribute(dataset: h5py.Dataset, name: str):
    if name not in dataset.attrs:
        return None
    value = np.asarray(dataset.attrs[name]).reshape(-1)
    return value[0] if value.size else None


def validity_mask(dataset: h5py.Dataset, values: np.ndarray) -> np.ndarray:
    """Derive validity without changing any packed source value."""
    valid = np.ones(values.shape, dtype=np.bool_)
    valid_min = scalar_attribute(dataset, "valid_min")
    valid_max = scalar_attribute(dataset, "valid_max")
    if valid_min is not None:
        valid &= values >= valid_min
    if valid_max is not None:
        valid &= values <= valid_max
    for attribute in ("missing_value", "_FillValue"):
        missing = scalar_attribute(dataset, attribute)
        if missing is not None:
            valid &= values != missing
    return valid


def read_frame_once(
    path: Path,
    channels: Sequence[str],
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Open one NetCDF once and extract every channel/tile from one window."""
    n_tile = target_u.shape[0]
    result = np.empty(
        (n_tile, len(channels), TILE_PIXELS, TILE_PIXELS), dtype=np.int16
    )
    valid_result = np.empty(result.shape, dtype=np.bool_)

    u_min, u_max = int(target_u.min()), int(target_u.max())
    v_min, v_max = int(target_v.min()), int(target_v.max())
    if u_min < 0 or v_min < 0:
        raise RuntimeError("target_u/target_v contains an invalid source index")
    local_u = target_u.astype(np.int64) - u_min
    local_v = target_v.astype(np.int64) - v_min

    with h5py.File(path, "r") as source:
        for channel_index, name in enumerate(channels):
            if name not in source:
                raise KeyError(f"{path} lacks requested channel {name!r}")
            dataset = source[name]
            if dataset.ndim != 2:
                raise RuntimeError(f"{path}:{name} is not a two-dimensional image")
            if u_max >= dataset.shape[1] or v_max >= dataset.shape[0]:
                raise RuntimeError(f"target mapping is outside {path}:{name}")
            if dataset.dtype.kind != "i" or dataset.dtype.itemsize != 2:
                raise RuntimeError(
                    f"{path}:{name} has dtype {dataset.dtype}; expected packed int16"
                )

            source_window = np.asarray(
                dataset[v_min : v_max + 1, u_min : u_max + 1]
            )
            sampled = source_window[local_v, local_u]
            result[:, channel_index] = sampled.astype(np.int16, copy=False)
            valid_result[:, channel_index] = validity_mask(dataset, sampled)
    return result, valid_result


# Command-line and Python entry points
# ====================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop Himawari L2 NetCDF frames into the common tiled Zarr."
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
    """Create/resume an L2 store and return (written, already_valid, failed)."""
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
    normalized_channels = tuple(str(name) for name in channels)
    if not normalized_channels or len(set(normalized_channels)) != len(
        normalized_channels
    ):
        raise ValueError("channels must contain one or more unique names")

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

    source_files, representative = scan_source_files(
        input_root, start_time, end_time
    )
    print(f"Found {len(source_files)} source frames inside the requested range")
    root = initialise_or_validate_store(
        output_path,
        timestamps,
        normalized_channels,
        n_tile,
        lambda: build_tile_mappings(n_tile, seed, representative),
        root_attributes={
            "product_level": "L2",
            "source_format": "Himawari cloud-property NetCDF",
            "source_projection": "regular one-dimensional latitude/longitude",
            "resampling": "nearest source coordinate",
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
