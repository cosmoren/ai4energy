"""Convert FY-4B regular-grid GHI NetCDF frames to the common tiled Zarr."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import zarr

from china_region import CHINA_REGION_POLYGONS
from zarr_common import (
    REGION_MAX_LAT,
    REGION_MAX_LON,
    REGION_MIN_LAT,
    REGION_MIN_LON,
    TILE_PIXELS,
    initialise_or_validate_store,
    ordered_tile_groups,
    parse_utc,
    process_frames,
    regular_times,
    sample_tiles,
    target_lon_lat,
    unix_seconds,
)


DEFAULT_PATH = Path("trial/")
DEFAULT_TILE = 100
DEFAULT_OUTPUT = Path("trial_l2.zarr")
DEFAULT_START = "2026-01-01 05:00"
DEFAULT_END = "2026-01-01 05:50"
CHANNELS = ("GHI",)

FILENAME_RE = re.compile(
    r"^FY4B_REGC_1050E_500M_(?P<date>\d{8})(?P<time>\d{4})_grid_GHI\.nc$",
    re.IGNORECASE,
)


# L2 NetCDF discovery
# ===================

def datetime_from_filename(path: Path) -> datetime | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match["date"] + match["time"], "%Y%m%d%H%M")
    except ValueError:
        return None


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

    for path in sorted(root.rglob("*.nc")):
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
    # NetCDF can configure the HDF5 storage fill value without exposing an
    # _FillValue attribute.  The FY-4B GHI files do this for NC_FILL_SHORT
    # (-32767), so inspect the dataset creation properties as well.
    creation = dataset.id.get_create_plist()
    if creation.fill_value_defined() == h5py.h5d.FILL_VALUE_USER_DEFINED:
        valid &= values != dataset.fillvalue
    return valid


def sample_dataset(
    dataset: h5py.Dataset,
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> np.ndarray:
    """Sample a 2-D dataset while reading each intersected HDF5 chunk once."""
    flat_u = np.asarray(target_u, dtype=np.int64).reshape(-1)
    flat_v = np.asarray(target_v, dtype=np.int64).reshape(-1)

    if dataset.chunks is None:
        u_min, u_max = int(flat_u.min()), int(flat_u.max())
        v_min, v_max = int(flat_v.min()), int(flat_v.max())
        source_window = np.asarray(dataset[v_min : v_max + 1, u_min : u_max + 1])
        sampled = source_window[flat_v - v_min, flat_u - u_min]
        return sampled.reshape(target_u.shape)

    chunk_height, chunk_width = dataset.chunks
    chunks_per_row = (dataset.shape[1] + chunk_width - 1) // chunk_width
    chunk_ids = (flat_v // chunk_height) * chunks_per_row + flat_u // chunk_width
    order = np.argsort(chunk_ids, kind="stable")
    ordered_chunk_ids = chunk_ids[order]
    starts = np.concatenate(
        ([0], np.flatnonzero(np.diff(ordered_chunk_ids)) + 1, [order.size])
    )
    sampled = np.empty(flat_u.shape, dtype=dataset.dtype)

    for start, end in zip(starts[:-1], starts[1:]):
        indices = order[start:end]
        selected_u = flat_u[indices]
        selected_v = flat_v[indices]
        u_min, u_max = int(selected_u.min()), int(selected_u.max())
        v_min, v_max = int(selected_v.min()), int(selected_v.max())
        source_window = np.asarray(dataset[v_min : v_max + 1, u_min : u_max + 1])
        sampled[indices] = source_window[selected_v - v_min, selected_u - u_min]

    return sampled.reshape(target_u.shape)


def read_frame_once(
    path: Path,
    channels: Sequence[str],
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Open one NetCDF once and extract every requested channel and tile."""
    if tuple(channels) != CHANNELS:
        raise ValueError(f"FY-4B GHI input requires channels={CHANNELS}")
    n_tile = target_u.shape[0]
    result = np.empty(
        (n_tile, len(channels), TILE_PIXELS, TILE_PIXELS), dtype=np.int16
    )
    valid_result = np.empty(result.shape, dtype=np.bool_)

    u_min, u_max = int(target_u.min()), int(target_u.max())
    v_min, v_max = int(target_v.min()), int(target_v.max())
    if u_min < 0 or v_min < 0:
        raise RuntimeError("target_u/target_v contains an invalid source index")
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

            sampled = sample_dataset(dataset, target_u, target_v)
            result[:, channel_index] = sampled.astype(np.int16, copy=False)
            valid_result[:, channel_index] = validity_mask(dataset, sampled)
    return result, valid_result


# Optional whole-frame tile-centre diagnostic
# ===========================================

def open_zarr_readonly(path: Path):
    try:
        return zarr.open_group(str(path), mode="r", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def select_overview_source(root: Path) -> Path:
    """Select one deterministic arbitrary NetCDF below the input directory."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"input root does not exist or is not a directory: {root}"
        )
    candidates = sorted(root.rglob("*.nc"))
    if not candidates:
        raise RuntimeError(f"no NetCDF file was found under {root}")
    return candidates[0]


def configured_invalid_values(dataset: h5py.Dataset) -> list[int | float]:
    values: list[int | float] = []
    for attribute in ("missing_value", "_FillValue"):
        value = scalar_attribute(dataset, attribute)
        if value is not None:
            values.append(value.item() if isinstance(value, np.generic) else value)
    creation = dataset.id.get_create_plist()
    if creation.fill_value_defined() == h5py.h5d.FILL_VALUE_USER_DEFINED:
        value = dataset.fillvalue
        values.append(value.item() if isinstance(value, np.generic) else value)
    return list(dict.fromkeys(values))


def source_statistics(dataset: h5py.Dataset) -> dict[str, object]:
    """Scan the complete source dataset without loading it all at once."""
    row_block = dataset.chunks[0] if dataset.chunks is not None else 512
    valid_min: int | float | None = None
    valid_max: int | float | None = None
    valid_count = 0
    invalid_count = 0
    invalid_counts: Counter[int | float] = Counter()

    for row_start in range(0, dataset.shape[0], row_block):
        values = np.asarray(dataset[row_start : row_start + row_block])
        valid = validity_mask(dataset, values)
        usable = values[valid]
        invalid = values[~valid]
        valid_count += int(usable.size)
        invalid_count += int(invalid.size)
        if usable.size:
            block_min = usable.min().item()
            block_max = usable.max().item()
            valid_min = block_min if valid_min is None else min(valid_min, block_min)
            valid_max = block_max if valid_max is None else max(valid_max, block_max)
        if invalid.size:
            unique, counts = np.unique(invalid, return_counts=True)
            invalid_counts.update(
                {
                    value.item(): int(count)
                    for value, count in zip(unique, counts)
                }
            )

    scale_attribute = scalar_attribute(dataset, "scale_factor")
    offset_attribute = scalar_attribute(dataset, "add_offset")
    scale = 1.0 if scale_attribute is None else float(scale_attribute)
    offset = 0.0 if offset_attribute is None else float(offset_attribute)
    physical_limits = (
        []
        if valid_min is None or valid_max is None
        else sorted((valid_min * scale + offset, valid_max * scale + offset))
    )
    return {
        "valid_min": valid_min,
        "valid_max": valid_max,
        "physical_min": physical_limits[0] if physical_limits else None,
        "physical_max": physical_limits[1] if physical_limits else None,
        "scale_factor": scale,
        "add_offset": offset,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "configured_invalid_values": configured_invalid_values(dataset),
        "observed_invalid_values": dict(sorted(invalid_counts.items())),
    }


def robust_limits(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    usable = values[valid]
    if usable.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(usable.astype(np.float64), [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(usable.min()), float(usable.max())
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def coordinate_edges(coordinate: np.ndarray) -> tuple[float, float]:
    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise RuntimeError("plot coordinate must be a one-dimensional vector")
    delta = np.diff(values)
    if not np.all(delta > 0):
        raise RuntimeError("plot coordinate must be strictly increasing")
    return (
        float(values[0] - delta[0] / 2.0),
        float(values[-1] + delta[-1] / 2.0),
    )


def draw_all_tile_centres(
    data_path: str | Path = DEFAULT_PATH,
    zarr_path: str | Path = DEFAULT_OUTPUT,
    png_path: str | Path | None = None,
    show: bool = False,
    maximum_display_pixels: int = 4_000_000,
) -> dict[str, object]:
    """Draw one whole L2 image with China, bounds, and all tile centres."""
    if png_path is None and not show:
        raise ValueError("set png_path or show=True")
    if png_path is not None and show:
        raise ValueError("png_path and show=True are mutually exclusive")
    if maximum_display_pixels <= 0:
        raise ValueError("maximum_display_pixels must be positive")

    input_root = Path(data_path).expanduser().resolve()
    store_path = Path(zarr_path).expanduser().resolve()
    source_path = select_overview_source(input_root)
    root = open_zarr_readonly(store_path)
    tiles = ordered_tile_groups(root)
    if not tiles:
        raise RuntimeError(f"{store_path} contains no tile groups")
    tile_longitude = np.asarray(
        [float(tile["longitude"][0]) for tile in tiles], dtype=np.float64
    )
    tile_latitude = np.asarray(
        [float(tile["latitude"][0]) for tile in tiles], dtype=np.float64
    )

    with h5py.File(source_path, "r") as source:
        channel = CHANNELS[0]
        if channel not in source:
            raise KeyError(f"{source_path} lacks channel {channel!r}")
        dataset = source[channel]
        if dataset.ndim != 2:
            raise RuntimeError(f"{source_path}:{channel} is not a 2-D image")
        latitude = np.asarray(source["latitude"][:], dtype=np.float64)
        longitude = np.asarray(source["longitude"][:], dtype=np.float64)
        statistics = source_statistics(dataset)
        stride = max(
            1,
            int(np.ceil(np.sqrt(dataset.size / maximum_display_pixels))),
        )
        image = np.asarray(dataset[::stride, ::stride])
        image_valid = validity_mask(dataset, image)

    if longitude[0] > longitude[-1]:
        longitude = longitude[::-1]
        image = image[:, ::-1]
        image_valid = image_valid[:, ::-1]
    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1]
        image = image[::-1, :]
        image_valid = image_valid[::-1, :]
    lon_min, lon_max = coordinate_edges(longitude)
    lat_min, lat_max = coordinate_edges(latitude)

    print(f"Overview source: {source_path}")
    print(
        f"Packed valid min/max: {statistics['valid_min']} / "
        f"{statistics['valid_max']}"
    )
    print(
        f"Physical min/max: {statistics['physical_min']} / "
        f"{statistics['physical_max']} "
        f"(packed * {statistics['scale_factor']} + "
        f"{statistics['add_offset']})"
    )
    print(
        f"Valid/invalid pixels: {statistics['valid_count']} / "
        f"{statistics['invalid_count']}"
    )
    print(
        "Configured invalid values: "
        f"{statistics['configured_invalid_values'] or 'none'}"
    )
    print(
        "Observed invalid values and counts: "
        f"{statistics['observed_invalid_values'] or 'none'}"
    )

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "himawari_matplotlib")
    )
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low, high = robust_limits(image, image_valid)
    displayed = np.ma.array(image, mask=~image_valid)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#8c8c8c")
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    plotted = axis.imshow(
        displayed,
        cmap=cmap,
        vmin=low,
        vmax=high,
        origin="lower",
        extent=(lon_min, lon_max, lat_min, lat_max),
        interpolation="nearest",
        aspect="equal",
    )
    for polygon_index, polygon in enumerate(CHINA_REGION_POLYGONS):
        closed = np.vstack((polygon, polygon[0]))
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color="#ffd54f",
            linewidth=1.5,
            label="China region" if polygon_index == 0 else None,
        )
    boundary_longitude = np.asarray(
        [
            REGION_MIN_LON,
            REGION_MAX_LON,
            REGION_MAX_LON,
            REGION_MIN_LON,
            REGION_MIN_LON,
        ]
    )
    boundary_latitude = np.asarray(
        [
            REGION_MIN_LAT,
            REGION_MIN_LAT,
            REGION_MAX_LAT,
            REGION_MAX_LAT,
            REGION_MIN_LAT,
        ]
    )
    axis.plot(
        boundary_longitude,
        boundary_latitude,
        color="#00e5ff",
        linewidth=1.8,
        label="Tile-centre bounds",
    )
    axis.scatter(
        tile_longitude,
        tile_latitude,
        s=24,
        color="#ff2d2d",
        edgecolor="white",
        linewidth=0.5,
        label=f"Tile centres ({len(tiles)})",
        zorder=4,
    )
    timestamp = datetime_from_filename(source_path)
    time_text = (
        "unknown time"
        if timestamp is None
        else f"{timestamp:%Y-%m-%d %H:%M} UTC"
    )
    axis.set_title(f"All tile centres | {time_text} | {CHANNELS[0]}")
    axis.set_xlabel("longitude (degrees east)")
    axis.set_ylabel("latitude (degrees north)")
    axis.legend(loc="best")
    figure.colorbar(plotted, ax=axis, shrink=0.85).set_label(
        f"{CHANNELS[0]} packed value"
    )

    if png_path is not None:
        destination = Path(png_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
        print(f"Tile-centres PNG: {destination}")
    else:
        plt.show()
    plt.close(figure)
    return statistics


# Command-line and Python entry points
# ====================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop FY-4B GHI NetCDF frames into the common tiled Zarr."
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-fast", action="store_true")
    overview_output = parser.add_mutually_exclusive_group()
    overview_output.add_argument(
        "--png-path",
        type=Path,
        default=None,
        help="save the tile-centres overview to this PNG without showing it",
    )
    overview_output.add_argument(
        "--show",
        action="store_true",
        help="show the tile-centres overview without saving it",
    )
    return parser


def convert(
    start: str | datetime = DEFAULT_START,
    end: str | datetime = DEFAULT_END,
    n_tile: int = DEFAULT_TILE,
    path: str | Path = DEFAULT_PATH,
    output: str | Path = DEFAULT_OUTPUT,
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
    print(f"Tiles:  {n_tile}; channels: {', '.join(CHANNELS)}")

    source_files, representative = scan_source_files(
        input_root, start_time, end_time
    )
    print(f"Found {len(source_files)} source frames inside the requested range")
    root = initialise_or_validate_store(
        output_path,
        timestamps,
        CHANNELS,
        n_tile,
        lambda: build_tile_mappings(n_tile, seed, representative),
        root_attributes={
            "product_level": "L2",
            "source_format": "FY-4B GHI NetCDF-4/HDF5",
            "source_projection": "regular one-dimensional latitude/longitude",
            "resampling": "nearest source coordinate",
        },
    )
    result = process_frames(
        root,
        timeline,
        source_files,
        CHANNELS,
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
            seed=args.seed,
            fail_fast=args.fail_fast,
        )
        if args.png_path is not None or args.show:
            draw_all_tile_centres(
                data_path=args.data_path,
                zarr_path=args.output,
                png_path=args.png_path,
                show=args.show,
            )
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
