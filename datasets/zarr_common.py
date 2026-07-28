"""
Common Zarr machinery for the Himawari L1 and L2 converters.

Both products write exactly this per-tile array structure:

    tile_N/
      images       (time, channel, y, x)  int16; missing frame fill is -1
      pixel_valid  (time, channel, y, x)  bool
      frame_valid  (time,)                bool
      time_utc     (time,)                int64 Unix seconds
      channel      (channel,)             source channel names
      latitude     (1,)                   tile centre latitude
      longitude    (1,)                   tile centre longitude
      target_u     (y, x)                 source image column indices
      target_v     (y, x)                 source image row indices
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import zarr
from numcodecs import Blosc
from pyproj import CRS, Transformer


# Dataset contract
# =======================

FRAME_INTERVAL = timedelta(minutes=10)
TILE_WIDTH_M = 500_000.0
TARGET_PIXEL_M = 5_000.0
TILE_PIXELS = 100
TIME_CHUNK = 12
IMAGE_FILL_VALUE = np.int16(-1)
LAYOUT = "tile_groups_v2"

# Candidate tile centres are sampled uniformly from this rough rectangle
REGION_MIN_LAT = 17.0
REGION_MAX_LAT = 55.0
REGION_MIN_LON = 73.0
REGION_MAX_LON = 136.0
REGION_BOUNDS_TEXT = (
    f"{REGION_MIN_LON:g}E-{REGION_MAX_LON:g}E, "
    f"{REGION_MIN_LAT:g}N-{REGION_MAX_LAT:g}N"
)


# Timeline and target-grid projection
# ===================================

def parse_utc(value: str) -> datetime:
    """Parse a timestamp and return a timezone-naive UTC datetime."""
    text = value.strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1].strip()
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    if parsed.second or parsed.microsecond or parsed.minute % 10:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not aligned to an exact 10-minute UTC boundary"
        )
    return parsed


def regular_times(start: datetime, end: datetime) -> list[datetime]:
    """Return every exact 10-minute timestamp in an inclusive UTC range."""
    for label, value in (("start", start), ("end", end)):
        if value.second or value.microsecond or value.minute % 10:
            raise ValueError(f"{label} is not aligned to an exact 10-minute boundary")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    count = int((end - start) // FRAME_INTERVAL) + 1
    return [start + index * FRAME_INTERVAL for index in range(count)]


def unix_seconds(values: Sequence[datetime]) -> np.ndarray:
    """Convert timezone-naive UTC datetimes to signed Unix seconds."""
    epoch = datetime(1970, 1, 1)
    return np.asarray(
        [(value - epoch).total_seconds() for value in values], dtype=np.int64
    )


def target_lon_lat(center_lat: float, center_lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Return lon/lat for a 500 km, 100 x 100 local WGS84 AEQD grid."""
    x = -TILE_WIDTH_M / 2 + (np.arange(TILE_PIXELS) + 0.5) * TARGET_PIXEL_M
    y = TILE_WIDTH_M / 2 - (np.arange(TILE_PIXELS) + 0.5) * TARGET_PIXEL_M
    target_x, target_y = np.meshgrid(x, y)
    local_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.12f} +lon_0={center_lon:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_wgs84 = Transformer.from_crs(
        local_aeqd, CRS.from_epsg(4326), always_xy=True
    )
    longitude, latitude = to_wgs84.transform(target_x, target_y)
    return np.asarray(longitude), np.asarray(latitude)


MappingForCenter = Callable[[float, float], tuple[np.ndarray, np.ndarray]]
MappingBuilder = Callable[
    [], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]
FrameReader = Callable[
    [Any, Sequence[str], np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]


def sample_tiles(
    n_tile: int,
    seed: int,
    mapping_for_center: MappingForCenter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample rectangle centres and retain tiles fully mappable by the source."""
    if n_tile <= 0:
        raise ValueError("n_tile must be positive")

    rng = np.random.default_rng(seed)
    latitudes: list[float] = []
    longitudes: list[float] = []
    mappings_u: list[np.ndarray] = []
    mappings_v: list[np.ndarray] = []
    maximum_attempts = max(10_000, n_tile * 1_000)

    for _ in range(maximum_attempts):
        if len(latitudes) == n_tile:
            break
        longitude = float(rng.uniform(REGION_MIN_LON, REGION_MAX_LON))
        latitude = float(rng.uniform(REGION_MIN_LAT, REGION_MAX_LAT))
        try:
            target_u, target_v = mapping_for_center(latitude, longitude)
        except ValueError:
            continue

        target_u = np.asarray(target_u)
        target_v = np.asarray(target_v)
        expected_shape = (TILE_PIXELS, TILE_PIXELS)
        if target_u.shape != expected_shape or target_v.shape != expected_shape:
            raise RuntimeError(
                "mapping_for_center must return target_u/target_v with shape "
                f"{expected_shape}, got {target_u.shape} and {target_v.shape}"
            )
        if target_u.min() < 0 or target_v.min() < 0:
            raise RuntimeError("mapping_for_center returned a negative source index")
        if target_u.max() > np.iinfo(np.int16).max or target_v.max() > np.iinfo(np.int16).max:
            raise RuntimeError("source indices do not fit in the common int16 mapping arrays")

        latitudes.append(latitude)
        longitudes.append(longitude)
        mappings_u.append(target_u.astype(np.int16, copy=False))
        mappings_v.append(target_v.astype(np.int16, copy=False))

    if len(latitudes) != n_tile:
        raise RuntimeError(
            f"could only sample {len(latitudes)} source-mappable tiles inside "
            f"{REGION_BOUNDS_TEXT} after {maximum_attempts} attempts"
        )

    return (
        np.asarray(latitudes, dtype=np.float64),
        np.asarray(longitudes, dtype=np.float64),
        np.stack(mappings_u),
        np.stack(mappings_v),
    )


# Zarr creation and validation
# ============================

def open_zarr_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="a", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="a")


def create_array(
    group,
    name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype,
    fill_value,
    compressor,
):
    kwargs = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "fill_value": fill_value,
        "compressor": compressor,
        "overwrite": False,
    }
    if hasattr(group, "create_array"):
        return group.create_array(name, **kwargs)
    return group.create_dataset(name, **kwargs)


def tile_group_name(index: int) -> str:
    return f"tile_{index}"


def ordered_tile_groups(root, expected_count: int | None = None) -> list:
    """Return tile_0, tile_1, ... and reject gaps or unrelated groups."""
    names = set(root.group_keys())
    indices: list[int] = []
    for name in names:
        match = re.fullmatch(r"tile_(\d+)", name)
        if match is None:
            raise RuntimeError(f"unexpected Zarr group at root: {name!r}")
        indices.append(int(match.group(1)))
    indices.sort()
    if indices != list(range(len(indices))):
        raise RuntimeError(f"tile groups must be contiguous from tile_0: {sorted(names)}")
    if expected_count is not None and len(indices) != expected_count:
        raise RuntimeError(
            f"existing Zarr has {len(indices)} tile groups, expected {expected_count}"
        )
    return [root[tile_group_name(index)] for index in indices]


def _validate_mapping_result(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_tile: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latitude, longitude, target_u, target_v = map(np.asarray, values)
    if latitude.shape != (n_tile,) or longitude.shape != (n_tile,):
        raise RuntimeError("mapping builder returned incorrect centre-coordinate shapes")
    expected_mapping_shape = (n_tile, TILE_PIXELS, TILE_PIXELS)
    if target_u.shape != expected_mapping_shape or target_v.shape != expected_mapping_shape:
        raise RuntimeError(
            "mapping builder returned incorrect mapping shapes: "
            f"{target_u.shape}, {target_v.shape}; expected {expected_mapping_shape}"
        )
    if not (
        np.isfinite(latitude).all()
        and np.isfinite(longitude).all()
        and (latitude >= REGION_MIN_LAT).all()
        and (latitude <= REGION_MAX_LAT).all()
        and (longitude >= REGION_MIN_LON).all()
        and (longitude <= REGION_MAX_LON).all()
    ):
        raise RuntimeError("mapping builder returned a centre outside the region rectangle")
    return (
        latitude.astype(np.float64, copy=False),
        longitude.astype(np.float64, copy=False),
        target_u.astype(np.int16, copy=False),
        target_v.astype(np.int16, copy=False),
    )


def initialise_or_validate_store(
    output: Path,
    timestamps: np.ndarray,
    channels: Sequence[str],
    n_tile: int,
    mapping_builder: MappingBuilder,
    root_attributes: Mapping[str, Any] | None = None,
):
    """Create or validate the exact common per-tile Zarr structure."""
    root = open_zarr_group(output)
    required = {
        "images",
        "pixel_valid",
        "frame_valid",
        "time_utc",
        "channel",
        "latitude",
        "longitude",
        "target_u",
        "target_v",
    }
    if set(root.array_keys()):
        raise RuntimeError(
            "existing Zarr has arrays at its root; choose a new --output for "
            "the tile_N group layout"
        )

    existing_tiles = ordered_tile_groups(root)
    if existing_tiles:
        if root.attrs.get("layout") != LAYOUT:
            raise RuntimeError(
                f"existing output layout is {root.attrs.get('layout')!r}, "
                f"expected {LAYOUT!r}"
            )
        if len(existing_tiles) != n_tile:
            raise RuntimeError(
                f"existing Zarr has {len(existing_tiles)} tiles, expected {n_tile}"
            )
        image_shape = (len(timestamps), len(channels), TILE_PIXELS, TILE_PIXELS)
        expected_shapes = {
            "images": image_shape,
            "pixel_valid": image_shape,
            "frame_valid": (len(timestamps),),
            "time_utc": (len(timestamps),),
            "channel": (len(channels),),
            "latitude": (1,),
            "longitude": (1,),
            "target_u": (TILE_PIXELS, TILE_PIXELS),
            "target_v": (TILE_PIXELS, TILE_PIXELS),
        }
        for tile_index, tile in enumerate(existing_tiles):
            arrays = set(tile.array_keys())
            if arrays != required:
                raise RuntimeError(
                    f"tile_{tile_index} arrays mismatch; "
                    f"missing={sorted(required-arrays)}, extra={sorted(arrays-required)}"
                )
            wrong_shapes = {
                name: (tuple(tile[name].shape), expected)
                for name, expected in expected_shapes.items()
                if tuple(tile[name].shape) != expected
            }
            if wrong_shapes:
                raise RuntimeError(
                    f"tile_{tile_index} has inconsistent shapes: {wrong_shapes}"
                )
            if np.dtype(tile["images"].dtype) != np.dtype(np.int16):
                raise RuntimeError(f"tile_{tile_index}/images must have dtype int16")
            if int(tile["images"].fill_value) != int(IMAGE_FILL_VALUE):
                raise RuntimeError(
                    f"tile_{tile_index}/images fill value is "
                    f"{tile['images'].fill_value!r}, expected -1"
                )
            if not np.array_equal(
                np.asarray(tile["time_utc"][:], dtype=np.int64), timestamps
            ):
                raise RuntimeError(
                    f"tile_{tile_index}/time_utc does not match --start/--end"
                )
            stored_channels = [str(value) for value in tile["channel"][:].tolist()]
            if stored_channels != list(channels):
                raise RuntimeError(
                    f"tile_{tile_index} channels {stored_channels} do not match "
                    f"{list(channels)}"
                )
        return root

    latitude, longitude, target_u, target_v = _validate_mapping_result(
        mapping_builder(), n_tile
    )
    n_time = len(timestamps)
    n_channel = len(channels)
    time_chunk = min(TIME_CHUNK, n_time)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    mask_compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    channel_width = max(1, max(len(name) for name in channels))
    channel_dtype = np.dtype(f"<U{channel_width}")
    encoded_channels = np.asarray(channels, dtype=channel_dtype)

    for tile_index in range(n_tile):
        tile = root.create_group(tile_group_name(tile_index))
        images = create_array(
            tile,
            "images",
            (n_time, n_channel, TILE_PIXELS, TILE_PIXELS),
            (time_chunk, n_channel, TILE_PIXELS, TILE_PIXELS),
            np.int16,
            IMAGE_FILL_VALUE,
            compressor,
        )
        pixel_valid = create_array(
            tile,
            "pixel_valid",
            (n_time, n_channel, TILE_PIXELS, TILE_PIXELS),
            (time_chunk, n_channel, TILE_PIXELS, TILE_PIXELS),
            np.bool_,
            False,
            mask_compressor,
        )
        frame_valid = create_array(
            tile,
            "frame_valid",
            (n_time,),
            (min(1_440, n_time),),
            np.bool_,
            False,
            mask_compressor,
        )
        time_utc = create_array(
            tile,
            "time_utc",
            (n_time,),
            (min(1_440, n_time),),
            np.int64,
            0,
            compressor,
        )
        channel = create_array(
            tile,
            "channel",
            (n_channel,),
            (n_channel,),
            channel_dtype,
            "",
            None,
        )
        latitude_array = create_array(
            tile, "latitude", (1,), (1,), np.float64, np.nan, compressor
        )
        longitude_array = create_array(
            tile, "longitude", (1,), (1,), np.float64, np.nan, compressor
        )
        target_u_array = create_array(
            tile,
            "target_u",
            (TILE_PIXELS, TILE_PIXELS),
            (TILE_PIXELS, TILE_PIXELS),
            np.int16,
            -1,
            compressor,
        )
        target_v_array = create_array(
            tile,
            "target_v",
            (TILE_PIXELS, TILE_PIXELS),
            (TILE_PIXELS, TILE_PIXELS),
            np.int16,
            -1,
            compressor,
        )

        time_utc[:] = timestamps
        channel[:] = encoded_channels
        latitude_array[:] = latitude[tile_index : tile_index + 1]
        longitude_array[:] = longitude[tile_index : tile_index + 1]
        target_u_array[:] = target_u[tile_index]
        target_v_array[:] = target_v[tile_index]

        images.attrs["_ARRAY_DIMENSIONS"] = ["time", "channel", "y", "x"]
        pixel_valid.attrs["_ARRAY_DIMENSIONS"] = ["time", "channel", "y", "x"]
        frame_valid.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        time_utc.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        channel.attrs["_ARRAY_DIMENSIONS"] = ["channel"]
        latitude_array.attrs["_ARRAY_DIMENSIONS"] = ["tile"]
        longitude_array.attrs["_ARRAY_DIMENSIONS"] = ["tile"]
        target_u_array.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
        target_v_array.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
        tile.attrs["tile_index"] = tile_index

    root.attrs.update(
        {
            "layout": LAYOUT,
            "n_tile": n_tile,
            "tile_width_m": TILE_WIDTH_M,
            "target_pixel_m": TARGET_PIXEL_M,
            "target_projection": "per-tile WGS84 azimuthal equidistant (AEQD)",
            "tile_center_region": REGION_BOUNDS_TEXT,
            "tile_center_longitude_bounds": [REGION_MIN_LON, REGION_MAX_LON],
            "tile_center_latitude_bounds": [REGION_MIN_LAT, REGION_MAX_LAT],
        }
    )
    if root_attributes:
        root.attrs.update(dict(root_attributes))
    return root


# Buffered processing and commit logic
# ===========================================

def process_frames(
    root,
    timeline: Sequence[datetime],
    source_files: Mapping[datetime, Any],
    channels: Sequence[str],
    frame_reader: FrameReader,
    fail_fast: bool,
    source_label: Callable[[Any], str] = str,
) -> tuple[int, int, int]:
    """Read source frames through a callback and commit complete time chunks."""
    tiles = ordered_tile_groups(root)
    if not tiles:
        raise RuntimeError("Zarr contains no tile_N groups")
    target_u = np.stack(
        [np.asarray(tile["target_u"][:], dtype=np.int16) for tile in tiles]
    )
    target_v = np.stack(
        [np.asarray(tile["target_v"][:], dtype=np.int16) for tile in tiles]
    )
    completed_by_tile = np.stack(
        [np.asarray(tile["frame_valid"][:], dtype=np.bool_) for tile in tiles]
    )
    # There is intentionally no separate frame_processed marker. A readable
    # all-invalid tile therefore remains incomplete and is retried on rerun.
    completed = np.all(completed_by_tile, axis=0)
    time_to_index = {timestamp: index for index, timestamp in enumerate(timeline)}

    pending_by_block: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    skipped = 0
    for timestamp, source in source_files.items():
        index = time_to_index[timestamp]
        if completed[index]:
            skipped += 1
            continue
        pending_by_block[index // TIME_CHUNK].append((index, source))

    written = 0
    failed = 0
    expected_frame_shape = (
        len(tiles),
        len(channels),
        TILE_PIXELS,
        TILE_PIXELS,
    )
    for block_number in sorted(pending_by_block):
        start = block_number * TIME_CHUNK
        end = min(start + TIME_CHUNK, len(timeline))
        block_length = end - start
        block_frame_valid = completed_by_tile[:, start:end].copy()
        block_images = np.full(
            (
                len(tiles),
                block_length,
                len(channels),
                TILE_PIXELS,
                TILE_PIXELS,
            ),
            IMAGE_FILL_VALUE,
            dtype=np.int16,
        )
        block_valid = np.zeros(block_images.shape, dtype=np.bool_)

        for tile_index, tile in enumerate(tiles):
            preserve = block_frame_valid[tile_index]
            if preserve.any():
                stored_images = np.asarray(tile["images"][start:end])
                stored_valid = np.asarray(tile["pixel_valid"][start:end])
                block_images[tile_index, preserve] = stored_images[preserve]
                block_valid[tile_index, preserve] = stored_valid[preserve]

        successful_indices: list[int] = []
        dirty_indices: list[int] = []
        for absolute_index, source in sorted(
            pending_by_block[block_number], key=lambda item: item[0]
        ):
            offset = absolute_index - start
            timestamp = timeline[absolute_index]
            try:
                frame_images, frame_pixel_valid = frame_reader(
                    source, channels, target_u, target_v
                )
                frame_images = np.asarray(frame_images)
                frame_pixel_valid = np.asarray(frame_pixel_valid)
                if frame_images.shape != expected_frame_shape:
                    raise RuntimeError(
                        f"frame reader returned images shape {frame_images.shape}; "
                        f"expected {expected_frame_shape}"
                    )
                if frame_images.dtype != np.int16:
                    raise RuntimeError(
                        f"frame reader returned images dtype {frame_images.dtype}; "
                        "expected int16"
                    )
                if frame_pixel_valid.shape != expected_frame_shape:
                    raise RuntimeError(
                        "frame reader returned pixel_valid shape "
                        f"{frame_pixel_valid.shape}; expected {expected_frame_shape}"
                    )
                frame_pixel_valid = frame_pixel_valid.astype(np.bool_, copy=False)
            except Exception as error:
                failed += 1
                block_images[:, offset] = IMAGE_FILL_VALUE
                block_valid[:, offset] = False
                block_frame_valid[:, offset] = False
                dirty_indices.append(absolute_index)
                print(
                    f"ERROR {timestamp.isoformat(sep=' ')} UTC "
                    f"{source_label(source)}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if fail_fast:
                    raise
                continue

            block_images[:, offset] = frame_images
            block_valid[:, offset] = frame_pixel_valid
            block_frame_valid[:, offset] = np.any(
                frame_pixel_valid, axis=(1, 2, 3)
            )
            successful_indices.append(absolute_index)
            dirty_indices.append(absolute_index)
            print(
                f"READ  {timestamp.isoformat(sep=' ')} UTC {source_label(source)}",
                flush=True,
            )

        if dirty_indices:
            # frame_valid is the commit marker and is written last
            for tile_index, tile in enumerate(tiles):
                tile["images"][start:end] = block_images[tile_index]
                tile["pixel_valid"][start:end] = block_valid[tile_index]
            for tile_index, tile in enumerate(tiles):
                tile["frame_valid"][start:end] = block_frame_valid[tile_index]
            completed_by_tile[:, start:end] = block_frame_valid
            completed[start:end] = np.all(block_frame_valid, axis=0)
            written += len(successful_indices)
            print(
                f"WROTE time block [{start}, {end}): "
                f"{len(successful_indices)} successed, "
                f"{len(dirty_indices) - len(successful_indices)} failed",
                flush=True,
            )

    return written, skipped, failed


__all__ = [
    "FRAME_INTERVAL",
    "IMAGE_FILL_VALUE",
    "LAYOUT",
    "REGION_MAX_LAT",
    "REGION_MAX_LON",
    "REGION_MIN_LAT",
    "REGION_MIN_LON",
    "TARGET_PIXEL_M",
    "TILE_PIXELS",
    "TILE_WIDTH_M",
    "TIME_CHUNK",
    "initialise_or_validate_store",
    "open_zarr_group",
    "ordered_tile_groups",
    "parse_utc",
    "process_frames",
    "regular_times",
    "sample_tiles",
    "target_lon_lat",
    "tile_group_name",
    "unix_seconds",
]
