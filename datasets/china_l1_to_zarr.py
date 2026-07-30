"""Convert FY-4B AGRI 500 m Level-1 HDF5 frames to tiled Zarr.

The source is a full-disk geostationary image.  Navigation is derived from
the HDF root attributes; valid 12-bit digital numbers are preserved exactly
and the source fill value is represented by the common Zarr ``-1`` fill.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import zarr
from pyproj import CRS, Transformer

from china_region import CHINA_REGION_POLYGONS
from zarr_common import (
    DEFAULT_INTERVAL,
    IMAGE_FILL_VALUE,
    REGION_MAX_LAT,
    REGION_MAX_LON,
    REGION_MIN_LAT,
    REGION_MIN_LON,
    TILE_PIXELS,
    initialise_or_validate_store,
    ordered_tile_groups,
    process_frames,
    sample_tiles,
    target_lon_lat,
    unix_seconds,
)


DEFAULT_PATH = Path("trial")
DEFAULT_TILE = 100
DEFAULT_OUTPUT = Path("trial_l1.zarr")
DEFAULT_START = "2025-01-01 05:00"
DEFAULT_END = "2025-01-01 06:00"
DEFAULT_CHANNELS = ("NOMChannel02",)
EXPECTED_SOURCE_SHAPE = (21_984, 21_984)
MICRORADIANS_TO_RADIANS = 1.0e-6

FILENAME_RE = re.compile(
    r"^FY4B-_AGRI--_N_DISK_(?P<sub_lon>\d{4})E_"
    r"L1-_FDI-_MULT_NOM_(?P<start>\d{14})_(?P<end>\d{14})_"
    r"0500M_V(?P<version>\d{4})\.HDF$",
    re.IGNORECASE,
)
CHANNEL_RE = re.compile(
    r"^(?:NOMCHANNEL(?P<nom>\d{2})|CHANNEL(?P<channel>\d{2})|C(?P<c>\d{2}))$",
    re.IGNORECASE,
)


# Time and source discovery
# =========================

def parse_utc(value: str) -> datetime:
    """Parse a timezone-aware or naive UTC timestamp aligned to a minute."""
    text = value.strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1].strip()
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    if parsed.second or parsed.microsecond:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be aligned to an exact UTC minute"
        )
    return parsed


def regular_times(
    start: datetime,
    end: datetime,
    interval_minutes: int,
) -> list[datetime]:
    """Return an inclusive, regular timeline at the requested minute cadence."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    step = timedelta(minutes=interval_minutes)
    elapsed_seconds = int((end - start).total_seconds())
    step_seconds = int(step.total_seconds())
    if elapsed_seconds % step_seconds:
        raise ValueError(
            "end must be an integer number of --interval-minutes after start"
        )
    return [
        start + index * step
        for index in range(elapsed_seconds // step_seconds + 1)
    ]


def datetime_from_filename(path: Path) -> datetime | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match["start"], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def source_candidates(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".hdf":
            raise ValueError(f"input file is not an HDF file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {path}")
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() == ".hdf"
    )


def scan_source_files(
    path: Path,
    start: datetime,
    end: datetime,
) -> tuple[dict[datetime, Path], Path | None]:
    """Index canonical FY-4B AGRI HDF files by scan-start timestamp."""
    indexed: dict[datetime, Path] = {}
    representative: Path | None = None
    candidates = source_candidates(path)
    for candidate in candidates:
        timestamp = datetime_from_filename(candidate)
        if timestamp is None:
            if path.is_file():
                raise ValueError(
                    f"input filename does not match the FY-4B AGRI convention: "
                    f"{candidate.name}"
                )
            continue
        if representative is None:
            representative = candidate
        if not start <= timestamp <= end:
            continue
        previous = indexed.get(timestamp)
        if previous is not None:
            raise RuntimeError(
                f"multiple source files resolve to {timestamp}: "
                f"{previous} and {candidate}"
            )
        indexed[timestamp] = candidate
    return indexed, representative


# HDF metadata and FY-4 navigation
# ================================

def attribute_scalar(container, name: str):
    if name not in container.attrs:
        raise RuntimeError(f"{container.name or '/'} lacks attribute {name!r}")
    values = np.asarray(container.attrs[name]).reshape(-1)
    if values.size != 1:
        raise RuntimeError(
            f"{container.name or '/'} attribute {name!r} is not scalar"
        )
    return values[0]


def attribute_text(container, name: str) -> str:
    value = attribute_scalar(container, name)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def canonical_channel(name: str) -> str:
    token = str(name).replace("_", "").replace("-", "")
    match = CHANNEL_RE.fullmatch(token)
    if match is None:
        raise ValueError(
            f"unsupported FY-4 channel {name!r}; expected a name such as "
            "NOMChannel02, Channel02, or C02"
        )
    number = next(value for value in match.groupdict().values() if value is not None)
    return f"NOMChannel{number}"


def channel_dataset_path(channel: str) -> str:
    return f"/Data/{canonical_channel(channel)}"


@dataclass(frozen=True)
class Fy4Grid:
    shape: tuple[int, int]
    longitude_of_origin: float
    satellite_height_m: float
    semimajor_axis_m: float
    semiminor_axis_m: float
    column_step_m: float
    line_step_m: float
    sweep_axis: str = "y"

    @property
    def crs(self) -> CRS:
        return CRS.from_proj4(
            "+proj=geos "
            f"+lon_0={self.longitude_of_origin:.12f} "
            f"+h={self.satellite_height_m:.6f} "
            f"+a={self.semimajor_axis_m:.6f} "
            f"+b={self.semiminor_axis_m:.6f} "
            f"+sweep={self.sweep_axis} +units=m +no_defs"
        )

    @property
    def area_extent(self) -> tuple[float, float, float, float]:
        height, width = self.shape
        half_width = width * self.column_step_m / 2.0
        half_height = height * self.line_step_m / 2.0
        return (-half_width, -half_height, half_width, half_height)


def read_grid(path: Path, channels: Sequence[str]) -> Fy4Grid:
    with h5py.File(path, "r") as source:
        satellite = attribute_text(source, "Satellite Name")
        sensor = attribute_text(source, "Sensor Name")
        observation_type = attribute_text(source, "OBIType")
        if satellite != "FY-4B" or sensor != "AGRI" or observation_type != "DISK":
            raise RuntimeError(
                f"{path} is {satellite}/{sensor}/{observation_type}, expected "
                "FY-4B/AGRI/DISK"
            )

        shapes: set[tuple[int, int]] = set()
        for channel in channels:
            dataset_path = channel_dataset_path(channel)
            if dataset_path not in source:
                raise KeyError(f"{path} lacks requested dataset {dataset_path}")
            dataset = source[dataset_path]
            if dataset.ndim != 2:
                raise RuntimeError(f"{path}:{dataset_path} is not a 2-D image")
            if dataset.dtype.kind != "u" or dataset.dtype.itemsize != 2:
                raise RuntimeError(
                    f"{path}:{dataset_path} has dtype {dataset.dtype}; "
                    "expected unsigned 16-bit DN"
                )
            shapes.add(tuple(dataset.shape))
        if len(shapes) != 1:
            raise RuntimeError(
                "requested channels do not share one source grid: "
                f"{sorted(shapes)}"
            )
        shape = shapes.pop()
        if shape != EXPECTED_SOURCE_SHAPE:
            raise RuntimeError(
                f"{path} source shape is {shape}, expected {EXPECTED_SOURCE_SHAPE}"
            )

        registered_height = int(round(float(attribute_scalar(source, "RegLength"))))
        registered_width = int(round(float(attribute_scalar(source, "RegWidth"))))
        if (registered_height, registered_width) != shape:
            raise RuntimeError(
                "RegLength/RegWidth disagree with the image shape: "
                f"{(registered_height, registered_width)} versus {shape}"
            )

        satellite_height = float(attribute_scalar(source, "NOMSatHeight"))
        sampling_angle = float(attribute_scalar(source, "dSamplingAngle"))
        stepping_angle = float(attribute_scalar(source, "dSteppingAngle"))
        if satellite_height <= 0 or sampling_angle <= 0 or stepping_angle <= 0:
            raise RuntimeError("invalid FY-4 satellite height or angular sampling")

        # FY-4 stores the sampling and stepping angles in microradians.  PROJ's
        # geostationary coordinates are satellite height multiplied by scan angle.
        return Fy4Grid(
            shape=shape,
            longitude_of_origin=float(attribute_scalar(source, "NOMCenterLon")),
            satellite_height_m=satellite_height,
            semimajor_axis_m=float(
                attribute_scalar(source, "Semimajor axis of ellipsoid")
            ),
            semiminor_axis_m=float(
                attribute_scalar(source, "Semiminor axis of ellipsoid")
            ),
            column_step_m=(
                satellite_height * sampling_angle * MICRORADIANS_TO_RADIANS
            ),
            line_step_m=(
                satellite_height * stepping_angle * MICRORADIANS_TO_RADIANS
            ),
        )


def mapping_for_center(
    center_lat: float,
    center_lon: float,
    grid: Fy4Grid,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = target_lon_lat(center_lat, center_lon)
    source_x, source_y = transformer.transform(longitude, latitude)
    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)

    height, width = grid.shape
    target_u = np.rint(
        source_x / grid.column_step_m + (width - 1) / 2.0
    )
    target_v = np.rint(
        (height - 1) / 2.0 - source_y / grid.line_step_m
    )
    invalid = ~np.isfinite(target_u) | ~np.isfinite(target_v)
    # Replace non-finite coordinates before converting to an integer dtype.
    target_u = np.where(invalid, -1, target_u).astype(np.int32)
    target_v = np.where(invalid, -1, target_v).astype(np.int32)
    invalid |= target_u < 0
    invalid |= target_v < 0
    invalid |= target_u >= width
    invalid |= target_v >= height
    if invalid.any():
        raise ValueError("target footprint extends outside the FY-4B full-disk grid")
    return target_u.astype(np.int16), target_v.astype(np.int16)


def build_tile_mappings(
    n_tile: int,
    seed: int,
    grid: Fy4Grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), grid.crs, always_xy=True
    )

    def mapper(center_lat: float, center_lon: float):
        return mapping_for_center(center_lat, center_lon, grid, transformer)

    return sample_tiles(n_tile, seed, mapper)


# HDF sampling and source validity
# ================================

def optional_attribute(dataset: h5py.Dataset, name: str) -> np.ndarray | None:
    if name not in dataset.attrs:
        return None
    return np.asarray(dataset.attrs[name]).reshape(-1)


def validity_mask(dataset: h5py.Dataset, values: np.ndarray) -> np.ndarray:
    """Interpret FY-4 ``valid_range`` and nonstandard ``FillValue`` metadata."""
    valid = np.ones(values.shape, dtype=np.bool_)
    valid_range = optional_attribute(dataset, "valid_range")
    if valid_range is not None:
        if valid_range.size != 2:
            raise RuntimeError(f"{dataset.name} valid_range must contain two values")
        valid &= values >= valid_range[0]
        valid &= values <= valid_range[1]
    for name in ("FillValue", "_FillValue", "missing_value"):
        fill = optional_attribute(dataset, name)
        if fill is not None:
            if fill.size != 1:
                raise RuntimeError(f"{dataset.name} {name} must be scalar")
            valid &= values != fill[0]
    return valid


def sample_dataset(
    dataset: h5py.Dataset,
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> np.ndarray:
    """Sample every target while decompressing each intersected chunk once."""
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
        sampled[indices] = source_window[
            selected_v - v_min,
            selected_u - u_min,
        ]
    return sampled.reshape(target_u.shape)


def read_frame_once(
    path: Path,
    channels: Sequence[str],
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Open one FY-4 HDF once and sample all requested channels and tiles."""
    n_tile = target_u.shape[0]
    result = np.full(
        (n_tile, len(channels), TILE_PIXELS, TILE_PIXELS),
        IMAGE_FILL_VALUE,
        dtype=np.int16,
    )
    valid_result = np.zeros(result.shape, dtype=np.bool_)

    u_min, u_max = int(target_u.min()), int(target_u.max())
    v_min, v_max = int(target_v.min()), int(target_v.max())
    if u_min < 0 or v_min < 0:
        raise RuntimeError("target_u/target_v contains a negative source index")

    with h5py.File(path, "r") as source:
        for channel_index, channel in enumerate(channels):
            dataset_path = channel_dataset_path(channel)
            if dataset_path not in source:
                raise KeyError(f"{path} lacks requested dataset {dataset_path}")
            dataset = source[dataset_path]
            if dataset.ndim != 2:
                raise RuntimeError(f"{path}:{dataset_path} is not a 2-D image")
            if dataset.dtype.kind != "u" or dataset.dtype.itemsize != 2:
                raise RuntimeError(
                    f"{path}:{dataset_path} has dtype {dataset.dtype}; "
                    "expected unsigned 16-bit DN"
                )
            if u_max >= dataset.shape[1] or v_max >= dataset.shape[0]:
                raise RuntimeError(f"target mapping is outside {path}:{dataset_path}")

            sampled = sample_dataset(dataset, target_u, target_v)
            valid = validity_mask(dataset, sampled)
            if valid.any() and int(sampled[valid].max()) > np.iinfo(np.int16).max:
                raise RuntimeError(
                    f"{path}:{dataset_path} has a valid value that does not fit int16"
                )
            destination = result[:, channel_index]
            destination[valid] = sampled[valid].astype(np.int16, copy=False)
            valid_result[:, channel_index] = valid
    return result, valid_result


def source_metadata(path: Path, grid: Fy4Grid) -> dict[str, object]:
    metadata: dict[str, object] = {
        "product_level": "L1",
        "source_format": "FY-4B AGRI 500 m full-disk HDF5",
        "source_projection": "geostationary, sweep=y",
        "source_grid_shape": list(grid.shape),
        "source_projection_longitude_degrees": grid.longitude_of_origin,
        "source_satellite_height_m": grid.satellite_height_m,
        "source_column_step_m": grid.column_step_m,
        "source_line_step_m": grid.line_step_m,
        "source_area_extent_m": list(grid.area_extent),
        "source_value_meaning": "raw unsigned 12-bit digital number (DN)",
        "source_fill_value": 65535,
        "stored_fill_value": int(IMAGE_FILL_VALUE),
        "pixel_valid_rule": "valid_range and FillValue attributes from each HDF dataset",
        "resampling": "nearest source pixel",
        "source_filename_convention": FILENAME_RE.pattern,
    }
    with h5py.File(path, "r") as source:
        metadata.update(
            {
                "satellite": attribute_text(source, "Satellite Name"),
                "sensor": attribute_text(source, "Sensor Name"),
                "observing_beginning_date": attribute_text(
                    source, "Observing Beginning Date"
                ),
                "observing_beginning_time": attribute_text(
                    source, "Observing Beginning Time"
                ),
            }
        )
        coefficient_path = "/Calibration/CALIBRATION_COEF(SCALE+OFFSET)"
        if coefficient_path in source:
            coefficient = np.asarray(source[coefficient_path][:], dtype=np.float64)
            if coefficient.shape == (1, 2):
                metadata["calibration_scale"] = float(coefficient[0, 0])
                metadata["calibration_offset"] = float(coefficient[0, 1])
                metadata["calibration_formula"] = (
                    "max(0, calibration_scale * DN + calibration_offset)"
                )
    return metadata


# Optional whole-frame tile-centre diagnostic
# ===========================================

def open_zarr_readonly(path: Path):
    try:
        return zarr.open_group(str(path), mode="r", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def select_overview_source(path: Path) -> Path:
    """Select one deterministic canonical FY-4B AGRI source frame."""
    for candidate in source_candidates(path):
        if datetime_from_filename(candidate) is not None:
            return candidate
    raise RuntimeError(f"no canonical FY-4B AGRI HDF file was found under {path}")


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


def source_pixel_coordinates(
    longitude: np.ndarray,
    latitude: np.ndarray,
    grid: Fy4Grid,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project geographic positions into fractional FY-4B source pixels."""
    source_x, source_y = transformer.transform(longitude, latitude)
    source_u = (
        np.asarray(source_x, dtype=np.float64) / grid.column_step_m
        + (grid.shape[1] - 1) / 2.0
    )
    source_v = (
        (grid.shape[0] - 1) / 2.0
        - np.asarray(source_y, dtype=np.float64) / grid.line_step_m
    )
    valid = (
        np.isfinite(source_u)
        & np.isfinite(source_v)
        & (source_u >= 0)
        & (source_v >= 0)
        & (source_u < grid.shape[1])
        & (source_v < grid.shape[0])
    )
    return source_u, source_v, valid


def region_boundary(samples_per_edge: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return a densely sampled closed tile-centre bounds rectangle."""
    west_to_east = np.linspace(REGION_MIN_LON, REGION_MAX_LON, samples_per_edge)
    south_to_north = np.linspace(REGION_MIN_LAT, REGION_MAX_LAT, samples_per_edge)
    longitude = np.concatenate(
        (
            west_to_east,
            np.full(samples_per_edge, REGION_MAX_LON),
            west_to_east[::-1],
            np.full(samples_per_edge, REGION_MIN_LON),
        )
    )
    latitude = np.concatenate(
        (
            np.full(samples_per_edge, REGION_MIN_LAT),
            south_to_north,
            np.full(samples_per_edge, REGION_MAX_LAT),
            south_to_north[::-1],
        )
    )
    return longitude, latitude


def draw_all_tile_centres(
    data_path: str | Path = DEFAULT_PATH,
    zarr_path: str | Path = DEFAULT_OUTPUT,
    channel: str = DEFAULT_CHANNELS[0],
    png_path: str | Path | None = None,
    show: bool = False,
    maximum_display_pixels: int = 4_000_000,
) -> dict[str, object]:
    """Draw a downsampled L1 full disk with China and all Zarr tile centres."""
    if png_path is None and not show:
        raise ValueError("set png_path or show=True")
    if png_path is not None and show:
        raise ValueError("png_path and show=True are mutually exclusive")
    if maximum_display_pixels <= 0:
        raise ValueError("maximum_display_pixels must be positive")

    input_path = Path(data_path).expanduser().resolve()
    store_path = Path(zarr_path).expanduser().resolve()
    source_path = select_overview_source(input_path)
    normalized_channel = canonical_channel(channel)
    dataset_path = channel_dataset_path(normalized_channel)
    grid = read_grid(source_path, (normalized_channel,))

    root = open_zarr_readonly(store_path)
    tiles = ordered_tile_groups(root)
    if not tiles:
        raise RuntimeError(f"{store_path} contains no tile groups")
    stored_channels = [str(value) for value in tiles[0]["channel"][:].tolist()]
    if normalized_channel not in stored_channels:
        raise RuntimeError(
            f"{store_path} channels {stored_channels} do not contain "
            f"{normalized_channel!r}"
        )
    tile_longitude = np.asarray(
        [float(tile["longitude"][0]) for tile in tiles], dtype=np.float64
    )
    tile_latitude = np.asarray(
        [float(tile["latitude"][0]) for tile in tiles], dtype=np.float64
    )

    with h5py.File(source_path, "r") as source:
        dataset = source[dataset_path]
        stride = max(
            1,
            int(np.ceil(np.sqrt(dataset.size / maximum_display_pixels))),
        )
        image = np.asarray(dataset[::stride, ::stride])
        image_valid = validity_mask(dataset, image)

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), grid.crs, always_xy=True
    )
    tile_u, tile_v, tile_visible = source_pixel_coordinates(
        tile_longitude, tile_latitude, grid, transformer
    )
    if not tile_visible.all():
        raise RuntimeError("one or more stored tile centres are outside the L1 disk")

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
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    plotted = axis.imshow(
        displayed,
        cmap=cmap,
        vmin=low,
        vmax=high,
        origin="upper",
        extent=(-0.5, grid.shape[1] - 0.5, grid.shape[0] - 0.5, -0.5),
        interpolation="nearest",
        aspect="equal",
    )

    for polygon_index, polygon in enumerate(CHINA_REGION_POLYGONS):
        closed = np.vstack((polygon, polygon[0]))
        polygon_u, polygon_v, visible = source_pixel_coordinates(
            closed[:, 0], closed[:, 1], grid, transformer
        )
        axis.plot(
            polygon_u[visible],
            polygon_v[visible],
            color="#ffd54f",
            linewidth=1.5,
            label="China region" if polygon_index == 0 else None,
            zorder=3,
        )

    boundary_longitude, boundary_latitude = region_boundary()
    boundary_u, boundary_v, boundary_visible = source_pixel_coordinates(
        boundary_longitude, boundary_latitude, grid, transformer
    )
    axis.plot(
        boundary_u[boundary_visible],
        boundary_v[boundary_visible],
        color="#00e5ff",
        linewidth=1.8,
        label="Tile-centre bounds",
        zorder=3,
    )
    axis.scatter(
        tile_u,
        tile_v,
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
    axis.set_title(
        f"All tile centres | {time_text} | {normalized_channel}"
    )
    axis.set_xlabel("FY-4B source u (column)")
    axis.set_ylabel("FY-4B source v (row)")
    axis.legend(loc="best")
    figure.colorbar(plotted, ax=axis, shrink=0.82).set_label(
        f"{normalized_channel} raw DN"
    )

    destination: Path | None = None
    if png_path is not None:
        destination = Path(png_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
        print(f"Tile-centres PNG: {destination}")
    else:
        plt.show()
    plt.close(figure)

    usable = image[image_valid]
    statistics: dict[str, object] = {
        "source": source_path,
        "channel": normalized_channel,
        "source_shape": list(grid.shape),
        "display_stride": stride,
        "display_shape": list(image.shape),
        "display_valid_count": int(image_valid.sum()),
        "display_invalid_count": int((~image_valid).sum()),
        "display_valid_min": None if usable.size == 0 else int(usable.min()),
        "display_valid_max": None if usable.size == 0 else int(usable.max()),
        "png": destination,
    }
    print(
        f"Overview source: {source_path}; channel: {normalized_channel}; "
        f"stride: {stride}; displayed valid DN min/max: "
        f"{statistics['display_valid_min']} / {statistics['display_valid_max']}"
    )
    return statistics


# Command-line and Python entry points
# ====================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop FY-4B AGRI 500 m L1 HDF frames into tiled Zarr."
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
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=DEFAULT_INTERVAL,
        help=(
            "timeline interval in minutes "
            f"(default: {DEFAULT_INTERVAL})"
        ),
    )
    parser.add_argument("--n-tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
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
    interval_minutes: int = DEFAULT_INTERVAL,
    n_tile: int = DEFAULT_TILE,
    path: str | Path = DEFAULT_PATH,
    output: str | Path = DEFAULT_OUTPUT,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    seed: int = 42,
    fail_fast: bool = False,
) -> tuple[int, int, int]:
    """Create or resume an FY-4B L1 store.

    Returns ``(written, already_valid, failed)``.
    """
    start_time = parse_utc(start) if isinstance(start, str) else start
    end_time = parse_utc(end) if isinstance(end, str) else end
    if start_time.tzinfo is not None:
        start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)
    if end_time.tzinfo is not None:
        end_time = end_time.astimezone(timezone.utc).replace(tzinfo=None)
    if n_tile <= 0:
        raise ValueError("n_tile must be positive")

    normalized_channels = tuple(canonical_channel(channel) for channel in channels)
    if not normalized_channels or len(set(normalized_channels)) != len(
        normalized_channels
    ):
        raise ValueError("channels must contain one or more unique FY-4 names")

    input_path = Path(path).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = regular_times(start_time, end_time, interval_minutes)
    timestamps = unix_seconds(timeline)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(
        f"Range:  {start_time.isoformat(sep=' ')} to "
        f"{end_time.isoformat(sep=' ')} UTC "
        f"({len(timeline)} slots at {interval_minutes}-minute cadence)"
    )
    print(f"Tiles:  {n_tile}; channels: {', '.join(normalized_channels)}")

    source_files, representative = scan_source_files(
        input_path, start_time, end_time
    )
    if representative is None:
        raise RuntimeError(
            f"no canonical FY-4B AGRI HDF file was found under {input_path}"
        )
    grid = read_grid(representative, normalized_channels)
    print(
        f"Source grid: {grid.shape[1]} x {grid.shape[0]}, "
        f"{grid.column_step_m:.3f} x {grid.line_step_m:.3f} m, "
        f"sub-satellite longitude {grid.longitude_of_origin:g}E"
    )
    print(f"Found {len(source_files)} source frames inside the requested range")

    attributes = source_metadata(representative, grid)
    attributes["frame_interval_minutes"] = interval_minutes
    root = initialise_or_validate_store(
        output_path,
        timestamps,
        normalized_channels,
        n_tile,
        lambda: build_tile_mappings(n_tile, seed, grid),
        root_attributes=attributes,
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
            interval_minutes=args.interval_minutes,
            n_tile=args.n_tile,
            path=args.data_path,
            output=args.output,
            channels=args.channels,
            seed=args.seed,
            fail_fast=args.fail_fast,
        )
        if args.png_path is not None or args.show:
            draw_all_tile_centres(
                data_path=args.data_path,
                zarr_path=args.output,
                channel=args.channels[0],
                png_path=args.png_path,
                show=args.show,
            )
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
