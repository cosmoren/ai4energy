"""
Crop Night Microphysics full-disk RGB PNGs around one site into a single Zarr.

Reads 5,500 x 5,500 RGB frames from ``~/datasets/night_microphysics``
(``YYYYMMDD_HHMM.png``) and resamples a square ROI centred on one latitude /
longitude onto a local azimuthal-equidistant grid.

The output mirrors ``luoyang_demo/SPMF_preprocessing/luoyang/
himawari_L2_processing_solarfeats.py``:

    images            (time_utc, channel, H, W)  uint8, raw PNG levels 0..255
    frame_valid       (time_utc,)                bool, False for missing frames
    zenith            (time_utc,)                float32 degrees
    azimuth           (time_utc,)                float32 degrees
    day_of_year       (time_utc,)                int32, local solar time
    hour_of_day       (time_utc,)                float32, local solar time
    local_solar_time  (time_utc,)                datetime64[ns]

The time axis is a strict 10-minute grid built from ``--start`` / ``--end``;
timestamps without a readable PNG are stored as black frames.

The store is created up front with its full time axis and then filled block by
block, so peak memory stays at one block and an interrupted run can continue
with ``--resume``. The group attribute ``frames_completed`` is the resume
watermark and ``run_complete`` marks a finished store.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from PIL import Image
from pvlib import solarposition
from pyproj import CRS, Transformer
from satpy.area import get_area_def

from zarr_common import parse_utc, regular_times


DEFAULT_PATH = Path.home() / "datasets" / "night_microphysics"
DEFAULT_OUTPUT = Path("himawari_l1rgb_local.zarr")
DEFAULT_START = "2024-01-01 00:00"
DEFAULT_END = "2024-01-31 23:50"
DEFAULT_SIZE_KM = 500.0
DEFAULT_RESAMPLE_SIZE = 100
CHANNELS = ("R", "G", "B")
SOURCE_AREA_NAME = "himawari_ahi_fes_2km"
SOURCE_SHAPE = (5_500, 5_500)
TIME_CHUNK = 24
DEFAULT_BLOCK_FRAMES = 720  # 5 days at 10 min; a whole number of TIME_CHUNK
DEFAULT_WORKERS = min(16, (os.cpu_count() or 1))


# Solar geometry
# ==============

def utc_to_local_solar_time_pvlib(
    utc_times: pd.DatetimeIndex,
    longitude: float,
) -> pd.DatetimeIndex:
    """Convert UTC to local (apparent) solar time using longitude and pvlib equation of time."""
    if utc_times.tz is not None:
        utc_naive = utc_times.tz_convert("UTC").tz_localize(None)
    else:
        utc_naive = utc_times
    lmst_offset_hours = longitude / 15.0
    dayofyear = utc_naive.dayofyear
    eot_minutes = solarposition.equation_of_time_spencer71(dayofyear)
    return (
        utc_naive
        + pd.Timedelta(hours=lmst_offset_hours)
        + pd.to_timedelta(eot_minutes, unit="m")
    )


# Target grid and full-disk mapping
# =================================

def local_lon_lat(
    center_lat: float,
    center_lon: float,
    size_km: float,
    resample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lon/lat for a square local WGS84 AEQD grid centred on the site."""
    width_m = size_km * 1000.0
    pixel_m = width_m / resample_size
    x = -width_m / 2 + (np.arange(resample_size) + 0.5) * pixel_m
    y = width_m / 2 - (np.arange(resample_size) + 0.5) * pixel_m
    target_x, target_y = np.meshgrid(x, y)
    local_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.12f} +lon_0={center_lon:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_wgs84 = Transformer.from_crs(local_aeqd, CRS.from_epsg(4326), always_xy=True)
    longitude, latitude = to_wgs84.transform(target_x, target_y)
    return np.asarray(longitude), np.asarray(latitude)


def source_area():
    area = get_area_def(SOURCE_AREA_NAME)
    if tuple(area.shape) != SOURCE_SHAPE:
        raise RuntimeError(
            f"Satpy area {SOURCE_AREA_NAME!r} has shape {area.shape}, "
            f"expected {SOURCE_SHAPE}"
        )
    return area


def local_mapping(
    center_lat: float,
    center_lon: float,
    size_km: float,
    resample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest source column / row indices for every ROI pixel."""
    area = source_area()
    longitude, latitude = local_lon_lat(
        center_lat, center_lon, size_km, resample_size
    )
    target_u, target_v = area.get_array_indices_from_lonlat(longitude, latitude)
    masked = np.ma.getmaskarray(target_u) | np.ma.getmaskarray(target_v)
    target_u = np.asarray(np.ma.filled(target_u, -1), dtype=np.int64)
    target_v = np.asarray(np.ma.filled(target_v, -1), dtype=np.int64)
    invalid = masked | (target_u < 0) | (target_v < 0)
    invalid |= target_u >= area.width
    invalid |= target_v >= area.height
    if invalid.any():
        raise ValueError(
            f"ROI around lat={center_lat}, lon={center_lon} with size_km={size_km} "
            "extends outside the AHI full-disk grid"
        )
    return target_u, target_v


# PNG reading
# ===========

class RoiReader:
    """Read one ROI per PNG, decoding only the bounding window of the mapping."""

    def __init__(self, target_u: np.ndarray, target_v: np.ndarray) -> None:
        self.u_min = int(target_u.min())
        self.v_min = int(target_v.min())
        self.u_max = int(target_u.max())
        self.v_max = int(target_v.max())
        self.local_u = target_u - self.u_min
        self.local_v = target_v - self.v_min

    def read(self, path: Path) -> np.ndarray:
        """Return the ROI as (channel, H, W) uint8."""
        with Image.open(path) as image:
            if image.size != (SOURCE_SHAPE[1], SOURCE_SHAPE[0]):
                raise RuntimeError(
                    f"{path} size is {image.size}; expected "
                    f"{(SOURCE_SHAPE[1], SOURCE_SHAPE[0])}"
                )
            if image.mode != "RGB":
                image = image.convert("RGB")
            window = np.asarray(
                image.crop(
                    (self.u_min, self.v_min, self.u_max + 1, self.v_max + 1)
                ),
                dtype=np.uint8,
            )
        if window.ndim != 3 or window.shape[-1] != 3:
            raise RuntimeError(f"{path} crop is not HxWx3 RGB")
        sampled = window[self.local_v, self.local_u]
        return np.moveaxis(sampled, -1, 0)


def expected_png_path(root: Path, timestamp: datetime) -> Path:
    return root / f"{timestamp:%Y%m%d_%H%M}.png"


# Incremental Zarr writing
# ========================

def create_store(
    out_store: Path,
    ts_index: pd.DatetimeIndex,
    lat: float,
    lon: float,
    size_km: float,
    resample_size: int,
) -> None:
    """Write coords, solar features, and empty image/valid arrays for the full timeline."""
    n_time = len(ts_index)
    local_solar = utc_to_local_solar_time_pvlib(ts_index, lon)
    solpos = solarposition.get_solarposition(ts_index, lat, lon)

    ds = xr.Dataset(
        data_vars={
            "images": (
                ("time_utc", "channel", "H", "W"),
                da.zeros(
                    (n_time, len(CHANNELS), resample_size, resample_size),
                    dtype=np.uint8,
                    chunks=(TIME_CHUNK, len(CHANNELS), resample_size, resample_size),
                ),
            ),
            "frame_valid": (
                ("time_utc",),
                da.zeros(n_time, dtype=np.bool_, chunks=(TIME_CHUNK,)),
            ),
            "zenith": (("time_utc",), solpos["zenith"].to_numpy(dtype=np.float32)),
            "azimuth": (("time_utc",), solpos["azimuth"].to_numpy(dtype=np.float32)),
            "day_of_year": (("time_utc",), local_solar.dayofyear.to_numpy(np.int32)),
            "hour_of_day": (
                ("time_utc",),
                (
                    local_solar.hour
                    + local_solar.minute / 60.0
                    + local_solar.second / 3600.0
                ).to_numpy(dtype=np.float32),
            ),
            "local_solar_time": (
                ("time_utc",),
                local_solar.to_numpy(dtype="datetime64[ns]"),
            ),
        },
        coords={
            "time_utc": ts_index.to_numpy(dtype="datetime64[ns]"),
            "channel": list(CHANNELS),
        },
    )
    ds["time_utc"].attrs["timezone"] = "UTC+0"
    ds.attrs["latitude"] = lat
    ds.attrs["longitude"] = lon
    ds.attrs["size_km"] = size_km
    ds.attrs["resample_size"] = resample_size
    ds.attrs["product_level"] = "L1 Night Microphysics RGB"
    ds.attrs["source_format"] = "5500 x 5500 uint8 RGB PNG"
    ds.attrs["source_projection"] = (
        f"Satpy {SOURCE_AREA_NAME} geostationary full-disk grid, nearest source pixel"
    )
    ds.attrs["target_projection"] = "local WGS84 azimuthal equidistant square"
    ds.attrs["value_range"] = "uint8 0-255, raw PNG levels; not calibrated radiance"
    ds.attrs["missing_frame_rule"] = "black image, frame_valid False"
    ds.attrs["frames_completed"] = 0
    ds.attrs["run_complete"] = 0

    # compute=False writes metadata plus the eager solar variables; the lazy
    # images / frame_valid arrays are filled later by write_block.
    ds.to_zarr(str(out_store), mode="w", compute=False)


def write_block(
    out_store: Path,
    images: np.ndarray,
    frame_valid: np.ndarray,
    lo: int,
    hi: int,
) -> None:
    """Write one contiguous time slice into an existing store."""
    block = xr.Dataset(
        data_vars={
            "images": (("time_utc", "channel", "H", "W"), images),
            "frame_valid": (("time_utc",), frame_valid),
        }
    )
    block.to_zarr(str(out_store), region={"time_utc": slice(lo, hi)})


def update_store_attribute(out_store: Path, key: str, value: int) -> None:
    """Set a group attribute and refresh the consolidated metadata xarray reads."""
    try:
        group = zarr.open_group(str(out_store), mode="a", zarr_format=2)
    except TypeError:
        group = zarr.open_group(str(out_store), mode="a")
    group.attrs[key] = int(value)
    zarr.consolidate_metadata(group.store)


def set_frames_completed(out_store: Path, frames: int) -> None:
    update_store_attribute(out_store, "frames_completed", frames)


def set_run_complete(out_store: Path) -> None:
    update_store_attribute(out_store, "run_complete", 1)


def validate_resumable_store(
    out_store: Path,
    ts_index: pd.DatetimeIndex,
    lat: float,
    lon: float,
    size_km: float,
    resample_size: int,
) -> int:
    """Check an existing store matches this run and return its completed frame count."""
    with xr.open_zarr(out_store) as existing:
        expected = {
            "latitude": lat,
            "longitude": lon,
            "size_km": size_km,
            "resample_size": resample_size,
        }
        for key, value in expected.items():
            stored = existing.attrs.get(key)
            if stored is None or not np.isclose(float(stored), float(value)):
                raise RuntimeError(
                    f"cannot resume {out_store}: {key} is {stored!r}, "
                    f"this run uses {value!r}"
                )
        stored_time = pd.DatetimeIndex(existing["time_utc"].values)
        if len(stored_time) != len(ts_index) or not stored_time.equals(ts_index):
            raise RuntimeError(
                f"cannot resume {out_store}: its timeline "
                f"({len(stored_time)} frames, {stored_time[0]} .. {stored_time[-1]}) "
                f"differs from this run "
                f"({len(ts_index)} frames, {ts_index[0]} .. {ts_index[-1]})"
            )
        completed = int(existing.attrs.get("frames_completed", 0))

    if completed < 0 or completed > len(ts_index):
        raise RuntimeError(
            f"cannot resume {out_store}: frames_completed={completed} is out of range"
        )
    return completed


# Command-line entry point
# ========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop 5,500 x 5,500 Night Microphysics RGB PNGs around one site "
            "into a single Zarr with solar-geometry features."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument(
        "--size-km",
        type=float,
        default=DEFAULT_SIZE_KM,
        help=f"Square ROI edge length in km (default: {DEFAULT_SIZE_KM:g})",
    )
    parser.add_argument(
        "--resample-size",
        type=int,
        default=DEFAULT_RESAMPLE_SIZE,
        help=f"Output square side length in pixels (default: {DEFAULT_RESAMPLE_SIZE})",
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
        "--skip-existing",
        action="store_true",
        help="Skip entire run if the output Zarr store already exists",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run instead of rewriting the store",
    )
    parser.add_argument(
        "--block-frames",
        type=int,
        default=DEFAULT_BLOCK_FRAMES,
        help=(
            "Timestamps decoded and written per block; caps peak memory "
            f"(default: {DEFAULT_BLOCK_FRAMES}, 5 days at 10 min)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "PNG decoding threads; Pillow releases the GIL so this scales "
            f"nearly linearly (default: {DEFAULT_WORKERS}, 1 disables threading)"
        ),
    )
    return parser


def convert(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    data_path: Path,
    out_dir: Path,
    size_km: float = DEFAULT_SIZE_KM,
    resample_size: int = DEFAULT_RESAMPLE_SIZE,
    skip_existing: bool = False,
    resume: bool = False,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
    workers: int = DEFAULT_WORKERS,
) -> tuple[int, int]:
    """Write the site Zarr block by block; return (readable, black) frame counts."""
    if size_km <= 0:
        raise ValueError("size_km must be positive")
    if resample_size <= 0:
        raise ValueError("resample_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if block_frames <= 0:
        raise ValueError("block_frames must be positive")

    data_path = data_path.expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"input root not found: {data_path}")

    out_store = out_dir.expanduser().resolve()
    if skip_existing and out_store.exists():
        print(f"Output already exists, skip: {out_store}")
        return 0, 0
    out_store.parent.mkdir(parents=True, exist_ok=True)

    timeline = regular_times(start, end)
    n_time = len(timeline)
    ts_index = pd.DatetimeIndex(timeline)

    target_u, target_v = local_mapping(lat, lon, size_km, resample_size)
    reader = RoiReader(target_u, target_v)

    print(f"Source: {data_path}")
    print(f"Output zarr store: {out_store}")
    print(
        f"Site lat={lat}, lon={lon}, size_km={size_km:g}, "
        f"resample_size={resample_size} "
        f"({size_km / resample_size:.3f} km per pixel)"
    )
    print(f"Timeline: {timeline[0]} .. {timeline[-1]} ({n_time} frames, 10 min)")
    print(f"Decoding threads: {workers}; block: {block_frames} frames")

    completed = 0
    if resume and out_store.exists():
        completed = validate_resumable_store(
            out_store, ts_index, lat, lon, size_km, resample_size
        )
        print(f"Resuming: {completed}/{n_time} frames already written")
    else:
        create_store(out_store, ts_index, lat, lon, size_km, resample_size)
        print("Created store skeleton (coords and solar features written)")

    n_read = 0
    n_black = 0
    if completed:
        with xr.open_zarr(out_store) as existing:
            done = existing["frame_valid"].values[:completed]
        n_read = int(done.sum())
        n_black = completed - n_read

    block = np.empty(
        (block_frames, len(CHANNELS), resample_size, resample_size), np.uint8
    )
    block_valid = np.empty(block_frames, dtype=np.bool_)
    pool = ThreadPoolExecutor(workers) if workers > 1 else None

    def load_one(offset_and_index: tuple[int, int]) -> bool:
        """Fill block[offset] from disk; return False for a black placeholder."""
        offset, index = offset_and_index
        timestamp = timeline[index]
        path = expected_png_path(data_path, timestamp)
        if not path.is_file():
            block[offset] = 0
            return False
        try:
            block[offset] = reader.read(path)
        except Exception as error:
            block[offset] = 0
            print(
                f"         SKIP {timestamp:%Y-%m-%d %H:%M} "
                f"({type(error).__name__}): {error}",
                flush=True,
            )
            return False
        return True

    try:
        for lo in range(completed, n_time, block_frames):
            hi = min(lo + block_frames, n_time)
            work = [(offset, lo + offset) for offset in range(hi - lo)]

            if pool is None:
                results = [load_one(item) for item in work]
            else:
                results = list(pool.map(load_one, work))
            block_valid[: hi - lo] = results

            write_block(out_store, block[: hi - lo], block_valid[: hi - lo], lo, hi)
            set_frames_completed(out_store, hi)

            block_read = int(block_valid[: hi - lo].sum())
            n_read += block_read
            n_black += (hi - lo) - block_read
            print(
                f"[{hi}/{n_time}] wrote {timeline[lo]:%Y-%m-%d %H:%M} .. "
                f"{timeline[hi - 1]:%Y-%m-%d %H:%M}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown()

    set_run_complete(out_store)
    print(
        f"Done. frames={n_time} readable={n_read} black={n_black} "
        f"output={out_store}"
    )
    return n_read, n_black


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        convert(
            lat=args.lat,
            lon=args.lon,
            start=args.start,
            end=args.end,
            data_path=args.data_path,
            out_dir=args.out_dir,
            size_km=args.size_km,
            resample_size=args.resample_size,
            skip_existing=args.skip_existing,
            resume=args.resume,
            block_frames=args.block_frames,
            workers=args.workers,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
