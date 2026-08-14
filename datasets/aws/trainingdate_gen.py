#!/usr/bin/env python3
"""Sample temporal windows from Himawari PNG dataset for training.

Steps implemented so far:
  1. List all filenames under data/B03
  2. Randomly pick one, parse time, build neighboring filenames
  3. Load those files from every band folder; missing → all-zero arrays
  4. Resample/crop a geostationary square patch via satpy (lon/lat, km)
"""

from __future__ import annotations

import argparse
import random
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

NAME_RE = re.compile(r"^(\d{8})_(\d{4})\.png$")

# Himawari-8/9 nominal GEO parameters (JMA / satpy ahi_hsd)
HIMAWARI_SAT_LON = 140.7
HIMAWARI_SAT_HEIGHT_M = 35785863.0
# Full-disk area extent in geos projection meters (matches AHI FLDK coverage)
HIMAWARI_FLDK_EXTENT = (
    -5499999.901531528,
    -5499999.901531528,
    5499999.901531528,
    5499999.901531528,
)


def list_b03_filenames(data_root: Path) -> list[str]:
    """1. Read all filenames under data/B03 into a list."""
    b03_dir = data_root / "B03"
    if not b03_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {b03_dir}")
    names = sorted(p.name for p in b03_dir.glob("*.png") if NAME_RE.match(p.name))
    if not names:
        raise FileNotFoundError(f"No YYYYMMDD_HHMM.png files in {b03_dir}")
    return names


def parse_name(name: str) -> datetime:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected filename: {name}")
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M")


def format_name(dt: datetime) -> str:
    return f"{dt:%Y%m%d_%H%M}.png"


def list_band_dirs(data_root: Path) -> list[str]:
    """Band folder names under data/, e.g. ['B03', 'B07', ...]."""
    bands = sorted(
        p.name
        for p in data_root.iterdir()
        if p.is_dir() and re.fullmatch(r"B\d{2}", p.name)
    )
    if not bands:
        raise FileNotFoundError(f"No BXX folders under {data_root}")
    return bands


def neighbor_names(
    center_name: str,
    n_before: int,
    n_after: int,
    past_interval_minutes: int,
    future_interval_minutes: int,
) -> tuple[list[str], list[str]]:
    """2. Build past / future filenames around the anchor.

    Past (n_before frames): includes the anchor, step = past_interval.
      e.g. n_before=3, past_interval=10 → [t-20, t-10, t]
    Future (n_after frames): excludes the anchor, step = future_interval.
      e.g. n_after=2, future_interval=30 → [t+30, t+60]
    """
    if n_before < 1:
        raise ValueError("n_before (past frames, including anchor) must be >= 1")
    if n_after < 0:
        raise ValueError("n_after (future frames, excluding anchor) must be >= 0")
    if past_interval_minutes <= 0 or future_interval_minutes <= 0:
        raise ValueError("past/future interval_minutes must be > 0")

    center = parse_name(center_name)
    past_step = timedelta(minutes=past_interval_minutes)
    future_step = timedelta(minutes=future_interval_minutes)
    # past: k = -(n_before-1) ... 0  → length n_before, ends at anchor
    past = [format_name(center + k * past_step) for k in range(-(n_before - 1), 1)]
    # future: k = 1 ... n_after  → length n_after, starts after anchor
    future = [
        format_name(center + k * future_step) for k in range(1, n_after + 1)
    ]
    return past, future


def window_names(
    center_name: str,
    n_before: int,
    n_after: int,
    past_interval_minutes: int,
    future_interval_minutes: int,
) -> list[str]:
    """Past + future filenames in chronological order."""
    past, future = neighbor_names(
        center_name,
        n_before,
        n_after,
        past_interval_minutes,
        future_interval_minutes,
    )
    return past + future


def _infer_frame_shape(
    data_root: Path,
    bands: list[str],
    names: list[str],
) -> tuple[int, ...]:
    """Infer HxW (or HxWxC) from any existing file in the window; else any PNG under bands."""
    for name in names:
        for band in bands:
            path = data_root / band / name
            if path.is_file():
                return tuple(np.asarray(Image.open(path)).shape)
    for band in bands:
        for path in sorted((data_root / band).glob("*.png")):
            return tuple(np.asarray(Image.open(path)).shape)
    raise FileNotFoundError(
        f"Cannot infer frame shape: no PNG found under {data_root} for bands={bands}"
    )


def load_window(
    data_root: Path,
    names: list[str],
    bands: list[str] | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load named frames from each band folder; missing files → all-zero arrays.

    Returns
    -------
    dict band -> list of arrays (len == len(names)), each HxW uint8 (or HxWxC)
    """
    if bands is None:
        bands = list_band_dirs(data_root)

    shape = _infer_frame_shape(data_root, bands, names)
    out: dict[str, list[np.ndarray]] = {b: [] for b in bands}
    missing: list[str] = []

    for name in names:
        for band in bands:
            path = data_root / band / name
            if path.is_file():
                arr = np.asarray(Image.open(path))
                if arr.shape != shape:
                    raise ValueError(
                        f"Shape mismatch for {path}: got {arr.shape}, expected {shape}"
                    )
            else:
                arr = np.zeros(shape, dtype=np.uint8)
                missing.append(f"{band}/{name}")
            out[band].append(arr)

    if missing:
        print(
            f"WARNING: {len(missing)} missing file(s) filled with zeros "
            f"(e.g. {missing[0]}"
            + (f", ... +{len(missing) - 1} more" if len(missing) > 1 else "")
            + ")",
            flush=True,
        )
    return out


def sample_window(
    data_root: Path,
    n_before: int = 3,
    n_after: int = 2,
    past_interval_minutes: int = 10,
    future_interval_minutes: int = 10,
    bands: list[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, list[str], list[str], dict[str, list[np.ndarray]]]:
    """Pick a random B03 anchor and load past+future for all bands.

    Missing files are loaded as zeros (see load_window).

    Returns
    -------
    center, past_names, future_names, arrays
      arrays[band] is past+future frames in chronological order.
    """
    rng = rng or random.Random()
    all_names = list_b03_filenames(data_root)
    if bands is None:
        bands = list_band_dirs(data_root)

    center = rng.choice(all_names)
    past, future = neighbor_names(
        center,
        n_before,
        n_after,
        past_interval_minutes,
        future_interval_minutes,
    )
    names = past + future
    arrays = load_window(data_root, names, bands)
    return center, past, future, arrays


def crop_square_geos(
    arrays: dict[str, list[np.ndarray]],
    center_lon: float,
    center_lat: float,
    resolution_km: float,
    side_km: float,
    *,
    sat_lon: float = HIMAWARI_SAT_LON,
    sat_height_m: float = HIMAWARI_SAT_HEIGHT_M,
    resampler: str = "nearest",
) -> tuple[bool, dict[str, list[np.ndarray]] | None]:
    """Resample all frames in ``arrays`` to a square geographic crop.

    Parameters
    ----------
    arrays:
        ``{band: [frame0, frame1, ...]}`` full-disk images (same HxW), e.g. from
        ``load_window`` / ``sample_window``.
    center_lon, center_lat:
        Crop center in degrees (WGS84). Visibility is tested in Himawari GEO
        geometry (``sat_lon`` / ``sat_height_m``).
    resolution_km:
        Output pixel size in kilometers.
    side_km:
        Square side length in kilometers.

    Returns
    -------
    ok, cropped
        ``ok`` is False if (lon, lat) is not visible on the Himawari full disk
        (then ``cropped`` is None). On success, ``cropped`` has the same nested
        structure as ``arrays``, each frame shape ≈ (N, N) with
        ``N = round(side_km / resolution_km)``.
    """
    from pyproj import Proj
    from pyresample.geometry import AreaDefinition, create_area_def
    from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info

    if resolution_km <= 0 or side_km <= 0:
        raise ValueError("resolution_km and side_km must be > 0")
    if not arrays:
        raise ValueError("arrays is empty")
    if resampler not in ("nearest", "nn"):
        raise ValueError(
            f"Only nearest/nn resampler is supported in the fast path, got {resampler!r}"
        )

    # Infer full-disk shape from first frame
    first_band = next(iter(arrays))
    if not arrays[first_band]:
        raise ValueError("arrays has no frames")
    sample = np.asarray(arrays[first_band][0])
    if sample.ndim != 2:
        raise ValueError(f"Expected 2D frames, got shape {sample.shape}")
    height, width = int(sample.shape[0]), int(sample.shape[1])

    proj_dict = {
        "proj": "geos",
        "lon_0": sat_lon,
        "h": sat_height_m,
        "a": 6378137.0,
        "b": 6356752.3,
        "units": "m",
        "sweep": "y",
    }

    # Visibility first (Himawari GEO): lon/lat → geos meters must lie in FLDK
    geos_proj = Proj(proj_dict)
    x_m, y_m = geos_proj(center_lon, center_lat)
    xmin, ymin, xmax, ymax = HIMAWARI_FLDK_EXTENT
    visible = (
        np.isfinite(x_m)
        and np.isfinite(y_m)
        and (xmin <= x_m <= xmax)
        and (ymin <= y_m <= ymax)
    )
    if not visible:
        return False, None

    n_pix = max(1, int(round(side_km / resolution_km)))
    half_m = side_km * 1000.0 / 2.0
    dst_area = create_area_def(
        "square_crop",
        {
            "proj": "laea",
            "lat_0": center_lat,
            "lon_0": center_lon,
            "ellps": "WGS84",
        },
        width=n_pix,
        height=n_pix,
        area_extent=(-half_m, -half_m, half_m, half_m),
        units="m",
    )

    # Crop source to a small GEO window covering the LAEA square before building
    # the KD-tree. Indexing the full 5500² disk is ~100× slower for no benefit.
    laea_proj = Proj(
        {
            "proj": "laea",
            "lat_0": center_lat,
            "lon_0": center_lon,
            "ellps": "WGS84",
        }
    )
    pad_m = half_m * 0.35 + max(resolution_km * 1000.0 * 3.0, 5000.0)
    gx_list: list[float] = []
    gy_list: list[float] = []
    for ex, ey in (
        (-half_m, -half_m),
        (half_m, -half_m),
        (-half_m, half_m),
        (half_m, half_m),
        (0.0, 0.0),
    ):
        clo, cla = laea_proj(ex, ey, inverse=True)
        gx, gy = geos_proj(clo, cla)
        if np.isfinite(gx) and np.isfinite(gy):
            gx_list.append(float(gx))
            gy_list.append(float(gy))
    if not gx_list:
        return False, None

    x_ll, y_ll, x_ur, y_ur = HIMAWARI_FLDK_EXTENT
    x0 = max(x_ll, min(gx_list) - pad_m)
    x1 = min(x_ur, max(gx_list) + pad_m)
    y0 = max(y_ll, min(gy_list) - pad_m)
    y1 = min(y_ur, max(gy_list) + pad_m)
    if x1 <= x0 or y1 <= y0:
        return False, None

    col0 = int(np.floor((x0 - x_ll) / (x_ur - x_ll) * width))
    col1 = int(np.ceil((x1 - x_ll) / (x_ur - x_ll) * width))
    row0 = int(np.floor((y_ur - y1) / (y_ur - y_ll) * height))
    row1 = int(np.ceil((y_ur - y0) / (y_ur - y_ll) * height))
    col0, col1 = max(0, col0), min(width, col1)
    row0, row1 = max(0, row0), min(height, row1)
    if col1 <= col0 or row1 <= row0:
        return False, None

    # Exact GEO extent of the pixel window (may expand slightly vs x0..x1).
    x0p = x_ll + col0 * (x_ur - x_ll) / width
    x1p = x_ll + col1 * (x_ur - x_ll) / width
    y1p = y_ur - row0 * (y_ur - y_ll) / height
    y0p = y_ur - row1 * (y_ur - y_ll) / height
    src_area = AreaDefinition(
        "himawari_crop",
        "Himawari local crop",
        "geos",
        proj_dict,
        col1 - col0,
        row1 - row0,
        (x0p, y0p, x1p, y1p),
    )

    # Build neighbour index once on the small patch, then gather every frame.
    radius_m = max(resolution_km * 1000.0 * 2.5, 5000.0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r".*invalid value encountered.*",
        )
        valid_in, valid_out, index_arr, _dist = get_neighbour_info(
            src_area,
            dst_area,
            radius_of_influence=radius_m,
            neighbours=1,
            epsilon=0.5,
        )

    def _apply(arr: np.ndarray) -> np.ndarray:
        patch = np.asarray(arr[row0:row1, col0:col1], dtype=np.float32)
        out = get_sample_from_neighbour_info(
            "nn",
            dst_area.shape,
            patch,
            valid_in,
            valid_out,
            index_arr,
            fill_value=0,
        )
        if np.issubdtype(arr.dtype, np.integer):
            return np.clip(np.rint(out), 0, 255).astype(arr.dtype)
        return out.astype(arr.dtype, copy=False)

    cropped: dict[str, list[np.ndarray]] = {}
    for band, frames in arrays.items():
        cropped[band] = []
        for frame in frames:
            arr = np.asarray(frame)
            if arr.shape[:2] != (height, width):
                raise ValueError(
                    f"{band}: frame shape {arr.shape} != expected ({height}, {width})"
                )
            cropped[band].append(_apply(arr))

    return True, cropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root containing BXX/ folders (default: ./data)",
    )
    parser.add_argument(
        "--n-before",
        type=int,
        default=3,
        help="Past frames INCLUDING the anchor (default: 3)",
    )
    parser.add_argument(
        "--n-after",
        type=int,
        default=2,
        help="Future frames EXCLUDING the anchor (default: 2)",
    )
    parser.add_argument(
        "--past-interval-minutes",
        type=int,
        default=10,
        help="Time step between past frames in minutes (default: 10)",
    )
    parser.add_argument(
        "--future-interval-minutes",
        type=int,
        default=10,
        help="Time step between future frames in minutes (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Crop center longitude (deg). Requires --lat/--resolution-km/--side-km",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Crop center latitude (deg)",
    )
    parser.add_argument(
        "--resolution-km",
        type=float,
        default=None,
        help="Crop pixel size in km",
    )
    parser.add_argument(
        "--side-km",
        type=float,
        default=None,
        help="Crop square side length in km",
    )
    args = parser.parse_args()

    crop_args = (args.lon, args.lat, args.resolution_km, args.side_km)
    if any(v is not None for v in crop_args) and not all(
        v is not None for v in crop_args
    ):
        raise SystemExit(
            "Provide all of --lon --lat --resolution-km --side-km, or none"
        )

    rng = random.Random(args.seed)
    names_b03 = list_b03_filenames(args.data_root)
    print(f"B03 files: {len(names_b03)}", flush=True)

    center, past, future, arrays = sample_window(
        args.data_root,
        n_before=args.n_before,
        n_after=args.n_after,
        past_interval_minutes=args.past_interval_minutes,
        future_interval_minutes=args.future_interval_minutes,
        rng=rng,
    )
    print(f"anchor : {center}", flush=True)
    print(
        f"past   : {past}  (len={len(past)}, Δ={args.past_interval_minutes}min, "
        f"includes anchor)",
        flush=True,
    )
    print(
        f"future : {future}  (len={len(future)}, Δ={args.future_interval_minutes}min, "
        f"excludes anchor)",
        flush=True,
    )
    for band, frames in arrays.items():
        shapes = [tuple(a.shape) for a in frames]
        print(f"  {band}: {len(frames)} frames, shapes={shapes}", flush=True)

    if args.lon is not None:
        ok, cropped = crop_square_geos(
            arrays,
            center_lon=args.lon,
            center_lat=args.lat,
            resolution_km=args.resolution_km,
            side_km=args.side_km,
        )
        print(
            f"crop   : ok={ok} lon={args.lon} lat={args.lat} "
            f"res={args.resolution_km}km side={args.side_km}km",
            flush=True,
        )
        if not ok:
            print("crop   : point not visible on Himawari full disk", flush=True)
            return 1
        assert cropped is not None
        for band, frames in cropped.items():
            shapes = [tuple(a.shape) for a in frames]
            print(f"  crop {band}: shapes={shapes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
