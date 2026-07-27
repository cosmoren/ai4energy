"""
Convert Himawari L2 cloud-property NetCDF frames into tiled Zarr groups.

The output uses a regular 10-minute UTC timeline.  Each 500 km x 500 km tile
is represented by a 100 x 100 nearest-neighbour grid.

Zarr Layout:

  output.zarr/
    tile_0/                                one group per geographic tile
      images        (time, channel, y, x)  packed int16 source values
      pixel_valid   (time, channel, y, x)  validity for each image value
      frame_valid   (time,)                at least one valid sampled pixel
      time_utc      (time,)                Unix seconds, exactly 10 min apart
      channel       (channel,)             names such as CLOT
      latitude      (1,)                   requested tile-centre latitude
      longitude     (1,)                   requested tile-centre longitude
      target_u      (y, x)                 source-image column indices
      target_v      (y, x)                 source-image row indices
    tile_1/
    ...
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from pyproj import CRS, Transformer


MAINLAND_CHINA_POLYGON = np.asarray(
    [
        (80.259990, 42.349999), (80.180150, 42.920068), (80.866206, 43.180362), (79.966106, 44.917517),
        (81.947071, 45.317027), (82.458926, 45.539650), (83.180484, 47.330031), (85.164290, 47.000956),
        (85.720484, 47.452969), (85.768233, 48.455751), (86.598776, 48.549182), (87.359970, 49.214981),
        (87.751264, 49.297198), (88.013832, 48.599463), (88.854298, 48.069082), (90.280826, 47.693549),
        (90.970809, 46.888146), (90.585768, 45.719716), (90.945540, 45.286073), (92.133891, 45.115076),
        (93.480734, 44.975472), (94.688929, 44.352332), (95.306875, 44.241331), (95.762455, 43.319449),
        (96.349396, 42.725635), (97.451757, 42.748890), (99.515817, 42.524691), (100.845866, 42.663804),
        (101.833040, 42.514873), (103.312278, 41.907468), (104.522282, 41.908347), (104.964994, 41.597410),
        (106.129316, 42.134328), (107.744773, 42.481516), (109.243596, 42.519446), (110.412103, 42.871234),
        (111.129682, 43.406834), (111.829588, 43.743118), (111.667737, 44.073176), (111.348377, 44.457442),
        (111.873306, 45.102079), (112.436062, 45.011646), (113.463907, 44.808893), (114.460332, 45.339817),
        (115.985096, 45.727235), (116.717868, 46.388202), (117.421701, 46.672733), (118.874326, 46.805412),
        (119.663270, 46.692680), (119.772824, 47.048059), (118.866574, 47.747060), (118.064143, 48.066730),
        (117.295507, 47.697709), (116.308953, 47.853410), (115.742837, 47.726545), (115.485282, 48.135383),
        (116.191802, 49.134598), (116.678801, 49.888531), (117.879244, 49.510983), (119.288461, 50.142883),
        (119.279390, 50.582920), (120.182080, 51.643550), (120.738200, 51.964110), (120.725789, 52.516226),
        (120.177089, 52.753886), (121.003085, 53.251401), (122.245748, 53.431726), (123.571470, 53.600000),
        (125.068211, 53.161045), (125.946349, 52.792799), (126.564399, 51.784255), (126.939157, 51.353894),
        (127.287456, 50.739797), (127.800000, 50.200000), (127.657400, 49.760270), (129.397818, 49.440600),
        (130.582293, 48.729687), (130.987260, 47.790130), (132.506690, 47.788960), (133.373596, 48.183442),
        (133.900000, 48.550000), (135.120000, 48.478230), (134.500810, 47.578450), (134.112350, 47.212480),
        (133.769644, 46.116927), (133.097120, 45.144090), (131.883454, 45.321162), (131.025190, 44.967960),
        (131.288555, 44.111520), (131.144688, 42.929990), (130.633866, 42.903015), (130.640000, 42.395024),
        (129.994267, 42.985387), (129.596669, 42.424982), (128.052215, 41.994285), (128.208433, 41.466772),
        (127.343783, 41.503152), (126.869083, 41.816569), (126.182045, 41.107336), (125.079942, 40.569824),
        (124.265625, 39.928493), (122.867570, 39.637788), (122.131388, 39.170452), (121.054554, 38.897471),
        (121.585995, 39.360854), (121.376757, 39.750261), (122.168595, 40.422443), (121.640359, 40.946390),
        (120.768629, 40.593388), (119.639602, 39.898056), (119.023464, 39.252333), (118.042749, 39.204274),
        (117.532702, 38.737636), (118.059699, 38.061476), (118.878150, 37.897325), (118.911636, 37.448464),
        (119.702802, 37.156389), (120.823457, 37.870428), (121.711259, 37.481123), (122.357937, 37.454484),
        (122.519995, 36.930614), (121.104164, 36.651329), (120.637009, 36.111440), (119.664562, 35.609791),
        (119.151208, 34.909859), (120.227525, 34.360332), (120.620369, 33.376723), (121.229014, 32.460319),
        (121.908146, 31.692174), (121.891919, 30.949352), (121.264257, 30.676267), (121.503519, 30.142915),
        (122.092114, 29.832520), (121.938428, 29.018022), (121.684439, 28.225513), (121.125661, 28.135673),
        (120.395473, 27.053207), (119.585497, 25.740781), (118.656871, 24.547391), (117.281606, 23.624501),
        (115.890735, 22.782873), (114.763827, 22.668074), (114.152547, 22.223760), (113.806780, 22.548340),
        (113.241078, 22.051367), (111.843592, 21.550494), (110.785466, 21.397144), (110.444039, 20.341033),
        (109.889861, 20.282457), (109.627655, 21.008227), (109.864488, 21.395051), (108.522813, 21.715212),
        (108.050180, 21.552380), (107.043420, 21.811899), (106.567273, 22.218205), (106.725403, 22.794268),
        (105.811247, 22.976892), (105.329209, 23.352063), (104.476858, 22.819150), (103.504515, 22.703757),
        (102.706992, 22.708795), (102.170436, 22.464753), (101.652018, 22.318199), (101.803120, 21.174367),
        (101.270026, 21.201652), (101.180005, 21.436573), (101.150033, 21.849984), (100.416538, 21.558839),
        (99.983489, 21.742937), (99.240899, 22.118314), (99.531992, 22.949039), (98.898749, 23.142722),
        (98.660262, 24.063286), (97.604720, 23.897405), (97.724609, 25.083637), (98.671838, 25.918703),
        (98.712094, 26.743536), (98.682690, 27.508812), (98.246231, 27.747221), (97.911988, 28.335945),
        (97.327114, 28.261583), (96.248833, 28.411031), (96.586591, 28.830980), (96.117679, 29.452802),
        (95.404802, 29.031717), (94.565990, 29.277438), (93.413348, 28.640629), (92.503119, 27.896876),
        (91.696657, 27.771742), (91.258854, 28.040614), (90.730514, 28.064954), (90.015829, 28.296439),
        (89.475810, 28.042759), (88.814248, 27.299316), (88.730326, 28.086865), (88.120441, 27.876542),
        (86.954517, 27.974262), (85.823320, 28.203576), (85.011638, 28.642774), (84.234580, 28.839894),
        (83.898993, 29.320226), (83.337115, 29.463732), (82.327513, 30.115268), (81.525804, 30.422717),
        (81.111256, 30.183481), (79.721367, 30.882715), (78.738894, 31.515906), (78.458446, 32.618164),
        (79.176129, 32.483780), (79.208892, 32.994395), (78.811086, 33.506198), (78.912269, 34.321936),
        (77.837451, 35.494010), (76.192848, 35.898403), (75.896897, 36.666806), (75.158028, 37.133031),
        (74.980002, 37.419990), (74.829986, 37.990007), (74.864816, 38.378846), (74.257514, 38.606507),
        (73.928852, 38.505815), (73.450000, 39.431237), (73.960013, 39.660008), (73.822244, 39.893973),
        (74.776862, 40.366425), (75.467828, 40.562072), (76.526368, 40.427946), (76.904484, 41.066486),
        (78.187197, 41.185316), (78.543661, 41.582243), (80.119430, 42.123941), (80.259990, 42.349999),
    ],
    dtype=np.float64,
)

HAINAN_POLYGON = np.asarray(
    [
        (109.475210, 18.197701), (108.655208, 18.507682), (108.626217, 19.367888), (109.119056, 19.821039),
        (110.211599, 20.101254), (110.786551, 20.077534), (111.010051, 19.695930), (110.570647, 19.255879),
        (110.339188, 18.678395), (109.475210, 18.197701),
    ],
    dtype=np.float64,
)

TAIWAN_POLYGON = np.asarray(
    [
        (119.3, 23.0), (120.2, 21.7), (121.3, 21.8), (122.2, 25.3), (121.4, 25.7), (120.0, 25.5),
    ],
    dtype=np.float64,
)

CHINA_REGION_POLYGONS = (
    MAINLAND_CHINA_POLYGON,
    HAINAN_POLYGON,
    TAIWAN_POLYGON,
)

REQUIRED_ISLAND_TILE_CENTERS = (
    (19.2, 110.0),  # Hainan
    (23.7, 120.9),  # Taiwan
)


DEFAULT_PATH = Path("data/")
DEFAULT_TILE = 100
DEFAULT_OUT = Path("himawari_tiles.zarr")
DEFAULT_START = "2024-01-01 00:00"
DEFAULT_END = "2024-01-31 23:50"
DEFAULT_CHANNELS = ("CLOT",)  # "CLOT", "CLTT", "CLTH", "CLER_23", "CLTYPE"

FRAME_INTERVAL = timedelta(minutes=10)
TILE_WIDTH_M = 500_000.0
TARGET_PIXEL_M = 5_000.0
TILE_PIXELS = 100
TIME_CHUNK = 12


FILENAME_RE = re.compile(
    r"^NC_[^_]+_(?P<date>\d{8})_(?P<time>\d{4})_.*\.nc$",
    re.IGNORECASE,
)


# Timeline and source-file discovery
# ==================================

def parse_utc(value: str) -> datetime:
    """Parse a CLI timestamp and return a timezone-naive UTC datetime."""
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
    for label, value in (("start", start), ("end", end)):
        if value.second or value.microsecond or value.minute % 10:
            raise ValueError(f"{label} is not aligned to an exact 10-minute boundary")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    count = int((end - start) // FRAME_INTERVAL) + 1
    return [start + index * FRAME_INTERVAL for index in range(count)]


def unix_seconds(values: Sequence[datetime]) -> np.ndarray:
    epoch = datetime(1970, 1, 1)
    return np.asarray([(value - epoch).total_seconds() for value in values], dtype=np.int64)


def datetime_from_filename(path: Path) -> datetime | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    return datetime.strptime(
        match.group("date") + match.group("time"), "%Y%m%d%H%M"
    )


def scan_source_files(
    root: Path, start: datetime, end: datetime
) -> tuple[dict[datetime, Path], Path | None]:
    """Index source files in range and return any file as a grid reference."""
    if not root.is_dir():
        raise FileNotFoundError(f"input root does not exist or is not a directory: {root}")

    indexed: dict[datetime, Path] = {}
    representative: Path | None = None
    for year in range(start.year, end.year + 1):
        year_dir = root / f"{year:04d}"
        if not year_dir.is_dir():
            continue
        for path in year_dir.rglob("*.nc"):
            timestamp = datetime_from_filename(path)
            if timestamp is None:
                continue
            if representative is None:
                representative = path
            if not (start <= timestamp <= end):
                continue
            previous = indexed.get(timestamp)
            if previous is not None:
                raise RuntimeError(
                    f"multiple source files resolve to {timestamp}: {previous} and {path}"
                )
            indexed[timestamp] = path

    # A custom range can contain no frames while another year still contains a
    # perfectly valid reference grid.  Search only until the first one is found.
    if representative is None:
        for path in root.rglob("*.nc"):
            if datetime_from_filename(path) is not None:
                representative = path
                break
    return indexed, representative


# Geographic tile selection and source-pixel mapping
# ==================================================

def point_in_polygon(lon: float, lat: float, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test; boundary precision is unimportant here."""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            boundary_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < boundary_lon:
                inside = not inside
        j = i
    return inside


def point_in_china_region(lon: float, lat: float) -> bool:
    """Return whether a point is in any rough mainland or island region."""
    return any(
        point_in_polygon(lon, lat, polygon) for polygon in CHINA_REGION_POLYGONS
    )


def target_lon_lat(center_lat: float, center_lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Return WGS84 lon/lat for the 100 x 100 projected target pixel centres."""
    x = -TILE_WIDTH_M / 2 + (np.arange(TILE_PIXELS) + 0.5) * TARGET_PIXEL_M
    y = TILE_WIDTH_M / 2 - (np.arange(TILE_PIXELS) + 0.5) * TARGET_PIXEL_M
    target_x, target_y = np.meshgrid(x, y)

    local_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.12f} +lon_0={center_lon:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_wgs84 = Transformer.from_crs(local_aeqd, CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_wgs84.transform(target_x, target_y)
    return np.asarray(lon), np.asarray(lat)


def nearest_coordinate_indices(coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Find nearest indices in a strictly monotonic one-dimensional coordinate."""
    coordinate = np.asarray(coordinates, dtype=np.float64)
    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError("source coordinate must be a one-dimensional array")

    delta = np.diff(coordinate)
    if np.all(delta > 0):
        ordered_coordinate = coordinate
        ordered_index = np.arange(coordinate.size, dtype=np.float64)
    elif np.all(delta < 0):
        ordered_coordinate = coordinate[::-1]
        ordered_index = np.arange(coordinate.size - 1, -1, -1, dtype=np.float64)
    else:
        raise ValueError("source coordinate must be strictly monotonic")

    if np.any(values < ordered_coordinate[0]) or np.any(values > ordered_coordinate[-1]):
        raise ValueError("target footprint extends outside the source coordinate range")

    fractional = np.interp(values, ordered_coordinate, ordered_index)
    return np.rint(fractional).astype(np.int16)


def mapping_for_center(
    center_lat: float,
    center_lon: float,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = target_lon_lat(center_lat, center_lon)
    target_u = nearest_coordinate_indices(source_longitude, lon)
    target_v = nearest_coordinate_indices(source_latitude, lat)
    return target_u, target_v


def sample_tiles(
    n_tile: int,
    seed: int,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_tile <= 0:
        raise ValueError("n-tile must be positive")

    rng = np.random.default_rng(seed)
    all_region_vertices = np.concatenate(CHINA_REGION_POLYGONS, axis=0)
    min_lon, min_lat = all_region_vertices.min(axis=0)
    max_lon, max_lat = all_region_vertices.max(axis=0)
    latitudes: list[float] = []
    longitudes: list[float] = []
    mappings_u: list[np.ndarray] = []
    mappings_v: list[np.ndarray] = []
    maximum_attempts = max(10_000, n_tile * 1_000)

    def append_tile(lat: float, lon: float) -> bool:
        try:
            target_u, target_v = mapping_for_center(
                lat, lon, source_latitude, source_longitude
            )
        except ValueError:
            return False
        latitudes.append(lat)
        longitudes.append(lon)
        mappings_u.append(target_u)
        mappings_v.append(target_v)
        return True

    if n_tile >= len(REQUIRED_ISLAND_TILE_CENTERS):
        for lat, lon in REQUIRED_ISLAND_TILE_CENTERS:
            if not append_tile(lat, lon):
                raise RuntimeError(
                    f"required island tile centered at ({lat}, {lon}) extends "
                    "outside the source coordinate range"
                )

    for _ in range(maximum_attempts):
        if len(latitudes) == n_tile:
            break
        lon = float(rng.uniform(min_lon, max_lon))
        lat = float(rng.uniform(min_lat, max_lat))
        if not point_in_china_region(lon, lat):
            continue
        append_tile(lat, lon)

    if len(latitudes) != n_tile:
        raise RuntimeError(
            f"could only sample {len(latitudes)} valid tiles after {maximum_attempts} attempts"
        )

    return (
        np.asarray(latitudes, dtype=np.float64),
        np.asarray(longitudes, dtype=np.float64),
        np.stack(mappings_u).astype(np.int16, copy=False),
        np.stack(mappings_v).astype(np.int16, copy=False),
    )


def read_source_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as source:
        try:
            latitude = np.asarray(source["latitude"][...], dtype=np.float64)
            longitude = np.asarray(source["longitude"][...], dtype=np.float64)
        except KeyError as error:
            raise RuntimeError(f"{path} lacks latitude/longitude datasets") from error
    return latitude, longitude


# Zarr group creation and layout validation
# =========================================

def open_zarr_group(path: Path):
    """Use Zarr v2 storage semantics under either zarr-python 2 or 3."""
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
    # zarr-python 3 removed Group.create_dataset; when the group is opened with
    # zarr_format=2, create_array still writes a regular Zarr v2-compatible
    # array and accepts the same numcodecs compressor.
    if hasattr(group, "create_array"):
        return group.create_array(name, **kwargs)
    return group.create_dataset(name, **kwargs)


def tile_group_name(index: int) -> str:
    return f"tile_{index}"


def ordered_tile_groups(root, expected_count: int | None = None) -> list:
    """Return tile_0, tile_1, ... in numeric order."""
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


def initialise_or_validate_store(
    output: Path,
    timestamps: np.ndarray,
    channels: Sequence[str],
    n_tile: int,
    seed: int,
    representative: Path | None,
):
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
    root_arrays = set(root.array_keys())
    if root_arrays:
        raise RuntimeError(
            "existing Zarr uses the old flat-array layout; choose a new --output "
            "for the tile_N group layout"
        )

    existing_groups = ordered_tile_groups(root)
    if existing_groups:
        # Resume path: verify every tile before trusting any stored completion
        # flags.  No tile centres or source mappings are recalculated here.
        if len(existing_groups) != n_tile:
            raise RuntimeError(
                f"existing Zarr has {len(existing_groups)} tile groups, expected {n_tile}"
            )
        expected_image_shape = (
            len(timestamps),
            len(channels),
            TILE_PIXELS,
            TILE_PIXELS,
        )
        expected_shapes = {
            "images": expected_image_shape,
            "pixel_valid": expected_image_shape,
            "frame_valid": (len(timestamps),),
            "time_utc": (len(timestamps),),
            "channel": (len(channels),),
            "latitude": (1,),
            "longitude": (1,),
            "target_u": (TILE_PIXELS, TILE_PIXELS),
            "target_v": (TILE_PIXELS, TILE_PIXELS),
        }
        for tile_index, tile in enumerate(existing_groups):
            existing = set(tile.array_keys())
            missing = required - existing
            extra = existing - required
            if missing or extra:
                raise RuntimeError(
                    f"tile_{tile_index} arrays do not match the required layout; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            wrong_shapes = {
                name: (tuple(tile[name].shape), expected)
                for name, expected in expected_shapes.items()
                if tuple(tile[name].shape) != expected
            }
            if wrong_shapes:
                raise RuntimeError(
                    f"tile_{tile_index} has inconsistent array shapes: {wrong_shapes}"
                )
            if int(tile["images"].fill_value) != -1:
                raise RuntimeError(
                    f"tile_{tile_index}/images has fill value "
                    f"{tile['images'].fill_value!r}, expected -1; choose a new "
                    "--output because the frame-validity layout has changed"
                )
            stored_time = np.asarray(tile["time_utc"][:], dtype=np.int64)
            stored_channels = [str(value) for value in tile["channel"][:].tolist()]
            if not np.array_equal(stored_time, timestamps):
                raise RuntimeError(
                    f"tile_{tile_index} time_utc does not match --start/--end"
                )
            if stored_channels != list(channels):
                raise RuntimeError(
                    f"tile_{tile_index} channels {stored_channels} do not match "
                    f"{list(channels)}"
                )
        return root

    if representative is None:
        raise RuntimeError(
            "no readable NetCDF file was found; one source file is needed to build target_u/target_v"
        )

    source_latitude, source_longitude = read_source_coordinates(representative)
    # New-store path: choose centres once and build reusable source indices.
    latitude, longitude, target_u, target_v = sample_tiles(
        n_tile, seed, source_latitude, source_longitude
    )

    n_time = len(timestamps)
    n_channel = len(channels)
    time_chunk = min(TIME_CHUNK, n_time)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    mask_compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    max_channel_length = max(1, max(len(name) for name in channels))
    encoded_channels = np.asarray(channels, dtype=f"<U{max_channel_length}")
    for tile_index in range(n_tile):
        # Everything belonging to a tile lives below this group.  Removing the
        # leading tile dimension gives images the shape (time, channel, y, x).
        tile = root.create_group(tile_group_name(tile_index))
        images = create_array(
            tile,
            "images",
            (n_time, n_channel, TILE_PIXELS, TILE_PIXELS),
            (time_chunk, n_channel, TILE_PIXELS, TILE_PIXELS),
            np.int16,
            -1,
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
            np.dtype(f"<U{max_channel_length}"),
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

        # These small coordinate/mapping arrays are written once.  The large
        # images and masks remain at their fill values until frames are read.
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

    root.attrs["layout"] = "tile_groups_v2"
    root.attrs["n_tile"] = n_tile
    return root


# NetCDF value validation and one-frame sampling
# ==============================================

def scalar_attribute(dataset: h5py.Dataset, name: str):
    if name not in dataset.attrs:
        return None
    value = np.asarray(dataset.attrs[name]).reshape(-1)
    return value[0] if value.size else None


def validity_mask(dataset: h5py.Dataset, values: np.ndarray) -> np.ndarray:
    """Derive validity without changing any source value."""
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
    """Open one frame once, read each channel once, and extract every tile."""
    n_tile = target_u.shape[0]
    result = np.empty(
        (n_tile, len(channels), TILE_PIXELS, TILE_PIXELS), dtype=np.int16
    )
    valid_result = np.empty(result.shape, dtype=np.bool_)

    u_min, u_max = int(target_u.min()), int(target_u.max())
    v_min, v_max = int(target_v.min()), int(target_v.max())
    if u_min < 0 or v_min < 0:
        raise RuntimeError("target_u/target_v contains an out-of-source marker")
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
                raise RuntimeError(f"target_u/target_v is outside {path}:{name} shape")
            if dataset.dtype.kind != "i" or dataset.dtype.itemsize != 2:
                raise RuntimeError(
                    f"{path}:{name} has dtype {dataset.dtype}; images requires packed int16"
                )

            # One rectangular read per channel.  All tiles are gathered from
            # this in-memory union window, so the source dataset is not reread.
            source_window = np.asarray(
                dataset[v_min : v_max + 1, u_min : u_max + 1]
            )
            sampled = source_window[local_v, local_u]
            result[:, channel_index] = sampled.astype(np.int16, copy=False)
            valid_result[:, channel_index] = validity_mask(dataset, sampled)

    return result, valid_result


# Buffered frame processing and per-tile writes
# =============================================

def process_frames(
    root,
    timeline: Sequence[datetime],
    source_files: dict[datetime, Path],
    channels: Sequence[str],
    fail_fast: bool,
) -> tuple[int, int, int]:
    # Only the small mapping and completion arrays are loaded for the full
    # store.  January's images are never loaded into memory as one array.
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
    # With no separate processed marker, a readable all-invalid tile remains
    # incomplete and will be read again if the converter is rerun.
    completed = np.all(completed_by_tile, axis=0)
    time_to_index = {timestamp: index for index, timestamp in enumerate(timeline)}

    pending_by_block: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    skipped = 0
    for timestamp, path in source_files.items():
        index = time_to_index[timestamp]
        if completed[index]:
            skipped += 1
            continue
        pending_by_block[index // TIME_CHUNK].append((index, path))

    written = 0
    failed = 0
    for block_number in sorted(pending_by_block):
        start = block_number * TIME_CHUNK
        end = min(start + TIME_CHUNK, len(timeline))
        block_length = end - start
        block_frame_valid = completed_by_tile[:, start:end].copy()
        # Main RAM buffer: all tiles, but only TIME_CHUNK (normally 12) times.
        # For 100 tiles and one channel, images + mask use about 36 MB.
        block_images = np.full(
            (
                target_u.shape[0],
                block_length,
                len(channels),
                TILE_PIXELS,
                TILE_PIXELS,
            ),
            -1,
            dtype=np.int16,
        )
        block_valid = np.zeros(block_images.shape, dtype=np.bool_)
        for tile_index, tile in enumerate(tiles):
            preserve = block_frame_valid[tile_index]
            if preserve.any():
                # Resume can encounter a partially completed two-hour block.
                # Preserve valid entries while leaving invalid/unwritten ones
                # at the -1/False defaults until they are read again.
                stored_images = np.asarray(
                    tile["images"][start:end, :, :, :]
                )
                stored_valid = np.asarray(
                    tile["pixel_valid"][start:end, :, :, :]
                )
                stored_frame_valid = np.asarray(
                    tile["frame_valid"][start:end], dtype=np.bool_
                )
                block_images[tile_index, preserve] = stored_images[preserve]
                block_valid[tile_index, preserve] = stored_valid[preserve]
                block_frame_valid[tile_index, preserve] = stored_frame_valid[preserve]

        successful_indices: list[int] = []
        dirty_indices: list[int] = []
        for absolute_index, path in sorted(pending_by_block[block_number]):
            offset = absolute_index - start
            timestamp = timeline[absolute_index]
            try:
                # Open this NetCDF once, then extract every tile from the same
                # in-memory source window for each requested channel.
                frame_images, frame_pixel_valid = read_frame_once(
                    path, channels, target_u, target_v
                )
            except Exception as error:
                failed += 1
                # Reset a failed read so an unreadable frame always has the
                # documented -1/False state.
                block_images[:, offset] = -1
                block_valid[:, offset] = False
                block_frame_valid[:, offset] = False
                dirty_indices.append(absolute_index)
                print(
                    f"ERROR {timestamp.isoformat(sep=' ')} UTC {path}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if fail_fast:
                    raise
                continue
            block_images[:, offset] = frame_images
            block_valid[:, offset] = frame_pixel_valid
            # frame_valid is per tile and combines all requested channels.  It
            # is true when at least one sampled pixel in that tile is valid.
            block_frame_valid[:, offset] = np.any(
                frame_pixel_valid, axis=(1, 2, 3)
            )
            successful_indices.append(absolute_index)
            dirty_indices.append(absolute_index)
            print(
                f"READ  {timestamp.isoformat(sep=' ')} UTC {path}",
                flush=True,
            )

        if dirty_indices:
            # Write one two-hour chunk into each tile_N group. frame_valid is
            # committed after images and masks reach disk.
            for tile_index, tile in enumerate(tiles):
                tile["images"][start:end, :, :, :] = block_images[tile_index]
                tile["pixel_valid"][start:end, :, :, :] = block_valid[tile_index]
            for tile_index, tile in enumerate(tiles):
                tile["frame_valid"][start:end] = block_frame_valid[tile_index]
            completed_by_tile[:, start:end] = block_frame_valid
            completed[start:end] = np.all(block_frame_valid, axis=0)
            written += len(successful_indices)
            print(
                f"WROTE time block [{start}, {end}) with "
                f"{len(successful_indices)} new frames and "
                f"{len(dirty_indices) - len(successful_indices)} failed slots reset",
                flush=True,
            )

    return written, skipped, failed


# Command-line and Python entry points
# ====================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop Himawari NetCDF frames into one nearest-neighbour tiled Zarr store."
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
    parser.add_argument(
        "--n-tile",
        type=int,
        default=DEFAULT_TILE,
        help="number of deterministic random tile centres",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_PATH,
        help="unprocessed data root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="single output Zarr store",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=list(DEFAULT_CHANNELS),
        help="NetCDF variables to place in images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="tile sampling seed used only when creating a new store",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop immediately instead of leaving an unreadable frame invalid",
    )
    return parser


def convert(
    start: str | datetime = DEFAULT_START,
    end: str | datetime = DEFAULT_END,
    n_tile: int = 100,
    path: str | Path = "data",
    output: str | Path = "himawari_tiles.zarr",
    channels: Sequence[str] = DEFAULT_CHANNELS,
    seed: int = 42,
    fail_fast: bool = False,
) -> tuple[int, int, int]:
    """Python API corresponding to the command-line interface.

    Returns ``(written, already_valid, failed)``.
    """
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
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("channels must contain one or more unique names")

    input_root = Path(path).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = regular_times(start_time, end_time)
    timestamp_values = unix_seconds(timeline)

    print(f"Input:  {input_root}")
    print(f"Output: {output_path}")
    print(
        f"Range:  {start_time.isoformat(sep=' ')} to {end_time.isoformat(sep=' ')} UTC "
        f"({len(timeline)} slots)"
    )
    print(f"Tiles:  {n_tile}; channels: {', '.join(channels)}")

    source_files, representative = scan_source_files(input_root, start_time, end_time)
    print(f"Found {len(source_files)} source frames inside the requested range")
    root = initialise_or_validate_store(
        output_path,
        timestamp_values,
        channels,
        n_tile,
        seed,
        representative,
    )
    result = process_frames(root, timeline, source_files, channels, fail_fast)
    print(f"Done: written={result[0]}, already_valid={result[1]}, failed={result[2]}")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.end < args.start:
        raise SystemExit("--end must be greater than or equal to --start")
    if args.n_tile <= 0:
        raise SystemExit("--n-tile must be positive")
    if not args.channels or len(set(args.channels)) != len(args.channels):
        raise SystemExit("--channels must contain one or more unique names")

    written, skipped, failed = convert(
        start=args.start,
        end=args.end,
        n_tile=args.n_tile,
        path=args.data_path,
        output=args.output,
        channels=args.channels,
        seed=args.seed,
        fail_fast=args.fail_fast,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
