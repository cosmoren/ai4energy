from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import zarr


DEFAULT_INPUT = Path.home() / "datasets" / "himawari_l1_nightmicrophysics.zarr"
DEFAULT_CHANNELS = ("R", "G", "B")
DEFAULT_VALUE_MIN = 0.0
DEFAULT_VALUE_MAX = 255.0
DEFAULT_FPS = 8.0

# Systematic Himawari Night Microphysics gaps (almost every day).
# BJT: 02:40 UTC ≈ 10:40, 14:40 UTC ≈ 22:40.
SPLIT_GAPS_UTC = ((2, 40), (14, 40))


def parse_utc(value: str) -> datetime:
    text = value.strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1].strip()
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def open_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def map_to_uint8(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    scaled = (values.astype(np.float32) - value_min) / (value_max - value_min)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def get_tile_index(name: str) -> int:
    if not name.startswith("tile_"):
        raise ValueError(f"invalid tile name: {name!r}")
    try:
        return int(name[5:])
    except ValueError as error:
        raise ValueError(f"invalid tile name: {name!r}") from error


def discover_tiles(root) -> dict[int, str]:
    result: dict[int, str] = {}
    for name in root.group_keys():
        index = get_tile_index(name)
        if index in result:
            raise RuntimeError(f"duplicate tile index: {index}")
        result[index] = name
    return result


def get_channel_indices(root, channels: tuple[str, ...]) -> list[int]:
    available = [str(value) for value in root["channel"][:].tolist()]
    indices: list[int] = []
    for channel in channels:
        try:
            indices.append(available.index(channel))
        except ValueError as error:
            raise ValueError(
                f"channel {channel!r} not found. available={available}"
            ) from error
    return indices


def utc_datetime(seconds: int) -> datetime:
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


def utc_day_range(start_utc: datetime, end_utc: datetime) -> list[date]:
    """UTC dates whose 02:40 / 14:40 segments can overlap [start, end].

    Do not look back before ``start``: frames from 00:00 up to the first 02:40
    on the start date are dropped if ``--start`` is not already on a gap.
    """
    first = start_utc.astimezone(timezone.utc).date()
    last = end_utc.astimezone(timezone.utc).date()
    days: list[date] = []
    current = first
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days


def split_segments_for_utc_day(day: date) -> list[tuple[datetime, datetime, str]]:
    """
    Two half-day segments bounded by the systematic gaps.

    Exclusive endpoints at the gap instants themselves:
      after 02:40 → before 14:40
      after 14:40 → before next 02:40
    """
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    gap_am = base.replace(hour=SPLIT_GAPS_UTC[0][0], minute=SPLIT_GAPS_UTC[0][1])
    gap_pm = base.replace(hour=SPLIT_GAPS_UTC[1][0], minute=SPLIT_GAPS_UTC[1][1])
    gap_am_next = gap_am + timedelta(days=1)
    return [
        (gap_am, gap_pm, "0240_1440"),
        (gap_pm, gap_am_next, "1440_0240"),
    ]


def write_video(
    output_path: Path,
    images,
    pixel_valid,
    frame_valid,
    indices: np.ndarray,
    channel_indices: list[int],
    value_min: float,
    value_max: float,
    fps: float,
    codec: str,
) -> dict[str, int]:
    height = int(images.shape[2])
    width = int(images.shape[3])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output_path}")

    invalid_pixel_count = 0
    invalid_frame_count = 0

    try:
        for index in indices:
            if not bool(frame_valid[index]):
                frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                invalid_frame_count += 1
            else:
                planes: list[np.ndarray] = []
                valid_union = np.zeros((height, width), dtype=np.bool_)
                for channel_index in channel_indices:
                    values = np.asarray(images[index, channel_index], dtype=np.int16)
                    valid = np.asarray(
                        pixel_valid[index, channel_index], dtype=np.bool_
                    )
                    valid_union |= valid
                    plane = map_to_uint8(values, value_min, value_max)
                    plane[~valid] = 0
                    planes.append(plane)

                if not valid_union.any():
                    frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                    invalid_frame_count += 1
                else:
                    invalid_pixel_count += int((~valid_union).sum())
                    if len(planes) == 1:
                        frame_bgr = cv2.cvtColor(planes[0], cv2.COLOR_GRAY2BGR)
                    elif len(planes) == 3:
                        # Stored channel order is R, G, B; OpenCV wants B, G, R.
                        frame_bgr = np.stack(
                            [planes[2], planes[1], planes[0]], axis=-1
                        )
                    else:
                        raise ValueError(
                            f"expected 1 or 3 channels for video, got {len(planes)}"
                        )

            writer.write(frame_bgr)
    finally:
        writer.release()

    return {
        "invalid_pixel_count": invalid_pixel_count,
        "invalid_frame_count": invalid_frame_count,
    }


def process_tile(
    root,
    tile_index: int,
    output_dir: Path,
    start_utc: datetime,
    end_utc: datetime,
    channels: tuple[str, ...],
    value_min: float,
    value_max: float,
    fps: float,
    codec: str,
) -> list[dict[str, object]]:
    tile_label = f"tile_{tile_index}"
    images = root["images"]
    pixel_valid = root["pixel_valid"]
    frame_valid = root["frame_valid"]
    time_utc = np.asarray(root["time_utc"][:], dtype=np.int64)

    if images.shape != pixel_valid.shape:
        raise RuntimeError(f"{tile_label}: images/pixel_valid shape mismatch")
    if images.shape[0] != frame_valid.shape[0]:
        raise RuntimeError(f"{tile_label}: frame_valid length mismatch")
    if images.shape[0] != time_utc.shape[0]:
        raise RuntimeError(f"{tile_label}: time_utc length mismatch")

    channel_indices = get_channel_indices(root, channels)
    latitude = float(root["latitude"][0])
    longitude = float(root["longitude"][0])
    records: list[dict[str, object]] = []
    tile_output = output_dir / f"tile_{tile_index}"

    range_start = int(np.ceil(start_utc.timestamp()))
    range_end = int(np.ceil(end_utc.timestamp()))

    for day in utc_day_range(start_utc, end_utc):
        for gap_start, gap_end, segment_id in split_segments_for_utc_day(day):
            # Exclusive of the gap instants (the missing slots used as splitters).
            seg_lo = int(np.floor(gap_start.timestamp())) + 1
            seg_hi = int(np.ceil(gap_end.timestamp())) - 1
            lo = max(seg_lo, range_start)
            hi = min(seg_hi, range_end)
            if lo > hi:
                continue

            indices = np.flatnonzero((time_utc >= lo) & (time_utc <= hi))
            if indices.size == 0:
                continue

            first_frame = utc_datetime(time_utc[indices[0]])
            last_frame = utc_datetime(time_utc[indices[-1]])
            filename = (
                f"{first_frame.strftime('%Y%m%d%H%M')}_"
                f"{last_frame.strftime('%Y%m%d%H%M')}.mkv"
            )
            output_path = tile_output / filename

            statistics = write_video(
                output_path,
                images,
                pixel_valid,
                frame_valid,
                indices,
                channel_indices,
                value_min,
                value_max,
                fps,
                codec,
            )

            record = {
                "tile": tile_index,
                "latitude": latitude,
                "longitude": longitude,
                "utc_split_day": day.isoformat(),
                "segment": segment_id,
                "split_gaps_utc": ["02:40", "14:40"],
                "channels": list(channels),
                "segment_start_exclusive_utc": gap_start.isoformat().replace(
                    "+00:00", "Z"
                ),
                "segment_end_exclusive_utc": gap_end.isoformat().replace(
                    "+00:00", "Z"
                ),
                "first_frame_utc": first_frame.isoformat().replace("+00:00", "Z"),
                "last_frame_utc": last_frame.isoformat().replace("+00:00", "Z"),
                "frame_count": int(indices.size),
                "value_min": value_min,
                "value_max": value_max,
                "fps": fps,
                "codec": codec,
                "video": str(output_path),
                **statistics,
            }
            records.append(record)
            print(
                f"DONE tile_{tile_index} {day} {segment_id}: "
                f"{indices.size} frames -> {output_path}",
                flush=True,
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export Night Microphysics L1 Zarr tiles to MKV clips, "
            "two segments per UTC day split at the systematic 02:40 / 14:40 gaps."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="path to the combined Zarr store containing tile_N groups",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=parse_utc, required=True)
    parser.add_argument("--end", type=parse_utc, required=True)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=list(DEFAULT_CHANNELS),
        help="channels to encode; default R G B as colour video",
    )
    parser.add_argument("--value-min", type=float, default=DEFAULT_VALUE_MIN)
    parser.add_argument("--value-max", type=float, default=DEFAULT_VALUE_MAX)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--codec", default="FFV1")
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be greater than or equal to --start")
    if args.value_max <= args.value_min:
        raise ValueError("--value-max must be greater than --value-min")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    channels = tuple(str(name) for name in args.channels)
    if len(channels) not in (1, 3):
        raise ValueError("--channels must be a single channel or R G B (3 channels)")
    if len(set(channels)) != len(channels):
        raise ValueError("--channels must not contain duplicates")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input store not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    root = open_group(input_dir)
    tiles = discover_tiles(root)
    tile_indices = sorted(tiles)
    if not tile_indices:
        raise RuntimeError("no tile_N groups found in the input Zarr")

    print(f"Zarr source: {input_dir}", flush=True)
    print(f"Tiles to process: {len(tile_indices)}", flush=True)
    print(
        f"Tile index range: {tile_indices[0]}..{tile_indices[-1]}",
        flush=True,
    )
    print(
        "Split gaps (UTC): "
        + ", ".join(f"{h:02d}:{m:02d}" for h, m in SPLIT_GAPS_UTC),
        flush=True,
    )

    records: list[dict[str, object]] = []
    for tile_index in tile_indices:
        tile_root = root[tiles[tile_index]]
        print(f"PROCESS tile_{tile_index}", flush=True)
        records.extend(
            process_tile(
                tile_root,
                tile_index,
                output_dir,
                args.start,
                args.end,
                channels,
                args.value_min,
                args.value_max,
                args.fps,
                args.codec,
            )
        )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Videos: {len(records)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
