# python l2_zarr_to_mp4.py --input-dir ~/data/himawari_l2_300.zarr --output-dir ~/data/himawari_l2_300_mp4 --start "2024-01-01 00:00" --end "2026-01-01 00:00" --channel CLOT

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import zarr
from astral import LocationInfo
from astral.sun import sunrise, sunset


DEFAULT_CHANNEL = "CLOT"
DEFAULT_VALUE_MIN = 0.0
DEFAULT_VALUE_MAX = 32767.0
DEFAULT_FPS = 8.0
DEFAULT_LOCAL_TIMEZONE = "Asia/Shanghai"


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


def discover_tiles(root, selected: list[int] | None) -> list[tuple[int, object]]:
    """Return (tile_index, tile_group) from a store with tile_N subgroups."""
    indices: list[int] = []
    for name in root.group_keys():
        match = re.fullmatch(r"tile_(\d+)", name)
        if match is None:
            raise RuntimeError(f"unexpected Zarr group at root: {name!r}")
        indices.append(int(match.group(1)))
    indices.sort()
    if not indices:
        return []
    if indices != list(range(len(indices))):
        raise RuntimeError(
            f"tile groups must be contiguous from tile_0: "
            f"{[f'tile_{i}' for i in indices]}"
        )

    if selected is None:
        selected_indices = indices
    else:
        selected_set = set(selected)
        missing = sorted(selected_set - set(indices))
        if missing:
            raise FileNotFoundError(f"missing tiles: {missing}")
        selected_indices = sorted(selected_set)

    return [(index, root[f"tile_{index}"]) for index in selected_indices]


def get_channel_index(tile, channel: str) -> int:
    channels = [str(value) for value in tile["channel"][:].tolist()]
    try:
        return channels.index(channel)
    except ValueError as error:
        raise ValueError(
            f"channel {channel!r} not found; available={channels}"
        ) from error


def utc_datetime(seconds: int) -> datetime:
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


def daylight_interval_utc(
    local_day,
    latitude: float,
    longitude: float,
    local_timezone: ZoneInfo,
) -> tuple[datetime, datetime] | None:
    """Return sunrise/sunset UTC, or None when the sun does not rise/set."""
    location = LocationInfo(
        name="tile",
        region="",
        timezone=str(local_timezone),
        latitude=latitude,
        longitude=longitude,
    )
<<<<<<< Updated upstream

    sunrise_local = sunrise(location.observer, date=local_day, tzinfo=local_timezone)
    sunset_local = sunset(location.observer, date=local_day, tzinfo=local_timezone)

    return (
        sunrise_local.astimezone(timezone.utc),
        sunset_local.astimezone(timezone.utc),
    )
=======
    try:
        rise = sunrise(location.observer, date=local_day, tzinfo=local_timezone)
        set_ = sunset(location.observer, date=local_day, tzinfo=local_timezone)
    except ValueError:
        return None
    return rise.astimezone(timezone.utc), set_.astimezone(timezone.utc)
>>>>>>> Stashed changes


def write_video(
    output_path: Path,
    images,
    pixel_valid,
    frame_valid,
    indices: np.ndarray,
    channel_index: int,
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
                gray = np.zeros((height, width), dtype=np.uint8)
                invalid_frame_count += 1
            else:
                values = np.asarray(
                    images[index, channel_index],
                    dtype=np.int16,
                )
                valid = np.asarray(
                    pixel_valid[index, channel_index],
                    dtype=np.bool_,
                )

                if valid.any():
                    filled = values.astype(np.float32)
                    invalid_pixel_count += int((~valid).sum())
                    filled[~valid] = float(filled[valid].mean())
                    gray = map_to_uint8(filled, value_min, value_max)
                else:
                    gray = np.zeros((height, width), dtype=np.uint8)
                    invalid_frame_count += 1

            writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()

    return {
        "invalid_pixel_count": invalid_pixel_count,
        "invalid_frame_count": invalid_frame_count,
    }


def process_tile(
    tile,
    tile_index: int,
    output_dir: Path,
    start_utc: datetime,
    end_utc: datetime,
    channel: str,
    value_min: float,
    value_max: float,
    fps: float,
    codec: str,
    local_timezone: ZoneInfo,
) -> list[dict[str, object]]:
    images = tile["images"]
    pixel_valid = tile["pixel_valid"]
    frame_valid = tile["frame_valid"]
    time_utc = np.asarray(tile["time_utc"][:], dtype=np.int64)
    label = f"tile_{tile_index}"

    if images.shape != pixel_valid.shape:
        raise RuntimeError(f"{label}: images/pixel_valid shape mismatch")
    if images.shape[0] != frame_valid.shape[0]:
        raise RuntimeError(f"{label}: frame_valid length mismatch")
    if images.shape[0] != time_utc.shape[0]:
        raise RuntimeError(f"{label}: time_utc length mismatch")

    channel_index = get_channel_index(tile, channel)
    latitude = float(tile["latitude"][0])
    longitude = float(tile["longitude"][0])

    local_start_day = start_utc.astimezone(local_timezone).date()
    local_end_day = end_utc.astimezone(local_timezone).date()

    records: list[dict[str, object]] = []
    current_day = local_start_day
    tile_output = output_dir / label

    while current_day <= local_end_day:
        daylight = daylight_interval_utc(
            current_day,
            latitude,
            longitude,
            local_timezone,
        )
        if daylight is None:
            print(f"SKIP {label} {current_day}: no sunrise/sunset")
            current_day += timedelta(days=1)
            continue

        sunrise_utc, sunset_utc = daylight
        interval_start = max(sunrise_utc, start_utc)
        interval_end = min(sunset_utc, end_utc + timedelta(microseconds=1))

        if interval_start < interval_end:
            indices = np.flatnonzero(
                (time_utc >= int(np.ceil(interval_start.timestamp())))
                & (time_utc <= int(np.ceil(interval_end.timestamp())))
            )

            if indices.size:
                first_frame = utc_datetime(time_utc[indices[0]])
                last_frame = utc_datetime(time_utc[indices[-1]])
                
                start_date_text = first_frame.strftime("%Y%m%d%H%M")
                end_date_text = last_frame.strftime("%Y%m%d%H%M")
                output_path = (
                    tile_output
                    / f"{start_date_text}_{end_date_text}.mp4"
                )

                statistics = write_video(
                    output_path,
                    images,
                    pixel_valid,
                    frame_valid,
                    indices,
                    channel_index,
                    value_min,
                    value_max,
                    fps,
                    codec,
                )

                record = {
                    "tile": tile_index,
                    "latitude": latitude,
                    "longitude": longitude,
                    "local_solar_date": current_day.isoformat(),
                    "channel": channel,
                    "sunrise_utc": sunrise_utc.isoformat().replace("+00:00", "Z"),
                    "sunset_utc": sunset_utc.isoformat().replace("+00:00", "Z"),
                    "first_frame_utc": utc_datetime(
                        time_utc[indices[0]]
                    ).isoformat().replace("+00:00", "Z"),
                    "last_frame_utc": utc_datetime(
                        time_utc[indices[-1]]
                    ).isoformat().replace("+00:00", "Z"),
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
                    f"DONE {label} {current_day}: "
                    f"{indices.size} frames -> {output_path.name}"
                )

        current_day += timedelta(days=1)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render daylight MP4s from a tiled L2 Zarr store "
            "(himawari_l2.zarr/tile_N/...)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="path to the combined Zarr store containing tile_N groups",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=parse_utc, required=True)
    parser.add_argument("--end", type=parse_utc, required=True)
    parser.add_argument("--tiles", nargs="+", type=int, default=None)
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--value-min", type=float, default=DEFAULT_VALUE_MIN)
    parser.add_argument("--value-max", type=float, default=DEFAULT_VALUE_MAX)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--local-timezone",
        default=DEFAULT_LOCAL_TIMEZONE,
    )
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be greater than or equal to --start")
    if args.value_max <= args.value_min:
        raise ValueError("--value-max must be greater than --value-min")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    local_timezone = ZoneInfo(args.local_timezone)

    root = open_group(input_dir)
    tiles = discover_tiles(root, args.tiles)
    if not tiles:
        raise RuntimeError("no tile_N groups found in the input Zarr")

    records: list[dict[str, object]] = []
    for tile_index, tile in tiles:
        records.extend(
            process_tile(
                tile,
                tile_index,
                output_dir,
                args.start,
                args.end,
                args.channel,
                args.value_min,
                args.value_max,
                args.fps,
                args.codec,
                local_timezone,
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
