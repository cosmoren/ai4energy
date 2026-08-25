from __future__ import annotations

import argparse
import json
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
DEFAULT_VALUE_MAX = 15000.0
DEFAULT_FPS = 8.0
DEFAULT_LOCAL_TIMEZONE = "Asia/Shanghai"
PRE_SUNRISE_PAD = timedelta(hours=2.5)


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


def get_channel_index(root, channel: str,) -> int:
    channels = [str(value)for value in root["channel"][:].tolist()]

    try:
        return channels.index(channel)
    except ValueError as error:
        raise ValueError(f"channel {channel!r} not found. available={channels}") from error


def utc_datetime(seconds: int) -> datetime:
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


def daylight_interval_utc(
    local_day,
    latitude: float,
    longitude: float,
    local_timezone: ZoneInfo,
) -> tuple[datetime, datetime] | None:
    
    location = LocationInfo(
        name="tile",
        region="",
        timezone=str(local_timezone),
        latitude=latitude,
        longitude=longitude,
    )

    # Sunrise or sunset can fall outside the local calendar day when the tile
    # longitude is far from the reference timezone meridian.
    try:
        sunrise_local = sunrise(location.observer, date=local_day, tzinfo=local_timezone)
        sunset_local = sunset(location.observer, date=local_day, tzinfo=local_timezone)
    except ValueError:
        return None

    return (
        sunrise_local.astimezone(timezone.utc),
        sunset_local.astimezone(timezone.utc),
    )


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
                values = np.asarray(images[index, channel_index], dtype=np.int16)
                valid = np.asarray(pixel_valid[index, channel_index], dtype=np.bool_)

                if valid.any():
                    invalid_pixel_count += int((~valid).sum())
                    gray = map_to_uint8(values, value_min, value_max)
                    gray[~valid] = 0
                else:
                    gray = np.zeros((height, width), dtype=np.uint8)
                    invalid_frame_count += 1

            frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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
    channel: str,
    value_min: float,
    value_max: float,
    fps: float,
    codec: str,
    local_timezone: ZoneInfo,
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

    channel_index = get_channel_index(root, channel)
    latitude = float(root["latitude"][0])
    longitude = float(root["longitude"][0])
    local_start_day = start_utc.astimezone(local_timezone).date()
    local_end_day = end_utc.astimezone(local_timezone).date()
    records: list[dict[str, object]] = []
    current_day = local_start_day
    tile_output = output_dir / f"tile_{tile_index}"

    while current_day <= local_end_day:
        daylight = daylight_interval_utc(
            current_day,
            latitude,
            longitude,
            local_timezone,
        )

        if daylight is None:
            print(
                f"SKIP "
                f"tile_{tile_index} "
                f"{current_day}: "
                f"no sunrise/sunset",
                flush=True,
            )
            current_day += timedelta(days=1)
            continue

        sunrise_utc, sunset_utc = daylight

        interval_start = max(sunrise_utc - PRE_SUNRISE_PAD, start_utc)
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
                
                filename = f"{start_date_text}_{end_date_text}.mkv"
                output_path = tile_output / filename

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
                    "tile":
                        tile_index,
                    "latitude":
                        latitude,
                    "longitude":
                        longitude,
                    "local_solar_date":
                        current_day.isoformat(),
                    "channel":
                        channel,
                    "sunrise_utc":
                        sunrise_utc.isoformat().replace("+00:00", "Z"),
                    "sunset_utc":
                        sunset_utc.isoformat().replace("+00:00", "Z"),
                    "first_frame_utc":
                        utc_datetime(time_utc[indices[0]]).isoformat().replace("+00:00", "Z"),
                    "last_frame_utc":
                        utc_datetime(time_utc[indices[-1]]).isoformat().replace("+00:00", "Z"),
                    "frame_count":
                        int(indices.size),
                    "value_min":
                        value_min,
                    "value_max":
                        value_max,
                    "fps":
                        fps,
                    "codec":
                        codec,
                    "video":
                        str(output_path),
                    **statistics,
                }

                records.append(
                    record
                )

                print(
                    f"DONE "
                    f"tile_{tile_index} "
                    f"{current_day}: "
                    f"{indices.size} frames "
                    f"-> "
                    f"{output_path}",
                    flush=True,
                )

        current_day += timedelta(days=1)
        
    return records


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="path to the combined Zarr store containing tile_N groups",
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--start", type=parse_utc, required=True)
    parser.add_argument("--end", type=parse_utc, required=True)

    parser.add_argument("--channel", default=DEFAULT_CHANNEL)

    parser.add_argument("--value-min", type=float, default=DEFAULT_VALUE_MIN)
    parser.add_argument("--value-max", type=float, default=DEFAULT_VALUE_MAX)

    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument( "--codec", default="FFV1")

    parser.add_argument("--local-timezone", default=DEFAULT_LOCAL_TIMEZONE)

    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be greater than or equal to --start")
    if args.value_max <= args.value_min:
        raise ValueError("--value-max must be greater than --value-min")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input store not found: {input_dir}")
    output_dir.mkdir(parents=True,exist_ok=True)

    local_timezone = ZoneInfo(args.local_timezone)

    root = open_group(input_dir)
    tiles = discover_tiles(root)
    tile_indices = sorted(tiles)
    if not tile_indices:
        raise RuntimeError("no tile_N groups found in the input Zarr")

    print(f"Zarr source: {input_dir}", flush=True)
    print(f"Tiles to process: {len(tile_indices)}", flush=True)

    if tile_indices:
        print(
            f"Tile index range: "
            f"{tile_indices[0]}..{tile_indices[-1]}",
            flush=True
        )

    records: list[dict[str, object]] = []

    for tile_index in tile_indices:
        tile_root = root[tiles[tile_index]]

        print(
            f"PROCESS "
            f"tile_{tile_index}",
            flush=True,
        )

        records.extend(
            process_tile(
                tile_root,
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
            output.write(json.dumps(record,ensure_ascii=False) + "\n")

    print(f"Videos: {len(records)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
