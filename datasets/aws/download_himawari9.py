#!/usr/bin/env python3
"""Download Himawari-9 AHI L1b Full Disk (FLDK) via s5cmd, then stitch to PNG.

Public bucket: s3://noaa-himawari9/  (anonymous, --no-sign-request)
Requires:
  - s5cmd on PATH (https://github.com/peak/s5cmd)
  - satpy (+ pillow) to compose full-disk images

Output layout (after compose):
  data/B01/20260721_0000.png
  data/B03/20260721_0000.png
Pipeline (default): prefetch next slot download while composing current slot
  (download ∥ compose); within a slot, bands compose in parallel
  (--compose-workers). Then delete raw. Images: data/BXX/YYYYMMDD_HHMM.png.

PNG normalization is fixed per band (see BAND_NORM_RANGE).
satpy units: B01–B06 reflectance [%], B07–B16 brightness temperature [K].
IR ranges follow JMA GSICS standard-scene BT + full-disk percentiles (inverted).

Optional:
  --downsample --target-size 5500   # all bands → ~5500 px width
  --day-start 00:00 --day-end 08:00 # only that UTC window each day

python3 download_himawari9.py --start 2023-07-01 --end 2023-07-31 --bands B03 B07 B10 B13 B15 --downsample --target-size 5500

"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

BUCKET = "noaa-himawari9"
PRODUCT = "AHI-L1b-FLDK"
TIMESTEP = timedelta(minutes=10)
N_SEGMENTS = 10

# Fixed per-band display normalization (same formula for every image of that band).
# Units from satpy ahi_hsd:
#   B01–B06: TOA reflectance [%]
#   B07–B16: brightness temperature [K]
# IR window edges are chosen so JMA GSICS "standard radiance" scenes sit mid-range
# and typical full-disk p1–p99 are not crushed. IR is inverted (cold → bright).
# JMA AHI-9 standard-scene BT [K]: B07≈286, B08≈235, B09≈244, B10≈255,
# B11≈284, B12≈259, B13≈286, B14≈286, B15≈284, B16≈269.
BAND_NORM_RANGE: dict[str, tuple[float, float]] = {
    # VIS/NIR reflectance [%] — clip to 0–100 (overshoots possible under sunglint)
    "B01": (0.0, 100.0),
    "B02": (0.0, 100.0),
    "B03": (0.0, 100.0),
    "B04": (0.0, 100.0),
    "B05": (0.0, 100.0),
    "B06": (0.0, 100.0),
    # IR brightness temperature [K]
    "B07": (200.0, 330.0),  # 3.9 µm; can be hot (fire) and cold (cloud)
    "B08": (190.0, 250.0),  # 6.2 µm upper WV (std ~235 K)
    "B09": (200.0, 260.0),  # 6.9 µm mid WV (std ~244 K)
    "B10": (205.0, 275.0),  # 7.3 µm lower WV (std ~255 K; was washed out at 200–320)
    "B11": (200.0, 320.0),  # 8.6 µm
    "B12": (200.0, 290.0),  # 9.6 µm ozone (std ~259 K)
    "B13": (200.0, 320.0),  # 10.4 µm window
    "B14": (200.0, 320.0),  # 11.2 µm window
    "B15": (200.0, 320.0),  # 12.4 µm window
    "B16": (190.0, 300.0),  # 13.3 µm CO2 (std ~269 K)
}


def parse_time(value: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y%m%d%H%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%Y-%m-%d":
                return dt.replace(hour=0, minute=0)
            return dt
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid time '{value}'. Use e.g. 2026-07-21T00:00 or 202607210000"
    )


def parse_hhmm(value: str) -> tuple[int, int]:
    """Parse daily clock time → (hour, minute). Accepts HH:MM, H:MM, HHMM."""
    value = value.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", value) or re.fullmatch(r"(\d{2})(\d{2})", value)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid clock time '{value}'. Use e.g. 00:00, 8:00, or 0800"
        )
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise argparse.ArgumentTypeError(f"Invalid clock time '{value}'")
    if minute % 10 != 0:
        raise argparse.ArgumentTypeError(
            f"Clock minute must be a multiple of 10 (got {value})"
        )
    return hour, minute


def minutes_of_day(hour: int, minute: int) -> int:
    return hour * 60 + minute


def slot_in_daily_window(
    slot: datetime,
    day_start: tuple[int, int] | None,
    day_end: tuple[int, int] | None,
) -> bool:
    """True if slot's UTC clock time is inside the daily window (inclusive).

    If day_start <= day_end: same-day window, e.g. 00:00–08:00.
    If day_start > day_end: wraps midnight, e.g. 22:00–06:00.
    If either is None: no filtering.
    """
    if day_start is None or day_end is None:
        return True
    m = minutes_of_day(slot.hour, slot.minute)
    a = minutes_of_day(*day_start)
    b = minutes_of_day(*day_end)
    # Align to 10-min grid already done for slots; compare inclusive.
    if a <= b:
        return a <= m <= b
    return m >= a or m <= b


def normalize_bands(bands: list[str]) -> list[str]:
    out: list[str] = []
    for raw in bands:
        for token in re.split(r"[,\s]+", raw.strip()):
            if not token:
                continue
            m = re.fullmatch(r"[Bb]?(\d{1,2})", token)
            if not m:
                raise argparse.ArgumentTypeError(
                    f"Invalid band '{token}'. Use B01..B16 or 1..16"
                )
            n = int(m.group(1))
            if not 1 <= n <= 16:
                raise argparse.ArgumentTypeError(f"Band out of range: {token}")
            name = f"B{n:02d}"
            if name not in out:
                out.append(name)
    if not out:
        raise argparse.ArgumentTypeError("At least one band is required")
    return out


def floor_to_10min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def iter_slots(
    start: datetime,
    end: datetime,
    day_start: tuple[int, int] | None = None,
    day_end: tuple[int, int] | None = None,
):
    t = floor_to_10min(start)
    end = floor_to_10min(end)
    if end < start:
        raise SystemExit("end time must be >= start time")
    while t <= end:
        if slot_in_daily_window(t, day_start, day_end):
            yield t
        t += TIMESTEP


def local_dir(out_root: Path, slot: datetime) -> Path:
    return out_root / PRODUCT / f"{slot:%Y}" / f"{slot:%m}" / f"{slot:%d}" / f"{slot:%H%M}"


def slot_from_dir(slot_dir: Path, out_root: Path) -> datetime | None:
    """Parse .../AHI-L1b-FLDK/YYYY/MM/DD/HHMM -> datetime."""
    try:
        rel = slot_dir.resolve().relative_to((out_root / PRODUCT).resolve())
        year, month, day, hhmm = rel.parts
        if len(hhmm) != 4 or not hhmm.isdigit():
            return None
        return datetime(
            int(year),
            int(month),
            int(day),
            int(hhmm[:2]),
            int(hhmm[2:]),
        )
    except (ValueError, IndexError):
        return None


def discover_local_slots(
    out_root: Path,
    start: datetime | None = None,
    end: datetime | None = None,
    day_start: tuple[int, int] | None = None,
    day_end: tuple[int, int] | None = None,
) -> list[datetime]:
    """Find local HHMM dirs that contain .DAT.bz2 files."""
    root = out_root / PRODUCT
    if not root.is_dir():
        return []
    slots: list[datetime] = []
    for bz2 in root.rglob("*.DAT.bz2"):
        slot = slot_from_dir(bz2.parent, out_root)
        if slot is None:
            continue
        if start is not None and slot < floor_to_10min(start):
            continue
        if end is not None and slot > floor_to_10min(end):
            continue
        if not slot_in_daily_window(slot, day_start, day_end):
            continue
        if slot not in slots:
            slots.append(slot)
    slots.sort()
    return slots


def s3_prefix(slot: datetime) -> str:
    return f"s3://{BUCKET}/{PRODUCT}/{slot:%Y}/{slot:%m}/{slot:%d}/{slot:%H%M}"


def find_s5cmd() -> str:
    path = shutil.which("s5cmd")
    if path:
        return path
    local = Path(__file__).resolve().parent / "s5cmd"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    raise SystemExit(
        "s5cmd not found. Install from https://github.com/peak/s5cmd/releases "
        "and put it on PATH."
    )


def build_cp_commands(slots: list[datetime], bands: list[str], out_root: Path) -> list[str]:
    lines: list[str] = []
    for slot in slots:
        dest = local_dir(out_root, slot)
        dest.mkdir(parents=True, exist_ok=True)
        prefix = s3_prefix(slot)
        for band in bands:
            lines.append(f"cp '{prefix}/*_{band}_*' '{dest}/'")
    return lines


def run_s5cmd(
    s5cmd: str,
    commands: list[str],
    workers: int,
    dry_run: bool,
) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".s5cmd", delete=False) as f:
        f.write("\n".join(commands) + "\n")
        cmdfile = f.name

    cmd = [
        s5cmd,
        "--no-sign-request",
        "--numworkers",
        str(workers),
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["run", cmdfile])

    print(f">>> {' '.join(cmd)}", flush=True)
    print(f"    ({len(commands)} cp command(s))", flush=True)
    try:
        return subprocess.run(cmd).returncode
    finally:
        Path(cmdfile).unlink(missing_ok=True)


def download_slot(
    s5cmd: str,
    slot: datetime,
    bands: list[str],
    out_root: Path,
    workers: int,
    dry_run: bool,
) -> int:
    """Download one slot's raw files. Safe to run in a background thread."""
    print(f"DOWNLOAD start {slot:%Y-%m-%d %H:%M} UTC", flush=True)
    commands = build_cp_commands([slot], bands, out_root)
    code = run_s5cmd(s5cmd, commands, workers, dry_run)
    print(f"DOWNLOAD done  {slot:%Y-%m-%d %H:%M} UTC (exit={code})", flush=True)
    return code


def run_download_compose_pipeline(
    s5cmd: str,
    slots: list[datetime],
    bands: list[str],
    out_root: Path,
    workers: int,
    dry_run: bool,
    no_compose: bool,
    max_width: int | None,
    overwrite_image: bool,
    keep_raw: bool,
    compose_workers: int,
) -> bool:
    """Overlap download(i+1) with compose(i). Returns True if any step failed."""
    any_fail = False
    # One background download at a time; main thread composes.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="himawari-dl") as pool:
        dl_fut: Future[int] | None = pool.submit(
            download_slot, s5cmd, slots[0], bands, out_root, workers, dry_run
        )

        for i, slot in enumerate(slots):
            print(
                f"\n======== [{i + 1}/{len(slots)}] {slot:%Y-%m-%d %H:%M} UTC ========",
                flush=True,
            )
            assert dl_fut is not None
            code = dl_fut.result()

            # Prefetch next slot while we compose this one
            if i + 1 < len(slots):
                next_slot = slots[i + 1]
                print(
                    f"PREFETCH download {next_slot:%Y-%m-%d %H:%M} "
                    f"(while composing {slot:%H:%M})",
                    flush=True,
                )
                dl_fut = pool.submit(
                    download_slot,
                    s5cmd,
                    next_slot,
                    bands,
                    out_root,
                    workers,
                    dry_run,
                )
            else:
                dl_fut = None

            if code != 0:
                print(
                    f"s5cmd failed for slot {slot:%Y-%m-%d %H:%M}",
                    file=sys.stderr,
                )
                any_fail = True
                continue
            if dry_run or no_compose:
                continue

            print(f"COMPOSE start {slot:%Y-%m-%d %H:%M} UTC", flush=True)
            ccode = compose_all(
                [slot],
                bands,
                out_root,
                max_width,
                overwrite_image,
                keep_raw,
                compose_workers,
            )
            if ccode != 0:
                any_fail = True

    return any_fail


def band_files(slot_dir: Path, band: str) -> list[Path]:
    files = sorted(
        p
        for p in slot_dir.glob(f"*_{band}_*.DAT.bz2")
        if p.is_file() and p.stat().st_size > 0
    )
    return files


def image_path(out_root: Path, band: str, slot: datetime) -> Path:
    """e.g. data/B01/20260721_0000.png"""
    return out_root / band / f"{slot:%Y%m%d_%H%M}.png"


def to_uint8_image(data, band: str):
    """Map physical values to uint8 with a fixed per-band formula (not per-image).

    Uses BAND_NORM_RANGE. IR bands (B07+) are inverted after scaling.
    """
    import numpy as np

    lo, hi = BAND_NORM_RANGE[band]
    arr = np.asarray(data, dtype=np.float32)
    mask = np.isfinite(arr)
    img = np.zeros(arr.shape, dtype=np.uint8)
    if not mask.any():
        return img

    scale = hi - lo
    norm = np.zeros_like(arr, dtype=np.float32)
    norm[mask] = (arr[mask] - lo) / scale
    np.clip(norm, 0.0, 1.0, out=norm)
    if band >= "B07":
        norm[mask] = 1.0 - norm[mask]
    img[mask] = (norm[mask] * 255.0 + 0.5).astype(np.uint8)
    return img


def delete_band_raw(slot_dir: Path, band: str) -> int:
    """Remove downloaded HSD segments (and .part) for one band. Return count."""
    n = 0
    for p in list(slot_dir.glob(f"*_{band}_*.DAT.bz2")) + list(
        slot_dir.glob(f"*_{band}_*.DAT.bz2.part")
    ):
        try:
            p.unlink(missing_ok=True)
            n += 1
        except OSError as e:
            print(f"WARNING: could not delete {p}: {e}", file=sys.stderr, flush=True)
    # legacy name in slot dir
    legacy = slot_dir / f"fulldisk_{band}.png"
    if legacy.exists():
        legacy.unlink(missing_ok=True)
    return n


def cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    """Remove empty directories from path up to (but not including) stop_at."""
    stop_at = stop_at.resolve()
    cur = path.resolve()
    while cur != stop_at and stop_at in cur.parents:
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def compose_band_image(
    slot: datetime,
    slot_dir: Path,
    out_root: Path,
    band: str,
    max_width: int | None,
    overwrite: bool,
    keep_raw: bool,
) -> Path | None:
    """Stitch 10 HSD segments -> out_root/BXX/YYYYMMDD_HHMM.png; delete raw unless keep_raw."""
    from PIL import Image
    from satpy import Scene

    out = image_path(out_root, band, slot)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0 and not overwrite:
        print(f"SKIP compose {out} (exists)", flush=True)
        if not keep_raw:
            n = delete_band_raw(slot_dir, band)
            if n:
                print(f"DELETE raw {band} @{slot:%H%M}: {n} file(s)", flush=True)
        return out

    files = band_files(slot_dir, band)
    if len(files) < N_SEGMENTS:
        print(
            f"WARNING: {slot_dir} {band}: need {N_SEGMENTS} segments, "
            f"found {len(files)}; skip compose",
            file=sys.stderr,
            flush=True,
        )
        return None

    scn = Scene(filenames=[str(f) for f in files], reader="ahi_hsd")
    scn.load([band])
    data = scn[band].values

    if max_width is not None and data.shape[1] > max_width:
        step = max(1, int(round(data.shape[1] / max_width)))
        data = data[::step, ::step]

    img = to_uint8_image(data, band)
    Image.fromarray(img).save(out)
    print(f"COMPOSE {out}  shape={img.shape[0]}x{img.shape[1]}", flush=True)

    if not keep_raw:
        n = delete_band_raw(slot_dir, band)
        print(f"DELETE raw {band} @{slot:%H%M}: {n} file(s)", flush=True)

    return out


def compose_all(
    slots: list[datetime],
    bands: list[str],
    out_root: Path,
    max_width: int | None,
    overwrite: bool,
    keep_raw: bool,
    compose_workers: int = 4,
) -> int:
    try:
        import satpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:
        print(
            "Compose needs satpy and pillow. Example:\n"
            "  /home/hw1/micromamba/envs/SimVP/bin/python download_himawari9.py ...\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 1

    n_workers = max(1, min(compose_workers, len(bands)))
    n_ok = 0
    n_fail = 0
    for slot in slots:
        slot_dir = local_dir(out_root, slot)
        if not slot_dir.is_dir():
            # Maybe PNGs already exist and raw was cleaned — count existing images.
            missing = False
            for band in bands:
                out = image_path(out_root, band, slot)
                if out.exists() and out.stat().st_size > 0:
                    print(f"SKIP compose {out} (exists, no raw dir)", flush=True)
                    n_ok += 1
                else:
                    missing = True
                    n_fail += 1
            if missing:
                print(f"WARNING: missing dir {slot_dir}", file=sys.stderr, flush=True)
            continue

        print(
            f"COMPOSE parallel bands={','.join(bands)} workers={n_workers} "
            f"@{slot:%Y-%m-%d %H:%M}",
            flush=True,
        )

        def _one(band: str) -> tuple[str, bool, str | None]:
            try:
                path = compose_band_image(
                    slot, slot_dir, out_root, band, max_width, overwrite, keep_raw
                )
                if path is None:
                    return band, False, "incomplete segments"
                return band, True, None
            except Exception as e:  # noqa: BLE001
                return band, False, str(e)

        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="himawari-compose"
        ) as pool:
            futs = {pool.submit(_one, band): band for band in bands}
            for fut in as_completed(futs):
                band, ok, err = fut.result()
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    print(
                        f"ERROR compose {slot:%Y-%m-%d %H:%M} {band}: {err}",
                        file=sys.stderr,
                        flush=True,
                    )

        if not keep_raw:
            cleanup_empty_parents(slot_dir, out_root)
    print(f"Compose done. ok={n_ok} fail={n_fail}", flush=True)
    print(f"Images under: {out_root.resolve()}/BXX/YYYYMMDD_HHMM.png", flush=True)
    return 1 if n_fail else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Himawari-9 L1b Full Disk via s5cmd, stitch to PNG."
    )
    parser.add_argument(
        "--start",
        type=parse_time,
        help="Start UTC (required for download; optional filter for --compose-only)",
    )
    parser.add_argument(
        "--end",
        type=parse_time,
        help="End UTC inclusive (required for download; optional filter for --compose-only)",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        required=True,
        help="Band list, e.g. B01 B03 B13  or  1 3 13",
    )
    parser.add_argument("--outdir", type=Path, default=Path("data"), help="Output root")
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="s5cmd --numworkers (default: 16)",
    )
    parser.add_argument(
        "--compose-workers",
        type=int,
        default=4,
        help="Parallel band compose threads per slot (default: 4; "
        "lower if RAM is tight with B03)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to s5cmd (no download / no compose)",
    )
    parser.add_argument(
        "--no-compose",
        action="store_true",
        help="Only download, do not stitch full-disk PNGs",
    )
    parser.add_argument(
        "--compose-only",
        action="store_true",
        help="Skip download; stitch all local slot folders under --outdir "
        "(optionally filter with --start/--end)",
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="Downsample all bands to --target-size (off = keep each band native resolution)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=5500,
        help="With --downsample: output width≈this for every band (default: 5500)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # backward-compat alias for --downsample --target-size
    )
    parser.add_argument(
        "--day-start",
        type=parse_hhmm,
        default=None,
        help="Daily UTC window start (HH:MM), e.g. 00:00. Use with --day-end",
    )
    parser.add_argument(
        "--day-end",
        type=parse_hhmm,
        default=None,
        help="Daily UTC window end (HH:MM), inclusive, e.g. 08:00. "
        "If start>end, window wraps midnight (e.g. 22:00–06:00)",
    )
    parser.add_argument(
        "--overwrite-image",
        action="store_true",
        help="Overwrite existing BXX/YYYYMMDD_HHMM.png",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded .DAT.bz2 after compose (default: delete to save space)",
    )
    args = parser.parse_args()

    bands = normalize_bands(args.bands)

    if (args.day_start is None) ^ (args.day_end is None):
        raise SystemExit("Specify both --day-start and --day-end, or neither")

    # Resolution: --downsample/--target-size; --max-width kept as hidden alias
    if args.max_width is not None:
        args.downsample = True
        args.target_size = args.max_width
    if args.downsample:
        if args.target_size <= 0:
            raise SystemExit("--target-size must be > 0 when --downsample is set")
        max_width: int | None = args.target_size
    else:
        max_width = None

    if args.compose_only:
        slots = discover_local_slots(
            args.outdir, args.start, args.end, args.day_start, args.day_end
        )
        if not slots:
            raise SystemExit(
                f"No local .DAT.bz2 found under {args.outdir / PRODUCT}. "
                "Download first, or check --outdir / --start / --end / day window."
            )
    else:
        if args.start is None or args.end is None:
            raise SystemExit("--start and --end are required unless using --compose-only")
        slots = list(
            iter_slots(args.start, args.end, args.day_start, args.day_end)
        )
        if not slots:
            raise SystemExit("No 10-min slots match the date range and daily window")

    print(f"Satellite : Himawari-9", flush=True)
    print(f"Product   : {PRODUCT}", flush=True)
    print(f"Bands     : {', '.join(bands)}", flush=True)
    print(f"Slots     : {len(slots)} x 10-min", flush=True)
    print(
        f"Range     : {slots[0]:%Y-%m-%d %H:%M} -> {slots[-1]:%Y-%m-%d %H:%M} UTC",
        flush=True,
    )
    if args.day_start is not None and args.day_end is not None:
        print(
            f"Daily UTC : {args.day_start[0]:02d}:{args.day_start[1]:02d}"
            f" -> {args.day_end[0]:02d}:{args.day_end[1]:02d} (each day)",
            flush=True,
        )
    if max_width is not None:
        print(f"Downsample: on → ~{max_width}px width (all bands)", flush=True)
    else:
        print("Downsample: off (native per-band resolution)", flush=True)
    print(f"Output    : {args.outdir.resolve()}", flush=True)
    print(f"Image layout: {args.outdir}/BXX/YYYYMMDD_HHMM.png", flush=True)
    print(f"Compose workers: {args.compose_workers}", flush=True)

    # --compose-only: stitch whatever raw slots are already on disk
    if args.compose_only:
        return compose_all(
            slots,
            bands,
            args.outdir,
            max_width,
            args.overwrite_image,
            args.keep_raw,
            args.compose_workers,
        )

    s5cmd = find_s5cmd()
    print(f"Tool      : {s5cmd}", flush=True)
    print(f"Workers   : {args.workers}", flush=True)
    print(
        "Pipeline  : download(slot i+1) ∥ compose(slot i, bands parallel) → delete raw",
        flush=True,
    )

    any_fail = run_download_compose_pipeline(
        s5cmd=s5cmd,
        slots=slots,
        bands=bands,
        out_root=args.outdir,
        workers=args.workers,
        dry_run=args.dry_run,
        no_compose=args.no_compose,
        max_width=max_width,
        overwrite_image=args.overwrite_image,
        keep_raw=args.keep_raw,
        compose_workers=args.compose_workers,
    )

    if args.dry_run:
        print("\nDry-run only.")
        return 0
    if args.no_compose:
        print("\nDone (download only).")
        return 1 if any_fail else 0

    print("\nAll slots finished.", flush=True)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
