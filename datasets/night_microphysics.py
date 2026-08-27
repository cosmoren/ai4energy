import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image


NAME_RE = re.compile(r"^\d{8}_\d{4}\.png$")
BT_RANGES = {
    "B07": (200.0, 330.0),
    "B13": (200.0, 320.0),
    "B15": (200.0, 320.0),
}

# JMA Night Microphysics RGB recipe
# R = B15 - B13
# G = B13 - B07
# B = B13
RGB_RANGES = {
    "R": (-4.0, 2.0),
    "G": (0.0, 10.0),
    "B": (243.0, 293.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Himawari B07/B13/B15 grayscale PNGs to Night Microphysics RGB PNGs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input root containing recursively searched B07/, B13/, B15/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for RGB PNGs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PNGs.",
    )
    return parser.parse_args()


def load_gray_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "L":
            image = image.convert("L")
        return np.asarray(image, dtype=np.uint8)


def png_to_bt(image: np.ndarray, band: str) -> np.ndarray:
    """
    Reverse the fixed IR PNG mapping used in download_himawari9.py.

    Original:
        norm = clip((BT - lo) / (hi - lo), 0, 1)
        norm = 1 - norm
        png  = round(norm * 255)

    Approximate inverse:
        BT = lo + (1 - png / 255) * (hi - lo)

    Values clipped during the original PNG creation cannot be recovered.
    """
    lo, hi = BT_RANGES[band]
    norm = image.astype(np.float32) / 255.0
    return lo + (1.0 - norm) * (hi - lo)


def scale_rgb_channel(
    values: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    values = (values - vmin) / (vmax - vmin)
    np.clip(values, 0.0, 1.0, out=values)
    return np.rint(values * 255.0).astype(np.uint8)


def build_night_microphysics(
    b07_png: np.ndarray,
    b13_png: np.ndarray,
    b15_png: np.ndarray,
) -> np.ndarray:
    if b07_png.shape != b13_png.shape or b07_png.shape != b15_png.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"B07={b07_png.shape}, "
            f"B13={b13_png.shape}, "
            f"B15={b15_png.shape}"
        )

    # invalid area is black
    invalid = (b07_png == 0) & (b13_png == 0) & (b15_png == 0)

    b07 = png_to_bt(b07_png, "B07")
    b13 = png_to_bt(b13_png, "B13")
    b15 = png_to_bt(b15_png, "B15")

    r_value = b15 - b13
    g_value = b13 - b07
    b_value = b13

    r = scale_rgb_channel(r_value, *RGB_RANGES["R"])
    g = scale_rgb_channel(g_value, *RGB_RANGES["G"])
    b = scale_rgb_channel(b_value, *RGB_RANGES["B"])

    rgb = np.stack((r, g, b), axis=-1)
    rgb[invalid] = 0

    return rgb


def process_one(
    name: str,
    b07_path: Path,
    b13_path: Path,
    b15_path: Path,
    output_dir: Path,
    overwrite: bool,
):
    output_path = output_dir / name

    if output_path.exists() and not overwrite:
        return name, "exists"

    b07 = load_gray_png(b07_path)
    b13 = load_gray_png(b13_path)
    b15 = load_gray_png(b15_path)

    rgb = build_night_microphysics(b07, b13, b15)

    Image.fromarray(rgb, mode="RGB").save(
        output_path,
        format="PNG",
        compress_level=6,
    )

    return name, "written"


def index_pngs(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.png")):
        if not path.is_file() or not NAME_RE.fullmatch(path.name):
            continue
        previous = indexed.get(path.name)
        if previous is not None:
            raise RuntimeError(
                f"Multiple PNG files have timestamp {path.stem}: "
                f"{previous} and {path}"
            )
        indexed[path.name] = path
    return indexed


def main():
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    b07_dir = input_dir / "B07"
    b13_dir = input_dir / "B13"
    b15_dir = input_dir / "B15"

    for directory in (b07_dir, b13_dir, b15_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing directory: {directory}")

    output_dir.mkdir(parents=True, exist_ok=True)

    b07_files = index_pngs(b07_dir)
    b13_files = index_pngs(b13_dir)
    b15_files = index_pngs(b15_dir)

    b07_names = set(b07_files)
    b13_names = set(b13_files)
    b15_names = set(b15_files)
    common_names = sorted(b07_names & b13_names & b15_names)

    print(f"B07:    {len(b07_names)}")
    print(f"B13:    {len(b13_names)}")
    print(f"B15:    {len(b15_names)}")
    print(f"Common: {len(common_names)}")

    if not common_names:
        raise RuntimeError("No common timestamps found across B07/B13/B15.")

    missing_b07 = (b13_names | b15_names) - b07_names
    missing_b13 = (b07_names | b15_names) - b13_names
    missing_b15 = (b07_names | b13_names) - b15_names

    if missing_b07:
        print(f"Missing B07 timestamps: {len(missing_b07)}")
    if missing_b13:
        print(f"Missing B13 timestamps: {len(missing_b13)}")
    if missing_b15:
        print(f"Missing B15 timestamps: {len(missing_b15)}")

    written = 0
    existing = 0
    failed = 0

    if args.workers <= 1:
        for i, name in enumerate(common_names, start=1):
            try:
                _, status = process_one(
                    name,
                    b07_files[name],
                    b13_files[name],
                    b15_files[name],
                    output_dir,
                    args.overwrite,
                )

                if status == "written":
                    written += 1
                else:
                    existing += 1

            except Exception as exc:
                failed += 1
                print(f"[ERROR] {name}: {exc}")

            if i % 100 == 0 or i == len(common_names):
                print(
                    f"[{i}/{len(common_names)}] "
                    f"written={written} "
                    f"existing={existing} "
                    f"failed={failed}"
                )

    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one,
                    name,
                    b07_files[name],
                    b13_files[name],
                    b15_files[name],
                    output_dir,
                    args.overwrite,
                ): name
                for name in common_names
            }

            completed = 0

            for future in as_completed(futures):
                name = futures[future]
                completed += 1

                try:
                    _, status = future.result()

                    if status == "written":
                        written += 1
                    else:
                        existing += 1

                except Exception as exc:
                    failed += 1
                    print(f"[ERROR] {name}: {exc}")

                if completed % 100 == 0 or completed == len(common_names):
                    print(
                        f"[{completed}/{len(common_names)}] "
                        f"written={written} "
                        f"existing={existing} "
                        f"failed={failed}"
                    )

    print()
    print("DONE")
    print(f"Output:   {output_dir}")
    print(f"Written:  {written}")
    print(f"Existing: {existing}")
    print(f"Failed:   {failed}")


if __name__ == "__main__":
    main()
