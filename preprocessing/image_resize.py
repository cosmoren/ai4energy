#!/usr/bin/env python3
"""
Script to recursively resize images in a folder structure while preserving directory hierarchy.

Usage:
    python image_resize.py <input_folder> <output_folder> --size 448 --workers 16
"""

import argparse
from pathlib import Path
from PIL import Image, ImageFile
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import os

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Common image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def resize_image_worker(args):
    """
    Worker function for parallel image resizing.
    
    Args:
        args: Tuple of (input_path, output_path, size, quality, input_base)
    
    Returns:
        Tuple of (success, relative_path, error_message)
    """
    input_path, output_path, size, quality, input_base = args
    
    try:
        # Open image
        with Image.open(input_path) as img:
            # Convert RGBA to RGB if necessary (removes alpha channel)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            if isinstance(size, int):
                # Square resize
                img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
            else:
                # Specific width and height
                img_resized = img.resize(size, Image.Resampling.LANCZOS)
            
            # Save resized image (create directory if needed)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with appropriate format and quality
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                img_resized.save(output_path, 'JPEG', quality=quality, optimize=True)
            else:
                img_resized.save(output_path, optimize=True)
        
        relative_path = input_path.relative_to(input_base)
        return (True, str(relative_path), None)
    except Exception as e:
        relative_path = input_path.relative_to(input_base) if input_path.exists() else str(input_path)
        return (False, str(relative_path), str(e))


def process_folder(input_folder, output_folder, size, quality=95, num_workers=None, dry_run=False):
    """
    Recursively process all images in a folder structure using parallel processing.
    
    Args:
        input_folder: Root folder containing images
        output_folder: Root folder for resized images
        size: Target size for resizing
        quality: JPEG quality (1-100)
        num_workers: Number of parallel workers (None = use all CPU cores)
        dry_run: If True, only print what would be processed without actually resizing
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_folder}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    # Walk through all directories
    print(f"Scanning folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Target size: {size}")
    print(f"Number of workers: {num_workers}")
    print("-" * 60)
    
    # Find all image files
    print("Finding image files...")
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    # Remove duplicates (case-insensitive matching can create duplicates)
    image_files = sorted(list(set(image_files)))
    total_images = len(image_files)
    print(f"Found {total_images} images to process\n")
    
    if dry_run:
        print("DRY RUN MODE - No images will be resized\n")
        for img_file in image_files:
            relative_path = img_file.relative_to(input_path)
            output_file = output_path / relative_path
            print(f"Would process: {relative_path} -> {output_file}")
        return
    
    # Prepare arguments for parallel processing
    tasks = []
    for img_file in image_files:
        relative_path = img_file.relative_to(input_path)
        output_file = output_path / relative_path
        tasks.append((img_file, output_file, size, quality, input_path))
    
    # Process images in parallel
    processed_images = 0
    failed_images = 0
    failed_list = []
    
    print(f"Processing {total_images} images with {num_workers} workers...\n")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(resize_image_worker, task): task for task in tasks}
        
        # Process completed tasks with progress
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            success, relative_path, error_msg = future.result()
            
            if success:
                processed_images += 1
                # Print progress every 100 images or for every image if less than 100
                if total_images <= 100 or completed % max(1, total_images // 100) == 0:
                    print(f"[{completed}/{total_images}] ✓ {relative_path}")
            else:
                failed_images += 1
                failed_list.append((relative_path, error_msg))
                print(f"[{completed}/{total_images}] ✗ {relative_path}: {error_msg}")
    
    # Print summary
    print("-" * 60)
    print(f"Processing complete!")
    print(f"  Total images: {total_images}")
    print(f"  Successfully processed: {processed_images}")
    print(f"  Failed: {failed_images}")
    
    # Print failed images if any
    if failed_list:
        print("\nFailed images:")
        for rel_path, error in failed_list[:10]:  # Show first 10 failures
            print(f"  {rel_path}: {error}")
        if len(failed_list) > 10:
            print(f"  ... and {len(failed_list) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description='Recursively resize images in a folder structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize all images to 448x448
  python image_resize.py /path/to/images /path/to/output --size 448
  
  # Resize to specific dimensions
  python image_resize.py /path/to/images /path/to/output --size 512 512
  
  # Dry run to see what would be processed
  python image_resize.py /path/to/images /path/to/output --size 448 --dry-run
        """
    )
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='Input folder containing images'
    )
    
    parser.add_argument(
        'output_folder',
        type=str,
        help='Output folder for resized images'
    )
    
    parser.add_argument(
        '--size',
        nargs='+',
        type=int,
        default=[448],
        help='Target size: single integer for square (e.g., 448) or width height (e.g., 512 512). Default: 448'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        choices=range(1, 101),
        metavar='[1-100]',
        help='JPEG quality (1-100). Default: 95'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode: show what would be processed without actually resizing'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: number of CPU cores = {cpu_count()})'
    )
    
    args = parser.parse_args()
    
    # Parse size argument
    if len(args.size) == 1:
        size = args.size[0]
    elif len(args.size) == 2:
        size = tuple(args.size)
    else:
        parser.error("--size must be either a single integer or two integers (width height)")
    
    try:
        process_folder(
            args.input_folder,
            args.output_folder,
            size=size,
            quality=args.quality,
            num_workers=args.workers,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

