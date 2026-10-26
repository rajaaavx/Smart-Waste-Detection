#!/usr/bin/env python3

import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def find_classes(src_dir: Path) -> List[str]:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    classes = [d.name for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not classes:
        raise ValueError(f"No class subfolders found in {src_dir}")
    classes.sort()
    return classes


def collect_images_by_class(src_dir: Path, classes: List[str]) -> Dict[str, List[Path]]:
    images_by_class: Dict[str, List[Path]] = {}
    for cls in classes:
        class_dir = src_dir / cls
        if not class_dir.exists():
            images_by_class[cls] = []
            continue
        files = [p for p in class_dir.rglob('*') if is_image_file(p)]
        images_by_class[cls] = files
    return images_by_class



def split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
   
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_move_file(src: Path, dst: Path, move: bool) -> None:
    ensure_dir(dst.parent)
   
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        parent = dst.parent
        counter = 1
        new_dst = parent / f"{stem}_{counter}{suffix}"
        while new_dst.exists():
            counter += 1
            new_dst = parent / f"{stem}_{counter}{suffix}"
        dst = new_dst
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def prepare_dataset(
    src: Path,
    dst: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    move: bool = False,
) -> None:
    
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    random.seed(seed)

    classes = find_classes(src)
    images_by_class = collect_images_by_class(src, classes)

    
    for split in ("train", "validation", "test"):
        for cls in classes:
            ensure_dir(dst / split / cls)

    summary: Dict[str, Dict[str, int]] = {cls: {"train": 0, "validation": 0, "test": 0} for cls in classes}

    for cls in classes:
        files = images_by_class.get(cls, [])
        if not files:
            print(f"Warning: No images found for class '{cls}'.")
            continue

        
        files_sorted = sorted(files, key=lambda p: p.as_posix())
        random.Random(seed).shuffle(files_sorted)

        n = len(files_sorted)
        n_train, n_val, n_test = split_counts(n, train_ratio, val_ratio, test_ratio)

        train_files = files_sorted[:n_train]
        val_files = files_sorted[n_train : n_train + n_val]
        test_files = files_sorted[n_train + n_val :]

        
        for f in train_files:
            target = dst / "train" / cls / f.name
            copy_or_move_file(f, target, move)
        for f in val_files:
            target = dst / "validation" / cls / f.name
            copy_or_move_file(f, target, move)
        for f in test_files:
            target = dst / "test" / cls / f.name
            copy_or_move_file(f, target, move)

        summary[cls]["train"] = len(train_files)
        summary[cls]["validation"] = len(val_files)
        summary[cls]["test"] = len(test_files)

    
    print("\nDataset split summary:")
    total_train = total_val = total_test = 0
    for cls in classes:
        t = summary[cls]["train"]
        v = summary[cls]["validation"]
        te = summary[cls]["test"]
        total_train += t
        total_val += v
        total_test += te
        print(f"  {cls:>12} -> train: {t:5d}, validation: {v:5d}, test: {te:5d}")
    print(
        f"Totals -> train: {total_train} | validation: {total_val} | test: {total_test} | all: {total_train + total_val + total_test}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare dataset by splitting images from a raw_dataset/ folder into train/validation/test."
        )
    )
    parser.add_argument(
        "--src",
        type=str,
        default="Dataset/archive",
        help="Path to source dataset root with top-level classes (default: Dataset/archive)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="dataset",
        help="Path to output dataset root (default: dataset)",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    prepare_dataset(
        src=src,
        dst=dst,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        move=args.move,
    )


if __name__ == "__main__":
    main()