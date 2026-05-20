#!/usr/bin/env python3
"""Prepare a single-class YOLO leaf detection dataset.

This script extracts the provided PlantVillage object-detection archive,
rewrites all class IDs to a single `leaf` class, and creates reproducible
train/val/test splits. It can also merge PlantDoc-style public bounding-box
data and user-provided hard-negative images with empty labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLIT_WEIGHTS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Record:
    image: Path
    boxes: tuple[Box, ...]
    source: str
    stratum: str
    stem: str
    forced_split: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plantdoc-dir", type=Path, default=None)
    parser.add_argument(
        "--extra-yolo-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional YOLO datasets with split/images and split/labels folders.",
    )
    parser.add_argument("--hard-negatives-dir", type=Path, default=None)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating hard links.")
    parser.add_argument("--force", action="store_true", help="Remove and rebuild the output dataset.")
    parser.add_argument("--no-extract", action="store_true", help="Assume the raw archive is already extracted.")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def extract_archive(archive: Path, raw_dir: Path, no_extract: bool) -> Path:
    dataset_dir = raw_dir / "PlantVillage_for_object_detection" / "Dataset"
    if dataset_dir.exists():
        log(f"Using existing extracted dataset: {dataset_dir}")
        return dataset_dir
    if no_extract:
        raise FileNotFoundError(f"Expected extracted dataset at {dataset_dir}")
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {archive} to {raw_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Archive extracted, but dataset folder was not found at {dataset_dir}")
    return dataset_dir


def parse_yolo_label(path: Path) -> tuple[Box, ...]:
    boxes: list[Box] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return tuple()
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no} has {len(parts)} fields; expected 5")
        cls_text, *coords_text = parts
        try:
            cls = int(float(cls_text))
            coords = [float(v) for v in coords_text]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} contains non-numeric values: {line!r}") from exc
        if any(v < 0.0 or v > 1.0 for v in coords):
            raise ValueError(f"{path}:{line_no} has coordinate outside [0, 1]: {line!r}")
        x, y, w, h = coords
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"{path}:{line_no} has non-positive box size: {line!r}")
        boxes.append(Box(cls=cls, x=x, y=y, w=w, h=h))
    return tuple(boxes)


def load_plantvillage(dataset_dir: Path) -> list[Record]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected images/ and labels/ under {dataset_dir}")

    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in labels_dir.glob("*.txt")}
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        raise ValueError(
            f"PlantVillage pairing mismatch: {len(missing_labels)} missing labels, "
            f"{len(missing_images)} orphan labels"
        )

    records: list[Record] = []
    prefix_pattern = re.compile(r"^([A-Z0-9]{4})_")
    for stem in sorted(images):
        boxes = parse_yolo_label(labels[stem])
        if not boxes:
            raise ValueError(f"PlantVillage label unexpectedly empty: {labels[stem]}")
        match = prefix_pattern.match(stem)
        stratum = match.group(1) if match else f"class_{boxes[0].cls}"
        records.append(Record(image=images[stem], boxes=boxes, source="plantvillage", stratum=stratum, stem=stem))
    log(f"Loaded PlantVillage records: {len(records):,}")
    return records


def find_column(row: dict[str, str], *candidates: str) -> str | None:
    lower_map = {key.lower().strip(): key for key in row}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            return key
    return None


def normalize_plantdoc_box(row: dict[str, str]) -> tuple[str, str, Box]:
    filename_key = find_column(row, "filename", "file", "image", "image_name")
    class_key = find_column(row, "class", "label", "name")
    xmin_key = find_column(row, "xmin", "x_min", "x1", "left")
    ymin_key = find_column(row, "ymin", "y_min", "y1", "top")
    xmax_key = find_column(row, "xmax", "x_max", "x2", "right")
    ymax_key = find_column(row, "ymax", "y_max", "y2", "bottom")
    width_key = find_column(row, "width", "image_width", "w")
    height_key = find_column(row, "height", "image_height", "h")
    required = [filename_key, class_key, xmin_key, ymin_key, xmax_key, ymax_key, width_key, height_key]
    if any(v is None for v in required):
        raise ValueError(f"Unsupported PlantDoc CSV columns: {sorted(row)}")

    filename = row[filename_key].strip()
    class_name = row[class_key].strip() or "leaf"
    xmin = float(row[xmin_key])
    ymin = float(row[ymin_key])
    xmax = float(row[xmax_key])
    ymax = float(row[ymax_key])
    width = float(row[width_key])
    height = float(row[height_key])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in PlantDoc row: {row}")
    xmin = max(0.0, min(xmin, width))
    xmax = max(0.0, min(xmax, width))
    ymin = max(0.0, min(ymin, height))
    ymax = max(0.0, min(ymax, height))
    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    if bw <= 1.0 or bh <= 1.0:
        raise ValueError(f"Degenerate PlantDoc box: {row}")
    x = (xmin + xmax) / 2.0 / width
    y = (ymin + ymax) / 2.0 / height
    w = bw / width
    h = bh / height
    return filename, class_name, Box(cls=0, x=x, y=y, w=w, h=h)


def locate_plantdoc_image(root: Path, filename: str, preferred_split: str) -> Path:
    candidates = [
        root / preferred_split.upper() / filename,
        root / preferred_split.lower() / filename,
        root / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find PlantDoc image {filename!r} under {root}")


def load_plantdoc(root: Path) -> list[Record]:
    if root is None or not root.exists():
        return []
    csv_files = [
        (root / "train_labels.csv", "train"),
        (root / "test_labels.csv", "test"),
    ]
    grouped: dict[tuple[str, str], list[tuple[str, Box]]] = defaultdict(list)
    for csv_path, split in csv_files:
        if not csv_path.exists():
            continue
        log(f"Loading PlantDoc annotations from {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename, class_name, box = normalize_plantdoc_box(row)
                grouped[(split, filename)].append((class_name, box))

    records: list[Record] = []
    for (split, filename), labeled_boxes in sorted(grouped.items()):
        image = locate_plantdoc_image(root, filename, split)
        boxes = tuple(box for _, box in labeled_boxes)
        class_names = [name for name, _ in labeled_boxes]
        stratum = "plantdoc_" + (Counter(class_names).most_common(1)[0][0] or "leaf").replace(" ", "_")
        forced_split = "test" if split == "test" else None
        records.append(
            Record(
                image=image,
                boxes=boxes,
                source="plantdoc",
                stratum=stratum,
                stem=f"plantdoc_{Path(filename).stem}",
                forced_split=forced_split,
            )
        )
    log(f"Loaded PlantDoc records: {len(records):,}")
    return records


def load_hard_negatives(root: Path | None) -> list[Record]:
    if root is None or not root.exists():
        return []
    records: list[Record] = []
    for image in sorted(root.rglob("*")):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
            records.append(Record(image=image, boxes=tuple(), source="hard_negative", stratum="hard_negative", stem=image.stem))
    log(f"Loaded hard-negative records: {len(records):,}")
    return records


def load_extra_yolo_dir(root: Path) -> list[Record]:
    if root is None or not root.exists():
        return []
    records: list[Record] = []
    for split in ("train", "val", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        labels = {p.stem: p for p in labels_dir.glob("*.txt")}
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels or missing_images:
            raise ValueError(
                f"Extra YOLO dataset pairing mismatch in {root}/{split}: "
                f"{len(missing_labels)} missing labels, {len(missing_images)} orphan labels"
            )
        source = root.name
        for stem in sorted(images):
            boxes = parse_yolo_label(labels[stem])
            records.append(
                Record(
                    image=images[stem],
                    boxes=boxes,
                    source=source,
                    stratum=f"{source}_{split}",
                    stem=f"{source}_{stem}",
                    forced_split=split,
                )
            )
    log(f"Loaded extra YOLO records from {root}: {len(records):,}")
    return records


def stratified_split(records: list[Record], seed: int) -> dict[str, list[Record]]:
    rng = random.Random(seed)
    split_records: dict[str, list[Record]] = {"train": [], "val": [], "test": []}

    flexible: list[Record] = []
    for record in records:
        if record.forced_split in split_records:
            split_records[record.forced_split].append(record)
        else:
            flexible.append(record)

    by_stratum: dict[str, list[Record]] = defaultdict(list)
    for record in flexible:
        by_stratum[record.stratum].append(record)

    for stratum, items in by_stratum.items():
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            split_records["train"].extend(items)
            continue
        n_test = max(1, round(n * DEFAULT_SPLIT_WEIGHTS[2])) if n >= 3 else 0
        n_val = max(1, round(n * DEFAULT_SPLIT_WEIGHTS[1])) if n - n_test >= 2 else 0
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        split_records["test"].extend(items[:n_test])
        split_records["val"].extend(items[n_test : n_test + n_val])
        split_records["train"].extend(items[n_test + n_val :])

    for split in split_records:
        split_records[split].sort(key=lambda r: (r.source, r.stratum, r.stem))
    return split_records


def safe_name(record: Record, index: int) -> str:
    suffix = record.image.suffix.lower()
    cleaned_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.stem).strip("._")
    return f"{record.source}_{index:06d}_{cleaned_stem}{suffix}"


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_label(path: Path, boxes: Iterable[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"0 {box.x:.8f} {box.y:.8f} {box.w:.8f} {box.h:.8f}" for box in boxes]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def build_dataset(split_records: dict[str, list[Record]], out_dir: Path, copy_images: bool) -> dict[str, object]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    copy_modes = Counter()
    global_index = 0
    for split, records in split_records.items():
        for record in records:
            global_index += 1
            filename = safe_name(record, global_index)
            image_dst = out_dir / split / "images" / filename
            label_dst = out_dir / split / "labels" / f"{Path(filename).stem}.txt"
            copy_mode = link_or_copy(record.image, image_dst, copy_images)
            copy_modes[copy_mode] += 1
            write_label(label_dst, record.boxes)
            manifest.append(
                {
                    "split": split,
                    "source": record.source,
                    "stratum": record.stratum,
                    "image": image_dst.relative_to(out_dir).as_posix(),
                    "label": label_dst.relative_to(out_dir).as_posix(),
                    "box_count": len(record.boxes),
                    "original_image": str(record.image),
                }
            )

    data_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["leaf"],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    stats = {
        "records": len(manifest),
        "boxes": sum(int(row["box_count"]) for row in manifest),
        "splits": Counter(row["split"] for row in manifest),
        "sources": Counter(row["source"] for row in manifest),
        "strata": Counter(row["stratum"] for row in manifest),
        "image_materialization": copy_modes,
    }
    serializable_stats = {
        "records": stats["records"],
        "boxes": stats["boxes"],
        "splits": dict(stats["splits"]),
        "sources": dict(stats["sources"]),
        "strata": dict(stats["strata"]),
        "image_materialization": dict(stats["image_materialization"]),
    }
    (out_dir / "stats.json").write_text(json.dumps(serializable_stats, indent=2), encoding="utf-8")
    return serializable_stats


def sanity_check_images(records: list[Record], sample_size: int, seed: int) -> None:
    rng = random.Random(seed)
    sample = records if len(records) <= sample_size else rng.sample(records, sample_size)
    for record in sample:
        try:
            with Image.open(record.image) as image:
                image.verify()
        except Exception as exc:  # noqa: BLE001 - keep validation error direct and useful.
            raise ValueError(f"Could not read image {record.image}: {exc}") from exc


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    raw_dir = (args.raw_dir or work_dir / "raw").resolve()
    out_dir = (args.out_dir or work_dir / "datasets" / "leaf_yolo").resolve()
    archive = args.archive.resolve()

    if out_dir.exists() and not args.force:
        log(f"Output dataset already exists: {out_dir}")
        log("Use --force to rebuild it.")
        return 0

    dataset_dir = extract_archive(archive, raw_dir, args.no_extract)
    records = load_plantvillage(dataset_dir)
    records.extend(load_plantdoc(args.plantdoc_dir.resolve() if args.plantdoc_dir else None))
    for extra_yolo_dir in args.extra_yolo_dir:
        records.extend(load_extra_yolo_dir(extra_yolo_dir.resolve()))
    records.extend(load_hard_negatives(args.hard_negatives_dir.resolve() if args.hard_negatives_dir else None))
    if not records:
        raise ValueError("No records found.")

    sanity_check_images(records, sample_size=200, seed=args.seed)
    split_records = stratified_split(records, args.seed)
    stats = build_dataset(split_records, out_dir, args.copy_images)

    log(f"Prepared dataset at {out_dir}")
    log(f"Records: {stats['records']:,}; boxes: {stats['boxes']:,}")
    log(f"Splits: {stats['splits']}")
    log(f"Sources: {stats['sources']}")
    log(f"Image materialization: {stats['image_materialization']}")
    log(f"YOLO data file: {out_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure.
        print(f"[prepare] ERROR: {exc}", file=sys.stderr)
        raise
