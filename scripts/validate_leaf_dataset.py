#!/usr/bin/env python3
"""Validate the prepared single-class YOLO leaf dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/leaf_yolo"))
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def read_yaml(path: Path) -> dict:
    if not path.exists():
        fail(f"Missing data.yaml: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_label(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, 0
    boxes = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            fail(f"{path}:{line_no} has {len(parts)} fields; expected 5")
        if parts[0] != "0":
            fail(f"{path}:{line_no} uses class {parts[0]!r}; expected only class 0")
        values = [float(v) for v in parts[1:]]
        if any(v < 0.0 or v > 1.0 for v in values):
            fail(f"{path}:{line_no} has coordinate outside [0, 1]: {line!r}")
        if values[2] <= 0.0 or values[3] <= 0.0:
            fail(f"{path}:{line_no} has non-positive width/height: {line!r}")
        boxes += 1
    return boxes, 1


def validate_split(dataset: Path, split: str) -> dict[str, int]:
    images_dir = dataset / split / "images"
    labels_dir = dataset / split / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        fail(f"Missing {split}/images or {split}/labels")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(labels_dir.glob("*.txt"))
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems
    if missing_labels or orphan_labels:
        fail(
            f"{split} pairing mismatch: {len(missing_labels)} missing labels, "
            f"{len(orphan_labels)} orphan labels"
        )

    image_failures = []
    for image_path in images[:1000]:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001
            image_failures.append(f"{image_path}: {exc}")
    if image_failures:
        fail("Unreadable images:\n" + "\n".join(image_failures[:10]))

    box_count = 0
    non_empty_labels = 0
    for label_path in labels:
        boxes, non_empty = validate_label(label_path)
        box_count += boxes
        non_empty_labels += non_empty

    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": box_count,
        "non_empty_labels": non_empty_labels,
        "empty_labels": len(labels) - non_empty_labels,
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    data = read_yaml(dataset / "data.yaml")
    if data.get("nc") != 1:
        fail(f"Expected nc: 1, got {data.get('nc')!r}")
    if data.get("names") != ["leaf"]:
        fail(f"Expected names: ['leaf'], got {data.get('names')!r}")

    split_stats = {split: validate_split(dataset, split) for split in ("train", "val", "test")}
    stem_seen: dict[str, str] = {}
    leakage = []
    for split in ("train", "val", "test"):
        for image in (dataset / split / "images").iterdir():
            if image.suffix.lower() not in IMAGE_EXTS:
                continue
            existing = stem_seen.get(image.stem)
            if existing is not None:
                leakage.append(f"{image.stem}: {existing} and {split}")
            stem_seen[image.stem] = split
    if leakage:
        fail("Duplicate image stems across splits:\n" + "\n".join(leakage[:10]))

    report = {
        "dataset": str(dataset),
        "data_yaml": data,
        "splits": split_stats,
        "totals": {
            "images": sum(s["images"] for s in split_stats.values()),
            "labels": sum(s["labels"] for s in split_stats.values()),
            "boxes": sum(s["boxes"] for s in split_stats.values()),
            "empty_labels": sum(s["empty_labels"] for s in split_stats.values()),
        },
    }

    manifest_path = dataset / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["sources"] = dict(Counter(row["source"] for row in manifest))
        report["strata_count"] = len(set(row["stratum"] for row in manifest))

    print(json.dumps(report, indent=2))
    if args.write_report:
        (dataset / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[validate] ERROR: {exc}", file=sys.stderr)
        raise
