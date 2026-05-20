#!/usr/bin/env python3
"""Convert PlantDoc's Git repository blobs to single-class YOLO.

The upstream repo contains file names that are invalid on Windows, so a normal
checkout can fail. This importer reads files directly from Git objects, writes
sanitized image names, and converts Pascal VOC XML boxes to YOLO `leaf` labels.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("public/PlantDoc-Object-Detection-Dataset"))
    parser.add_argument("--out", type=Path, default=Path("public/plantdoc_yolo"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8", errors="replace")


def git_bytes(repo: Path, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{path}"])


def safe_stem(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._")[:120] or "image"


def find_text(root: ET.Element, path: str, default: str = "") -> str:
    node = root.find(path)
    return node.text.strip() if node is not None and node.text else default


def parse_xml(xml_bytes: bytes) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    root = ET.fromstring(xml_bytes)
    width = int(float(find_text(root, "size/width", "0")))
    height = int(float(find_text(root, "size/height", "0")))
    boxes: list[tuple[float, float, float, float]] = []
    if width <= 0 or height <= 0:
        raise ValueError("Missing or invalid image size in XML")
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = max(0.0, min(float(find_text(box, "xmin", "0")), width))
        ymin = max(0.0, min(float(find_text(box, "ymin", "0")), height))
        xmax = max(0.0, min(float(find_text(box, "xmax", "0")), width))
        ymax = max(0.0, min(float(find_text(box, "ymax", "0")), height))
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1.0 or bh <= 1.0:
            continue
        x = (xmin + xmax) / 2.0 / width
        y = (ymin + ymax) / 2.0 / height
        w = bw / width
        h = bh / height
        boxes.append((x, y, w, h))
    return width, height, boxes


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"Git repo not found: {repo}")
    if out.exists():
        if not args.force:
            print(f"[plantdoc] Output already exists: {out}. Use --force to rebuild.")
            return 0
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    paths = [line for line in git_text(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines() if line]
    path_set = set(paths)
    xml_paths = [p for p in paths if p.lower().endswith(".xml") and (p.startswith("TRAIN/") or p.startswith("TEST/"))]
    imported = 0
    skipped = 0
    missing_images = 0

    for index, xml_path in enumerate(xml_paths, start=1):
        split = "test" if xml_path.startswith("TEST/") else "train"
        base = xml_path.rsplit(".", 1)[0]
        image_path = next((base + ext for ext in IMAGE_EXTS if base + ext in path_set), None)
        if image_path is None:
            image_path = next((base + ext.upper() for ext in IMAGE_EXTS if base + ext.upper() in path_set), None)
        if image_path is None:
            missing_images += 1
            continue

        try:
            _, _, boxes = parse_xml(git_bytes(repo, xml_path))
        except Exception:
            skipped += 1
            continue
        if not boxes:
            skipped += 1
            continue

        image_ext = Path(image_path).suffix.lower()
        name = f"plantdoc_{index:05d}_{safe_stem(Path(image_path).stem)}{image_ext}"
        image_out = out / split / "images" / name
        label_out = out / split / "labels" / f"{Path(name).stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        image_out.write_bytes(git_bytes(repo, image_path))
        label_out.write_text(
            "".join(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for x, y, w, h in boxes),
            encoding="utf-8",
        )
        imported += 1

    print(
        f"[plantdoc] Imported {imported:,} images to {out}; "
        f"skipped {skipped:,}; missing images {missing_images:,}."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[plantdoc] ERROR: {exc}", file=sys.stderr)
        raise
