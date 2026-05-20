#!/usr/bin/env python3
"""Generate simple no-leaf hard-negative images with empty YOLO labels.

These are not a substitute for real no-leaf camera frames, but they provide
an explicit negative signal for browser deployment until real negatives exist.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hard_negatives/synthetic"))
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=320)
    return parser.parse_args()


def random_color(rng: random.Random, allow_green: bool = True) -> tuple[int, int, int]:
    if allow_green and rng.random() < 0.35:
        return (rng.randint(20, 130), rng.randint(90, 180), rng.randint(20, 120))
    return (rng.randint(20, 230), rng.randint(20, 230), rng.randint(20, 230))


def gradient_image(rng: random.Random, size: int) -> Image.Image:
    c1 = random_color(rng)
    c2 = random_color(rng)
    angle = rng.random() * math.pi
    ax, ay = math.cos(angle), math.sin(angle)
    yy, xx = np.mgrid[0:size, 0:size]
    t = np.clip(((xx * ax + yy * ay) / size + 1.0) / 2.0, 0.0, 1.0)[..., None]
    arr = np.array(c1, dtype=np.float32) * (1.0 - t) + np.array(c2, dtype=np.float32) * t
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def noise_image(rng: random.Random, size: int) -> Image.Image:
    base = np.array(random_color(rng), dtype=np.int16)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).integers(-70, 71, size=(size, size, 3), dtype=np.int16)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 2.2)))


def geometric_image(rng: random.Random, size: int) -> Image.Image:
    img = gradient_image(rng, size)
    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(rng.randint(15, 45)):
        color = (*random_color(rng), rng.randint(45, 150))
        x1 = rng.randint(0, size)
        y1 = rng.randint(0, size)
        x2 = rng.randint(0, size)
        y2 = rng.randint(0, size)
        if rng.random() < 0.5:
            draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=color)
        else:
            draw.line((x1, y1, x2, y2), fill=color, width=rng.randint(2, 18))
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.0)))


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    images_dir = args.out.resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    generators = [gradient_image, noise_image, geometric_image]
    for index in range(args.count):
        img = generators[index % len(generators)](rng, args.size)
        img.save(images_dir / f"synthetic_negative_{index:04d}.jpg", quality=88, optimize=True)
    print(f"[negatives] Wrote {args.count:,} synthetic no-leaf images to {images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
