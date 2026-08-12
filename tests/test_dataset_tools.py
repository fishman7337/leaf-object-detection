"""Unit tests for deterministic leaf-dataset preparation and validation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from scripts.generate_synthetic_negatives import gradient_image, main, noise_image
from scripts.import_plantdoc_git import parse_xml, safe_stem
from scripts.prepare_leaf_dataset import (
    Box,
    Record,
    normalize_plantdoc_box,
    parse_yolo_label,
    safe_name,
    stratified_split,
    write_label,
)
from scripts.validate_leaf_dataset import validate_label


def test_parse_yolo_label_accepts_valid_rows(tmp_path: Path) -> None:
    label = tmp_path / "leaf.txt"
    label.write_text("7 0.5 0.4 0.2 0.3\n", encoding="utf-8")

    assert parse_yolo_label(label) == (Box(cls=7, x=0.5, y=0.4, w=0.2, h=0.3),)


@pytest.mark.parametrize(
    "content",
    [
        "0 0.5 0.5 0.2\n",
        "0 1.1 0.5 0.2 0.2\n",
        "0 0.5 0.5 0.0 0.2\n",
        "leaf 0.5 0.5 0.2 0.2\n",
    ],
)
def test_parse_yolo_label_rejects_invalid_rows(tmp_path: Path, content: str) -> None:
    label = tmp_path / "invalid.txt"
    label.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        parse_yolo_label(label)


def test_normalize_plantdoc_box_clamps_and_converts_coordinates() -> None:
    row = {
        "filename": "leaf.jpg",
        "class": "healthy leaf",
        "xmin": "-10",
        "ymin": "20",
        "xmax": "110",
        "ymax": "80",
        "width": "100",
        "height": "100",
    }

    filename, class_name, box = normalize_plantdoc_box(row)

    assert filename == "leaf.jpg"
    assert class_name == "healthy leaf"
    assert box == Box(cls=0, x=0.5, y=0.5, w=1.0, h=0.6)


def test_stratified_split_is_deterministic_and_preserves_records(tmp_path: Path) -> None:
    records = [
        Record(
            image=tmp_path / f"leaf-{index}.jpg",
            boxes=(Box(0, 0.5, 0.5, 0.2, 0.2),),
            source="fixture",
            stratum="healthy",
            stem=f"leaf-{index}",
        )
        for index in range(20)
    ]
    forced = Record(
        image=tmp_path / "forced.jpg",
        boxes=(),
        source="fixture",
        stratum="negative",
        stem="forced",
        forced_split="test",
    )
    records.append(forced)

    first = stratified_split(records, seed=42)
    second = stratified_split(records, seed=42)

    assert first == second
    assert forced in first["test"]
    assert sum(map(len, first.values())) == len(records)
    assert {record for split in first.values() for record in split} == set(records)


def test_safe_names_and_labels_are_portable(tmp_path: Path) -> None:
    image = tmp_path / "leaf image.JPG"
    image.touch()
    record = Record(image=image, boxes=(), source="plant doc", stratum="leaf", stem="bad:/name")

    assert safe_name(record, 12) == "plant_doc_000012_bad_name.jpg"
    assert safe_stem("bad:/name?.jpg") == "bad_name_.jpg"

    label = tmp_path / "labels" / "leaf.txt"
    write_label(label, [Box(0, 0.5, 0.5, 0.25, 0.75)])
    assert label.read_text(encoding="utf-8") == "0 0.50000000 0.50000000 0.25000000 0.75000000\n"
    assert validate_label(label) == (1, 1)


def test_parse_xml_converts_pascal_voc_box_to_yolo() -> None:
    xml = b"""<annotation><size><width>200</width><height>100</height></size>
    <object><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>80</ymax>
    </bndbox></object></annotation>"""

    width, height, boxes = parse_xml(xml)

    assert (width, height) == (200, 100)
    assert boxes == [(0.5, 0.5, 0.5, 0.6)]


def test_synthetic_generators_create_reproducible_rgb_images() -> None:
    first = gradient_image(random.Random(42), size=24)
    second = gradient_image(random.Random(42), size=24)
    noise = noise_image(random.Random(42), size=24)

    assert first.mode == noise.mode == "RGB"
    assert first.size == noise.size == (24, 24)
    assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize("option", ["--count", "--size"])
def test_synthetic_generator_rejects_non_positive_dimensions(
    option: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_synthetic_negatives.py", "--out", str(tmp_path), option, "0"],
    )

    with pytest.raises(ValueError, match="greater than zero"):
        main()
