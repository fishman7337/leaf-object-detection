#!/usr/bin/env python3
"""Train, evaluate, and export YOLO26 single-class leaf detectors.

Designed for Google Colab A100. Run this after preparing the dataset with
scripts/prepare_leaf_dataset.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


TRAIN_ARGS = {
    "imgsz": 640,
    "epochs": 180,
    "patience": 30,
    "batch": -1,
    "optimizer": "auto",
    "amp": True,
    "pretrained": True,
    "seed": 42,
    "deterministic": True,
    "plots": True,
    "save": True,
    "save_period": 10,
    "workers": 8,
    "close_mosaic": 15,
    "hsv_h": 0.02,
    "hsv_s": 0.5,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "flipud": 0.5,
    "degrees": 25,
    "translate": 0.1,
    "scale": 0.7,
    "perspective": 0.0005,
    "mosaic": 0.7,
    "mixup": 0.05,
    "cutmix": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/leaf_yolo/data.yaml"))
    parser.add_argument("--project", type=Path, default=Path("runs/leaf_yolo"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--models", nargs="+", default=["yolo26s.pt", "yolo26m.pt", "yolo26x.pt"])
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--smoke", action="store_true", help="Run a short 20-epoch yolo26s smoke test only.")
    parser.add_argument("--export", action="store_true", help="Export best weights to ONNX and TF.js after test evaluation.")
    return parser.parse_args()


def metric_value(metrics: object, name: str) -> float | None:
    box = getattr(metrics, "box", None)
    if box is None:
        return None
    value = getattr(box, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except TypeError:
        return None


def export_model(best_pt: Path, data_yaml: Path, imgsz: int) -> dict[str, str]:
    model = YOLO(str(best_pt))
    exported = {}
    exported["onnx"] = str(model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
    try:
        exported["tfjs"] = str(model.export(format="tfjs", imgsz=imgsz, dynamic=False, data=str(data_yaml)))
    except Exception as exc:  # noqa: BLE001 - keep ONNX even if TF.js dependencies fail.
        exported["tfjs_error"] = repr(exc)
    return exported


def main() -> int:
    args = parse_args()
    data_yaml = args.data.resolve()
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)

    models = ["yolo26s.pt"] if args.smoke else args.models
    epochs = 20 if args.smoke else args.epochs
    run_summary = []

    for model_name in models:
        run_name = f"{Path(model_name).stem}_leaf_{'smoke' if args.smoke else 'main'}"
        print(f"\n=== Training {model_name} -> {run_name} ===", flush=True)
        model = YOLO(model_name)
        train_args = dict(TRAIN_ARGS)
        train_args.update(
            {
                "data": str(data_yaml),
                "epochs": epochs,
                "imgsz": args.imgsz,
                "device": args.device,
                "project": str(project),
                "name": run_name,
            }
        )
        train_result = model.train(**train_args)
        save_dir = Path(train_result.save_dir)
        best_pt = save_dir / "weights" / "best.pt"

        print(f"\n=== Test evaluation for {best_pt} ===", flush=True)
        test_model = YOLO(str(best_pt))
        metrics = test_model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device=args.device, plots=True)

        summary = {
            "model": model_name,
            "run": run_name,
            "save_dir": str(save_dir),
            "best_pt": str(best_pt),
            "map50": metric_value(metrics, "map50"),
            "map50_95": metric_value(metrics, "map"),
            "precision": metric_value(metrics, "mp"),
            "recall": metric_value(metrics, "mr"),
            "exports": {},
        }
        if args.export:
            print(f"\n=== Exporting {best_pt} ===", flush=True)
            summary["exports"] = export_model(best_pt, data_yaml, args.imgsz)
        run_summary.append(summary)

        summary_path = project / "training_summary.json"
        summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)

    print(f"\nWrote summary: {project / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
