#!/usr/bin/env python3
"""Train, evaluate, and export YOLO26 single-class leaf detectors.

Designed for Google Colab A100. Run this after preparing the dataset with
scripts/prepare_leaf_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    "cos_lr": True,
    "multi_scale": False,
}


FINETUNE_ARGS = {
    "epochs": 40,
    "patience": 20,
    "batch": -1,
    "optimizer": "auto",
    "amp": True,
    "pretrained": True,
    "seed": 42,
    "deterministic": True,
    "plots": True,
    "save": True,
    "save_period": 5,
    "workers": 8,
    "close_mosaic": 0,
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.25,
    "fliplr": 0.5,
    "flipud": 0.5,
    "degrees": 10,
    "translate": 0.05,
    "scale": 0.35,
    "perspective": 0.0002,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "cos_lr": True,
    "multi_scale": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/leaf_yolo/data.yaml"))
    parser.add_argument("--project", type=Path, default=Path("runs/leaf_yolo"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--models", nargs="+", default=["yolo26s.pt", "yolo26m.pt", "yolo26x.pt"])
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None, help="Override batch size. Default uses Ultralytics AutoBatch.")
    parser.add_argument("--workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training. Disabled by default for ROCm stability.")
    parser.add_argument("--smoke", action="store_true", help="Run a short 20-epoch yolo26s smoke test only.")
    parser.add_argument("--export", action="store_true", help="Export best weights to ONNX and TF.js after test evaluation.")
    parser.add_argument("--fine-tune", action="store_true", help="Run a lower-augmentation refinement stage from each best.pt.")
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--drive-out", type=Path, default=None, help="Optional Google Drive output folder to sync runs into.")
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


def sync_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def sync_run_to_drive(save_dir: Path, project: Path, drive_out: Path | None) -> None:
    if drive_out is None:
        return
    drive_out = drive_out.resolve()
    drive_out.mkdir(parents=True, exist_ok=True)
    sync_path(save_dir, drive_out / save_dir.name)
    summary_path = project / "training_summary.json"
    sync_path(summary_path, drive_out / "training_summary.json")
    print(f"Synced run outputs to {drive_out}", flush=True)


def evaluate_and_export(best_pt: Path, data_yaml: Path, imgsz: int, device: str, export: bool) -> dict[str, object]:
    print(f"\n=== Test evaluation for {best_pt} ===", flush=True)
    test_model = YOLO(str(best_pt))
    metrics = test_model.val(data=str(data_yaml), split="test", imgsz=imgsz, device=device, plots=True)
    summary: dict[str, object] = {
        "best_pt": str(best_pt),
        "map50": metric_value(metrics, "map50"),
        "map50_95": metric_value(metrics, "map"),
        "precision": metric_value(metrics, "mp"),
        "recall": metric_value(metrics, "mr"),
        "exports": {},
    }
    if export:
        print(f"\n=== Exporting {best_pt} ===", flush=True)
        summary["exports"] = export_model(best_pt, data_yaml, imgsz)
    return summary


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
                "multi_scale": args.multi_scale,
            }
        )
        if args.batch is not None:
            train_args["batch"] = args.batch
        if args.workers is not None:
            train_args["workers"] = args.workers
        train_result = model.train(**train_args)
        save_dir = Path(train_result.save_dir)
        best_pt = save_dir / "weights" / "best.pt"

        summary = {
            "model": model_name,
            "run": run_name,
            "save_dir": str(save_dir),
            "stage": "main",
        }
        summary.update(evaluate_and_export(best_pt, data_yaml, args.imgsz, args.device, args.export))
        run_summary.append(summary)

        summary_path = project / "training_summary.json"
        summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        sync_run_to_drive(save_dir, project, args.drive_out)
        print(json.dumps(summary, indent=2), flush=True)

        if args.fine_tune and not args.smoke:
            ft_run_name = f"{Path(model_name).stem}_leaf_finetune"
            print(f"\n=== Fine-tuning {best_pt} -> {ft_run_name} ===", flush=True)
            ft_model = YOLO(str(best_pt))
            ft_args = dict(FINETUNE_ARGS)
            ft_args.update(
                {
                    "data": str(data_yaml),
                    "epochs": args.finetune_epochs,
                    "imgsz": args.imgsz,
                    "device": args.device,
                    "project": str(project),
                    "name": ft_run_name,
                }
            )
            if args.batch is not None:
                ft_args["batch"] = args.batch
            if args.workers is not None:
                ft_args["workers"] = args.workers
            ft_result = ft_model.train(**ft_args)
            ft_save_dir = Path(ft_result.save_dir)
            ft_best_pt = ft_save_dir / "weights" / "best.pt"
            ft_summary = {
                "model": model_name,
                "run": ft_run_name,
                "save_dir": str(ft_save_dir),
                "stage": "fine_tune",
                "base_best_pt": str(best_pt),
            }
            ft_summary.update(evaluate_and_export(ft_best_pt, data_yaml, args.imgsz, args.device, args.export))
            run_summary.append(ft_summary)
            summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
            sync_run_to_drive(ft_save_dir, project, args.drive_out)
            print(json.dumps(ft_summary, indent=2), flush=True)

    print(f"\nWrote summary: {project / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
