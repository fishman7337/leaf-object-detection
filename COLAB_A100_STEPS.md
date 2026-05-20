# Colab A100 Leaf YOLO Training

Use a Colab runtime with an A100 GPU. The notebook clones this repository,
mounts Google Drive, caches the raw dataset in Drive, uploads the prepared
YOLO dataset to Drive, and syncs checkpoints/exports to Drive after each model.

## 1. Setup

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/leaf-object-detection"
os.environ["DRIVE_ROOT"] = DRIVE_ROOT
Path(DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
```

```bash
cd /content
git clone https://github.com/fishman7337/leaf-object-detection.git
cd leaf-object-detection
pip install -U -r requirements-colab.txt
```

If you uploaded `kaggle.json`, run:

```bash
mkdir -p ~/.kaggle "$DRIVE_ROOT/datasets" "$DRIVE_ROOT/runs" "$DRIVE_ROOT/artifacts"
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install -U kaggle
kaggle datasets download -d sebastianpalaciob/plantvillage-for-object-detection-yolo -p . --force
mv plantvillage-for-object-detection-yolo.zip archive.zip
cp archive.zip "$DRIVE_ROOT/archive.zip"
```

If you prefer manual upload instead:

```python
from google.colab import files
files.upload()  # upload archive.zip into /content/leaf-object-detection
!cp archive.zip "$DRIVE_ROOT/archive.zip"
```

## 2. Optional Public Mixed-Scene Data

```bash
mkdir -p public
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset \
  public/PlantDoc-Object-Detection-Dataset || true
python scripts/import_plantdoc_git.py \
  --repo public/PlantDoc-Object-Detection-Dataset \
  --out public/plantdoc_yolo \
  --force
```

## 3. Hard Negatives

```bash
python scripts/generate_synthetic_negatives.py \
  --out hard_negatives/synthetic \
  --count 300 \
  --size 320
```

Replace or supplement `hard_negatives/synthetic` with real no-leaf
camera frames later. Real negatives are much stronger than synthetic textures.

## 4. Prepare and Validate Dataset

```bash
python scripts/prepare_leaf_dataset.py \
  --archive archive.zip \
  --work-dir . \
  --extra-yolo-dir public/plantdoc_yolo \
  --hard-negatives-dir hard_negatives/synthetic \
  --force

python scripts/validate_leaf_dataset.py \
  --dataset datasets/leaf_yolo \
  --write-report

tar -czf "$DRIVE_ROOT/datasets/leaf_yolo_dataset.tar.gz" datasets/leaf_yolo
cp datasets/leaf_yolo/validation_report.json "$DRIVE_ROOT/datasets/validation_report.json"
```

## 5. Smoke Test

```bash
python scripts/train_leaf_yolo.py \
  --data datasets/leaf_yolo/data.yaml \
  --project runs/leaf_yolo \
  --device 0 \
  --smoke \
  --export \
  --drive-out "$DRIVE_ROOT/runs/leaf_yolo"
```

## 6. Main Training

```bash
python scripts/train_leaf_yolo.py \
  --data datasets/leaf_yolo/data.yaml \
  --project runs/leaf_yolo \
  --device 0 \
  --models yolo26s.pt yolo26m.pt yolo26x.pt \
  --epochs 180 \
  --imgsz 640 \
  --fine-tune \
  --finetune-epochs 40 \
  --export \
  --drive-out "$DRIVE_ROOT/runs/leaf_yolo"
```

Select the final browser model by test recall, mAP50-95, false positives on
empty-label negatives, exported model size, and browser FPS. Start deployment
with the best `yolo26s` or `yolo26m` ONNX model unless `yolo26x` is still fast
enough in the browser.

## 7. Save Outputs

```bash
zip -r "$DRIVE_ROOT/artifacts/leaf_yolo_runs.zip" runs/leaf_yolo
cp runs/leaf_yolo/training_summary.json "$DRIVE_ROOT/artifacts/training_summary.json"
```

Copy the chosen `best.onnx` into `web/models/best.onnx` to test browser
inference with the included demo.
