# Leaf Object Detection

Single-class YOLO dataset and training pipeline for live leaf detection.

Open the training notebook in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fishman7337/leaf-object-detection/blob/main/colab_leaf_yolo_training.ipynb)

The Colab notebook saves everything under:

```text
MyDrive/leaf-object-detection/
```

It caches `archive.zip`, uploads the prepared YOLO dataset as
`datasets/leaf_yolo_dataset.tar.gz`, syncs run folders after each model, and
writes final ZIP/summary artifacts to `artifacts/`.

It also detects the current Drive upload layout directly:

```text
MyDrive/leaf-object-detection/datasets/archive/PlantVillage_for_object_detection/Dataset
```

If that folder exists, Colab prepares the single-class dataset from it without
asking for `archive.zip`.

## Current Local Status

- Prepared dataset, after running prep: `datasets/leaf_yolo`
- Class set: `nc: 1`, `names: ['leaf']`
- Total images: `57,164`
- Total leaf boxes: `63,225`
- Empty-label hard negatives: `300`
- Sources:
  - `54,293` images from the provided PlantVillage YOLO archive
  - `2,571` public PlantDoc mixed-scene images converted from Pascal VOC to YOLO
  - `300` synthetic no-leaf negatives
- Split:
  - Train: `46,008` images, `51,905` boxes
  - Val: `5,460` images, `5,432` boxes
  - Test: `5,696` images, `5,888` boxes

The prepared dataset uses hardlinks for images, so it does not duplicate the
full image storage on disk.

## Important Files

- `scripts/prepare_leaf_dataset.py`: extracts, validates, merges, splits, and rewrites labels to class `0`
- `scripts/validate_leaf_dataset.py`: checks image-label pairing, YOLO box validity, and class IDs
- `scripts/import_plantdoc_git.py`: imports PlantDoc directly from Git blobs, including Windows-hostile filenames
- `scripts/generate_synthetic_negatives.py`: creates temporary no-leaf negative images
- `scripts/train_leaf_yolo.py`: Colab A100 training, test evaluation, ONNX export, and TF.js export
- `datasets/leaf_yolo/data.yaml`: final YOLO dataset config, generated after prep
- `datasets/leaf_yolo/validation_report.json`: latest validation report, generated after validation
- `COLAB_A100_STEPS.md`: exact Colab commands
- `web/`: browser ONNX demo scaffold

## Rebuild Dataset Locally

```powershell
python scripts\prepare_leaf_dataset.py `
  --archive archive.zip `
  --work-dir . `
  --extra-yolo-dir public\plantdoc_yolo `
  --hard-negatives-dir hard_negatives\synthetic `
  --force

python scripts\validate_leaf_dataset.py `
  --dataset datasets\leaf_yolo `
  --write-report
```

## Train on Colab A100

Follow `COLAB_A100_STEPS.md`.

The smoke test trains `yolo26s.pt` for 20 epochs. The main run trains:

- `yolo26s.pt`
- `yolo26m.pt`
- `yolo26x.pt`

Training uses pretrained weights, early stopping, cosine LR, multi-scale main
training, mosaic closeout, HSV/geometry augmentation, mixup, and cutmix. The
main Colab run also performs a lower-augmentation fine-tune stage from each
candidate model's `best.pt`. The final model should be chosen by test recall,
mAP50-95, false positives on no-leaf images, model size, and browser FPS.

## Deployment

Exported ONNX models can be tested in `web/index.html` after placing the chosen
model at:

```text
leaf_detector/web/models/best.onnx
```

For a browser-first deployment, start with `yolo26s` or `yolo26m`. Keep `yolo26x`
only if the browser FPS is acceptable.

## Notes

The original PlantVillage data is mostly centered 256x256 single-leaf imagery.
The PlantDoc merge and hard negatives reduce that bias, but real webcam images
from the final environment are still the best way to prove live performance.

The original PlantVillage object-detection dataset is hosted on Kaggle under
CC BY-NC-SA 4.0. Check that license before any commercial deployment.
