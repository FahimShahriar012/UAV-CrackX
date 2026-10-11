# UAV-CrackX Pavement Crack Segmentation

A PyTorch baseline for binary pavement-crack segmentation on the UAV-CrackX dataset. This repository contains the model implementation, recorded training metrics, selected checkpoint, validation workbook, and the benchmark prediction archive.

## Results

The following results come from two different evaluation sources and must not be compared as if they were the same split.

### Official competition leaderboard

The leaderboard row named **Fahim** showed a last submission date of **2026-10-11** and these scores:

| Metric | Score shown |
|---|---:|
| mIoU | 79.1600 |
| Crack F1 | 75.8600 |
| Crack IoU | 61.1100 |
| Crack Precision | 81.0100 |
| Crack Recall | 71.3300 |
| aAcc | 97.3300 |

These values are transcribed from the competition leaderboard screenshot. The repository's local metric code reports `pixelAccuracy`; the exact relationship between that metric and the benchmark's `aAcc` has not been verified.

### Local holdout validation

Training ran for 60 epochs. The checkpoint was selected by the highest Crack IoU on a labeled holdout split; the best epoch was 55.

| Metric | Best checkpoint, epoch 55 | Final epoch 60 |
|---|---:|---:|
| mIoU | 77.9857% | 77.1643% |
| Crack IoU | 59.0930% | 57.5669% |
| Crack Precision | 80.2706% | 79.7389% |
| Crack Recall | 69.1342% | 67.4301% |
| Crack F1 | 74.2874% | 73.0698% |
| Pixel accuracy | 97.0132% | 96.8980% |

The full 60-epoch log is in [`artifacts/training_metrics.json`](artifacts/training_metrics.json), with a formatted summary in [`artifacts/validation_results.xlsx`](artifacts/validation_results.xlsx). These local validation scores are not competition scores.

## Dataset

The dataset has three scale folders: `UAV-CrackX4`, `UAV-CrackX8`, and `UAV-CrackX16`. Each scale contains 400 labeled training images and masks, plus 100 unlabeled benchmark images. Images and masks are 672 x 378 pixels. The 300 benchmark images have no public ground-truth masks in the provided dataset copy.

The dataset is not included in this repository. Obtain it from the authorized dataset or competition source, review its license and citation requirements, then point `--data-root` to the extracted `UAV-CrackX` directory. Expected layout:

```text
UAV-CrackX/
  gtFine/train/UAV-CrackX4/*.png
  gtFine/train/UAV-CrackX8/*.png
  gtFine/train/UAV-CrackX16/*.png
  leftImg8bit/train/UAV-CrackX4/*.jpg
  leftImg8bit/train/UAV-CrackX8/*.jpg
  leftImg8bit/train/UAV-CrackX16/*.jpg
  leftImg8bit/val/UAV-CrackX4/*.jpg
  leftImg8bit/val/UAV-CrackX8/*.jpg
  leftImg8bit/val/UAV-CrackX16/*.jpg
```

Training matches input and mask files by filename stem. RGB input values are normalized to [0, 1]; positive mask pixels are converted to binary foreground targets. Training augmentation consists of horizontal and vertical flips, each independently sampled with probability 0.5.

## Model

The implementation is a compact 2D U-Net with a 32-channel base width. It has four encoder stages, a bottleneck, bilinear upsampling with skip connections, batch normalization, and a one-channel logits head. It is trained using binary cross-entropy with logits plus soft Dice loss. The default prediction threshold is 0.5. See [`uav_crack/model.py`](uav_crack/model.py), [`uav_crack/data.py`](uav_crack/data.py), and [`uav_crack/metrics.py`](uav_crack/metrics.py).

## Setup on Windows

Use Python 3.10 or newer. Install a PyTorch build that matches the machine's CPU or CUDA setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). PyTorch is installed separately because the correct build depends on the hardware. Then install the remaining packages:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install the matching PyTorch build using the official selector first.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands call the virtual environment's Python directly, so PowerShell script activation is not required.

## Train

The recorded run used 60 epochs, batch size 2, a 15% holdout per scale, seed 42, AdamW, learning rate 0.001, and weight decay 0.0001. The saved best checkpoint was epoch 55. The CLI defaults to batch size 4, so pass `--batch-size 2` to reproduce the recorded run settings.

```powershell
$dataset = "D:\path\to\UAV-CrackX"
$python = ".\.venv\Scripts\python.exe"
& $python -m uav_crack train `
  --data-root $dataset `
  --output runs `
  --epochs 60 `
  --batch-size 2
```

The split is shuffled independently within each scale using seed 42. The run log records 1,020 training images and 180 holdout images. The model is selected by holdout Crack IoU. The training code saves `runs/latest.pt` at every epoch and supports resuming with `--resume runs\latest.pt`.

## Predict and submission files

A ready-to-submit archive is provided at [`artifacts/result.zip`](artifacts/result.zip). It contains exactly 300 PNG masks under a `result/` directory. The archive was checked for matching input basenames, 672 x 378 dimensions, and mask values limited to 0 (background) and 1 (crack).

To regenerate predictions with the selected checkpoint:

```powershell
$dataset = "D:\path\to\UAV-CrackX"
$python = ".\.venv\Scripts\python.exe"
& $python -m uav_crack predict `
  --data-root $dataset `
  --checkpoint artifacts\best.pt `
  --output result.zip `
  --bundle-output ""
```

`--bundle-output ""` skips creating a larger archive containing a copy of the dataset. Upload only the benchmark-required `result.zip` to the competition portal. The portal's official result is separate from the local holdout metrics above.

## Repository contents

- `uav_crack/` - training, data, model, metric, and prediction code.
- `artifacts/best.pt` - selected model checkpoint from epoch 55.
- `artifacts/training_metrics.json` - per-epoch local training and holdout metrics.
- `artifacts/validation_results.xlsx` - formatted epoch results and summary.
- `artifacts/result.zip` - 300 benchmark prediction masks.
- `requirements.txt` - non-PyTorch Python dependencies.
- `LICENSE` - repository software license. It does not change the dataset's own license or usage terms.

## Reproducibility notes

The checkpoint stores model weights, base width, prediction threshold, selected epoch, best Crack IoU, and selected-epoch metrics. The training command-line settings and hardware are not fully serialized in the checkpoint; the values above are taken from the project code, epoch log, and run session record. No per-scale validation breakdown or multi-seed experiment is included.
