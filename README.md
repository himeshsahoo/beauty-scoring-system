# AI-Based Facial Aesthetic Scoring Using Deep Learning

This research project predicts **perceived facial attractiveness learned from human-rated facial images** on a 1â€“10 scale. It is not a measurement of objective beauty, worth, health, personality, or suitability for any decision.

## Dataset and target

The default adapter supports **SCUT-FBP5500**: 5,500 frontal portraits (2,000 Asian female, 2,000 Asian male, 750 Caucasian female, 750 Caucasian male). Its official release states that 60 volunteers rated each image on a 1â€“5 aesthetic scale; this implementation uses the supplied aggregate rating and linearly maps it to 1â€“10: `1 + (rating_1_to_5 - 1) * 9 / 4`. It does not invent ratings. The official dataset is for **non-commercial research only**; review its terms and cite Liang et al. (ICPR 2018). Source: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

The download archiveâ€™s exact internal labels layout has varied in mirrors. `prepare_data.py` locates a ratings CSV/TXT, validates its 1â€“5 values, and stops rather than producing a fake manifest if it cannot parse labels. SCUT-FBP5500 is primarily one portrait per subject; its image stem is used as the identity grouping. If a replacement dataset has repeated identities, add a real `identity` column to `data/manifest.csv` before splitting.

## Architecture

`prepare_data.py` loads images, detects faces with MTCNN when the optional `facenet-pytorch` package is supported by the active Python version, and uses an OpenCV Haar fallback otherwise. It uses the provided SCUT landmark files when available to align the face, crops a tighter face region with limited context, and stores the source and crop paths in the curated manifest. Training uses an identity-aware group split (70/15/15 by default), train-only facial augmentation, ImageNet normalization, bounded regression, Adam, ReduceLROnPlateau, freeze-then-fine-tune training, and early stopping on validation MAE.

Models: `baseline` (small CNN fallback), `efficientnet` (EfficientNet-B0), `resnet50`, and `mobilenetv2`. Transfer models replace their classification layer with one regression output and use the bounded form `1 + 9 * sigmoid(raw_output)`. Checkpoints use `models/<model>_best.pth` for single runs or `models/<model>_seed<seed>_best.pth` for ensembles.

## Install

Python 3.10+ is required. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset setup

```powershell
python src/download_dataset.py
python src/prepare_data.py
```

This downloads from the official releaseâ€™s Google Drive identifier. If access fails, manually download the official archive linked above, extract it under `data/SCUT-FBP5500_v2/`, adjust `config.yaml` paths if required, then run `prepare_data.py`.

## Train and compare

```powershell
python src/train.py --model baseline
python src/train.py --model efficientnet
python src/train.py --model resnet50
python src/train.py --model mobilenetv2
```

Each run reports device/GPU, dataset and split sizes, parameter count, and epoch metrics; it writes a checkpoint, training CSV, and loss curve. Set `ensemble_size` in `config.yaml` or pass `--ensemble-size` to train multiple independently initialized checkpoints with the same split. Pretrained ImageNet weights may be downloaded by torchvision on the first transfer-learning run.

## Evaluate

```powershell
python src/evaluate.py --checkpoint models/efficientnet_best.pth
```

It produces MAE, MSE, RMSE, RÂ², Pearson, Spearman plus `results/test_metrics.csv` and `results/test_diagnostics.png` (scatter, residual, error, and score-distribution plots). Add `--sanity-check 6` to save `results/sanity_check.png` with original image, face crop, and Grad-CAM for several test samples.

## Inference and app

```powershell
python src/predict.py --image path/to/image.jpg --checkpoint models/efficientnet_best.pth
streamlit run app/app.py
```

Inference saves `results/inference/face_crop.png`, `gradcam.png`, and `prediction.json`. When multiple faces are detected, CLI inference requires `--face-index`; the app requires an explicit selection. If multiple seeded checkpoints are present, uncertainty is the standard deviation across models and the app shows model agreement. It is only an approximate model-disagreement signal, not calibrated confidence. With a single checkpoint, uncertainty is intentionally reported unavailable.

## Project structure

```
app/           Streamlit UI
data/          downloaded dataset, face crops, manifests
models/        best validation-MAE checkpoints
notebooks/     optional exploration
results/       metrics, plots, and inference artifacts
src/           preparation, training, evaluation, inference
```

## Limitations and ethical use

Beauty is subjective. Dataset labels capture a particular group of human raters and may encode cultural, demographic, and historical bias. Performance can vary by demographic group, lighting, expression, camera quality, pose, makeup, and image editing. Grad-CAM only shows regions that influenced a particular model output; it does not prove causal reasoning or that any feature is objectively beautiful.

Do not use this system for hiring, admissions, dating, insurance, medical care, law enforcement, or any other high-impact decision. Obtain consent before processing images, minimize retention, and comply with the datasetâ€™s non-commercial restriction and local privacy law.

