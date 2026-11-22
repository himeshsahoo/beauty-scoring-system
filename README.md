# AI-Based Facial Aesthetic Scoring

This project predicts perceived facial attractiveness from human-rated face images on a 1-10 scale. It is a learned pattern from the training data, not an objective measure of beauty, worth, health, personality, or suitability for any decision.

## What the pipeline does

Input image -> face detection -> face crop and optional alignment -> model inference -> prediction -> Grad-CAM and sensitivity analysis.

The project is designed to reduce background leakage by using face crops instead of full images. The training and inference pipelines share the same preprocessing logic.

## Dataset

The default dataset adapter supports SCUT-FBP5500, a set of 5,500 frontal portraits with human aesthetic ratings on a 1-5 scale. This code maps the dataset score to 1-10 using:

```text
1 + (rating_1_to_5 - 1) * 9 / 4
```

The dataset is for non-commercial research only. Please review the original dataset terms and cite the SCUT-FBP5500 paper when using it.

If the labels archive layout differs in your download, `src/prepare_data.py` will search for a ratings CSV/TXT, validate the numeric range, and stop rather than inventing labels.

## Performance Benchmark (Test Set Evaluation)

| Metric | Baseline (Untrained CNN) | Fine-Tuned EfficientNet-B0 (RTX 3050 Ti) | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE** | 4.3816 | **0.5336** | **-87.8% error reduction** |
| **RMSE** | 4.6512 | **0.7133** | **-84.7% error reduction** |
| **R² Score** | -7.7224 | **+0.7949** | **Strong predictive power** |
| **Pearson Correlation ($r$)** | 0.3553 | **0.8920** | **+151% ranking alignment** |
| **Spearman Rank Correlation** | 0.5025 | **0.8776** | **+74.6% monotonic rank correlation** |
| **Grad-CAM & Occlusion Corr.** | ~0.00 (Low) | **0.50 (High Agreement)** | **Consistent facial saliency** |

## Saliency Heatmap & Interpretability

To overcome noisy and inaccurate heatmaps from simple baselines, this system provides three advanced interpretability mechanisms:

1. **Hierarchical Multi-Scale Saliency (Default / Recommended)**: Aggregates mid-level facial geometry (eyes, lips, jawline) with high-level holistic aesthetic features.
2. **Grad-CAM++**: Uses higher-order gradient weighting to localize multiple distinct facial features simultaneously.
3. **Pre-Sigmoid Latent Backpropagation**: Gradients are computed with respect to latent unbounded aesthetic representations rather than saturating post-sigmoid scores.
4. **Vectorized GPU Occlusion Sensitivity**: Computes grid-based occlusion sensitivity in batched GPU forward passes.

## GPU Acceleration (NVIDIA RTX 3050 Ti)

- **Mixed Precision (AMP)**: `torch.amp.autocast('cuda')` and `GradScaler` provide ~50% VRAM savings.
- **TensorFloat-32 (TF32)**: Enabled for Ampere architecture matmul acceleration.
- **Inference Speed**: ~494 FPS in batched inference, ~58 ms latency for single image forward passes.

## Models

The project supports these regressors:

- `baseline` - small CNN fallback
- `efficientnet` - EfficientNet-B0 backbone
- `resnet50` - ResNet-50 backbone
- `mobilenetv2` - MobileNetV2 backbone

The transfer-learning models replace the classifier head with a single regression output. Predictions are bounded with:

```text
score = 1 + 9 * sigmoid(raw_output)
```

To support ensemble uncertainty, the repo can train multiple independently seeded checkpoints and average their predictions at inference time.

## Install

Python 3.10+ is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare data

```powershell
python src/download_dataset.py
python src/prepare_data.py
```

This downloads the dataset, detects faces, optionally aligns them using the provided landmarks, and writes a manifest plus cropped face images.

## Train

Single model:

```powershell
python src/train.py --model efficientnet
```

Ensemble training:

```powershell
python src/train.py --model efficientnet --ensemble-size 5
```

Training uses an identity-aware split, train-only augmentation, bounded regression targets, freeze-then-fine-tune behavior, and early stopping on validation MAE.

## Evaluate

```powershell
python src/evaluate.py --checkpoint models
```

Evaluation reports MAE, RMSE, R2, Pearson correlation, and Spearman correlation. It also writes predicted-vs-actual, residual, and score-distribution plots. For a small interpretability check, add:

```powershell
python src/evaluate.py --checkpoint models --sanity-check 6 --sensitivity-samples 3
```

This saves a sanity-check panel plus ablation, background, quality, and occlusion sensitivity summaries.

## Inference and app

CLI inference:

```powershell
python src/predict.py --image path/to/image.jpg --checkpoint models
```

Streamlit app:

```powershell
streamlit run app/app.py
```

If the repo contains at least five trained checkpoints, the app and CLI report ensemble prediction, ensemble uncertainty, and model agreement. If fewer checkpoints are available, uncertainty is reported as unavailable instead of being faked.

## Outputs

- `results/test_metrics.csv` - evaluation metrics
- `results/test_diagnostics.png` - test-set plots
- `results/sanity/` - optional sanity-check examples
- `results/inference/` - CLI inference artifacts
- `results/*.csv` - training and sensitivity summaries

## Project structure

```text
app/        Streamlit UI
src/        dataset prep, training, evaluation, inference, analysis
models/     trained checkpoints
results/    metrics, plots, and inference outputs
data/       downloaded dataset, crops, and manifests
notebooks/  optional exploration
```

## Limitations

Beauty is subjective and culturally shaped. The model learns patterns present in the training data and may reflect demographic, lighting, pose, expression, camera-quality, and dataset biases. Grad-CAM and occlusion maps show regions that influenced a model output; they do not prove causal reasoning or objective beauty features.

Do not use this system for hiring, admissions, dating, insurance, medical care, law enforcement, or any other high-impact decision. Obtain consent before processing images, minimize retention, and comply with the dataset's non-commercial restriction and local privacy law.
