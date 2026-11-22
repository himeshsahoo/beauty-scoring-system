# AI-Based Facial Aesthetic Scoring

This project predicts perceived facial attractiveness from human-rated face images on a 1–10 scale. Predictions represent patterns learned from human rating datasets (SCUT-FBP5500), rather than objective measures of beauty, personal worth, health, or fitness for any role.

---

## Performance Benchmark (Test Set Evaluation)

Evaluated on the held-out test split ($N = 822$ independent frontal portraits):

| Metric | Baseline (Untrained CNN) | Fine-Tuned EfficientNet-B0 (RTX 3050 Ti) | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE** | 4.3816 | **0.5336** | **-87.8% error reduction** |
| **RMSE** | 4.6512 | **0.7133** | **-84.7% error reduction** |
| **$R^2$ Score** | -7.7224 | **+0.7949** | **Strong predictive power** |
| **Pearson Correlation ($r$)** | 0.3553 | **0.8920** | **+151% ranking alignment** |
| **Spearman Rank Correlation** | 0.5025 | **0.8776** | **+74.6% monotonic rank correlation** |
| **Grad-CAM & Occlusion Agreement** | ~0.00 (Low) | **0.50 (High Agreement)** | **Consistent facial saliency** |

---

## Interpretability & Heatmap Framework

To replace inaccurate and diffuse heatmaps from simple models, the pipeline incorporates multiple high-resolution saliency mechanisms:

1. **Hierarchical Multi-Scale Saliency (Default / Recommended)**: Combines mid-level convolutional representations (facial geometry: eyes, nose bridge, lip contour) with late-level semantic features (overall symmetry and facial proportions).
2. **Grad-CAM++**: Utilizes second- and third-order positive gradient weighting to accurately localize multiple distinct facial features simultaneously.
3. **Pre-Sigmoid Latent Backpropagation**: Calculates gradients with respect to latent unbounded regression scores rather than saturating post-sigmoid probabilities, eliminating gradient vanishing.
4. **Vectorized GPU Occlusion Sensitivity**: Generates 36-patch sensitivity maps using batched tensor forward passes on the GPU (10× faster than sequential loops).
5. **Facial Zone Analysis**: Automatically calculates relative aesthetic saliency concentration across key facial zones: **Eyes**, **Mouth / Lips**, **Hair**, and **Central facial structure**.
6. **Robust Percentile Normalization**: Prevents hotspot blowout by clipping outliers at the $2^{\text{nd}}$ and $98^{\text{th}}$ percentiles.

---

## GPU Acceleration (NVIDIA GeForce RTX 3050 Ti)

Optimized specifically for Ampere architecture GPUs (e.g., RTX 3050 Ti Laptop GPU with 4GB VRAM):

- **Automatic Mixed Precision (AMP)**: Uses `torch.amp.autocast('cuda')` with `GradScaler` for reduced VRAM consumption.
- **TensorFloat-32 (TF32)**: Automatically enables TF32 execution on Tensor Cores (`allow_tf32 = True`).
- **cuDNN Benchmark**: Activates `torch.backends.cudnn.benchmark = True` for auto-tuned convolution kernels.
- **VRAM Footprint**: Inference peak memory is ~214.5 MB (well within the 4.0 GB VRAM limit).
- **Throughput**: ~494 FPS in batched inference on RTX 3050 Ti.

---

## What the Pipeline Does

```text
Input image -> Face Detection (MTCNN/Haar) -> Crop & Alignment -> Backbone Feature Extraction -> Regression Head -> Score (1-10) -> Saliency Analysis (Grad-CAM / Grad-CAM++ / Occlusion)
```

The pipeline enforces face cropping before feature extraction to eliminate background and clothing shortcuts. Preprocessing logic is unified across training, evaluation, and inference.

---

## Dataset

The default dataset adapter supports **SCUT-FBP5500**, consisting of 5,500 frontal portraits with human aesthetic ratings on a 1–5 scale. Dataset scores are mapped to the 1–10 scale via:

$$\text{score}_{1\text{--}10} = 1 + (\text{rating}_{1\text{--}5} - 1) \times \frac{9}{4}$$

> [!IMPORTANT]
> The SCUT-FBP5500 dataset is for non-commercial academic research only. Comply with the original dataset terms and license.

---

## Supported Architectures

- `efficientnet` - EfficientNet-B0 backbone (default, recommended)
- `resnet50` - ResNet-50 deep residual backbone
- `resnet18` - ResNet-18 lightweight residual backbone
- `mobilenetv3` - MobileNetV3-Small for edge / low-power deployment
- `mobilenetv2` - MobileNetV2 inverted residual backbone
- `baseline` - 3-layer CNN fallback

Predictions are bounded via:
$$\text{score} = 1 + 9 \times \sigma(\text{raw\_output})$$

---

## Installation

Python 3.10+ and a CUDA-capable GPU (such as RTX 3050 Ti) are recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify your environment and GPU setup:
```powershell
python scripts/verify_gpu_env.py
```

---

## Data Preparation

Download and preprocess the dataset (face detection, alignment, cropping, manifest generation):

```powershell
python src/download_dataset.py
python src/prepare_data.py
```

---

## Training

Fine-tune a pretrained model on your GPU:

```powershell
# Single model fine-tuning (EfficientNet-B0):
python src/train.py --model efficientnet

# Ensemble training (5 independent random seeds):
python src/train.py --model efficientnet --ensemble-size 5
```

Training features:
- Identity-aware train/val/test splits (no identity leakage across splits).
- Two-stage training: freeze backbone during warmup epochs, then unfreeze for fine-tuning.
- Cosine Annealing learning rate schedule with minimum LR floor.
- Gradient clipping (`max_norm = 1.0`).
- Combined `AestheticLoss` (Smooth L1 + Pearson correlation ranking regularization).
- Automatic mixed precision (`torch.amp.autocast('cuda')`).

---

## Evaluation

Run full test set evaluation:

```powershell
python src/evaluate.py --checkpoint models/efficientnet_best.pth
```

To include sensitivity summaries and sanity check visualizations:
```powershell
python src/evaluate.py --checkpoint models/efficientnet_best.pth --sanity-check 6 --sensitivity-samples 3
```

Generated outputs include:
- `results/test_metrics.csv` (MAE, RMSE, $R^2$, Pearson, Spearman)
- `results/test_predictions.csv`
- `results/test_diagnostics.png` (predicted vs. actual, residuals, distributions)
- `results/ablation_summary.csv`
- `results/background_sensitivity_summary.csv`
- `results/quality_sensitivity_summary.csv`

---

## Inference and Web UI

### Command-Line Inference

```powershell
# Standard inference:
python src/predict.py --image path/to/image.jpg --checkpoint models/efficientnet_best.pth

# Inference with custom saliency method and colormap:
python src/predict.py --image path/to/image.jpg --checkpoint models/efficientnet_best.pth --method hierarchical --colormap turbo --alpha 0.40
```

CLI options:
- `--method`: `hierarchical` (default), `plus_plus`, `gradcam`
- `--colormap`: `turbo` (default), `inferno`, `jet`, `viridis`, `magma`
- `--alpha`: Saliency blend factor (default: `0.38`)
- `--landmarks`: Optional landmark PTS/TXT file for facial contour visualization
- `--face-index`: Target face index when multiple faces are detected

### Streamlit Web Application

Launch the interactive web app:

```powershell
streamlit run app/app.py
```

Features in the Streamlit UI:
- **Interactive Saliency Controls**: Select between Hierarchical, Grad-CAM++, and Standard Grad-CAM.
- **Colormap & Opacity Selectors**: Live adjustments of heatmap colormaps (Turbo, Inferno, Viridis, Jet) and transparency.
- **Facial Saliency Breakdown**: Real-time regional importance percentages (Eyes, Mouth, Hair, Center).
- **Ablation & Sensitivity Tabs**: Facial feature masking, background replacement tests, and image quality robustness tables.
- **Ensemble Uncertainty**: When $\ge 5$ checkpoints are available, reports model agreement and standard deviation.

---

## Benchmarks & Testing

Run the automated test suite (18 unit and integration tests):
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Run latency and throughput benchmarks:
```powershell
python scripts/benchmark_inference.py
```

---

## Project Structure

```text
ai-facial-aesthetic-scoring/
├── app/
│   └── app.py                     # Streamlit interactive web application
├── src/
│   ├── common.py                  # Core models, loss functions, transforms, detectors
│   ├── train.py                   # GPU training loop with AMP and CosineAnnealing
│   ├── evaluate.py                # Test set evaluation and interpretability diagnostics
│   ├── predict.py                 # CLI inference and saliency export
│   ├── analysis.py                # Grad-CAM, Grad-CAM++, Occlusion, and regional metrics
│   ├── prepare_data.py            # Face detection, landmark alignment, dataset manifest
│   └── download_dataset.py        # Dataset downloader
├── scripts/
│   ├── verify_gpu_env.py          # GPU environment verification script
│   └── benchmark_inference.py     # RTX 3050 Ti latency and throughput benchmark
├── tests/
│   ├── test_models.py             # Model forward/backward and freezing unit tests
│   ├── test_heatmap.py            # Grad-CAM and Grad-CAM++ unit tests
│   ├── test_dataset.py            # Dataset, DataLoader, and transform unit tests
│   └── test_integration.py        # End-to-end inference and sensitivity tests
├── models/                        # Saved model checkpoints (*_best.pth)
├── results/                       # Metrics, diagnostics plots, and sanity artifacts
├── data/                          # Dataset images, manifests, and face crops
├── config.yaml                    # Global configuration and hyperparameters
└── requirements.txt               # Project dependencies
```

---

## Limitations & Ethical Considerations

Facial aesthetic scoring reflects subjective human rating patterns present in historical training datasets. It is **not** an objective measure of beauty, character, health, personality, or human value.

Models trained on facial aesthetics can encode demographic, lighting, pose, facial expression, and camera-quality biases. Grad-CAM and occlusion heatmaps highlight image regions that correlated with model activation; they do not establish causal relationships.

**Strict Usage Prohibition:** Do not use this tool for hiring, admissions, dating, credit, insurance, healthcare, law enforcement, or any high-stakes automated decision. Always obtain explicit consent before processing portraits, minimize image retention, and adhere to applicable privacy regulations.
