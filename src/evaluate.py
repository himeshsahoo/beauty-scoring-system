from __future__ import annotations

import argparse
import ast
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from analysis import (
    load_model_bundle,
    occlusion_sensitivity,
    predict_ablation_table,
    predict_background_table,
    predict_face,
    predict_quality_table,
)
from common import AestheticDataset, FaceDetector, device_info, discover_checkpoints, load_config, read_landmarks, resolve, unit_to_score


def _parse_box(value) -> tuple[int, int, int, int] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
            return tuple(int(v) for v in parsed)
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(v) for v in value)
    return None


def _load_models(checkpoint_arg: Path, device: torch.device):
    if checkpoint_arg.is_dir():
        checkpoint_paths = discover_checkpoints(checkpoint_arg)
    else:
        checkpoint_paths = [checkpoint_arg]
    if not checkpoint_paths:
        raise FileNotFoundError('No checkpoints found to evaluate.')
    models, ensemble_enabled, loaded_paths = load_model_bundle(checkpoint_paths, device, minimum_ensemble=5)
    return models, ensemble_enabled, loaded_paths


def _evaluate_split(models, cfg, device, frame: pd.DataFrame, split_name: str, result_dir: Path):
    loader = DataLoader(
        AestheticDataset(frame, cfg['image_size'], train=False),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=device.type == 'cuda',
    )
    actual_scores = []
    predicted_scores = []
    with torch.no_grad():
        for x, target_unit, _ in loader:
            x = x.to(device)
            predictions = []
            for model in models:
                raw = model(x)
                predictions.append(unit_to_score(torch.sigmoid(raw)).cpu())
            predicted_batch = torch.stack(predictions, dim=0).mean(dim=0)
            predicted_scores.extend(predicted_batch.numpy())
            actual_scores.extend(unit_to_score(target_unit).numpy())

    actual = np.asarray(actual_scores, dtype=np.float32)
    predicted = np.clip(np.asarray(predicted_scores, dtype=np.float32), 1, 10)
    errors = predicted - actual
    metrics = {
        'MAE': mean_absolute_error(actual, predicted),
        'RMSE': mean_squared_error(actual, predicted) ** 0.5,
        'R2': r2_score(actual, predicted),
        'Pearson': pearsonr(actual, predicted).statistic,
        'Spearman': spearmanr(actual, predicted).statistic,
    }
    print(f'\n{split_name.upper()} METRICS')
    print(pd.Series(metrics).to_string())

    pd.DataFrame([metrics]).to_csv(result_dir / f'{split_name}_metrics.csv', index=False)
    pd.DataFrame({'actual': actual, 'predicted': predicted, 'error': errors}).to_csv(result_dir / f'{split_name}_predictions.csv', index=False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].scatter(actual, predicted, alpha=0.55)
    axes[0].plot([1, 10], [1, 10], 'r--')
    axes[0].set(xlabel='Human-rated target (1-10)', ylabel='Prediction', title='Predicted vs actual')
    axes[1].scatter(predicted, errors, alpha=0.55)
    axes[1].axhline(0, color='r', ls='--')
    axes[1].set(xlabel='Prediction', ylabel='Error', title='Residuals')
    axes[2].hist(errors, bins=25)
    axes[2].set(xlabel='Prediction error', title='Error distribution')
    axes[3].hist(actual, bins=20, alpha=0.55, label='actual')
    axes[3].hist(predicted, bins=20, alpha=0.55, label='predicted')
    axes[3].set(xlabel='Score', title='Score distribution')
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(result_dir / f'{split_name}_diagnostics.png', dpi=160)
    plt.close(fig)

    return metrics


def _aggregate_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    combined['Difference from Original'] = combined['Difference from Original'].str.replace('+', '', regex=False)
    combined['Difference from Original'] = combined['Difference from Original'].astype(float)
    summary = combined.groupby('Variant', as_index=False).agg(
        Predicted_Score=('Predicted Score', 'mean'),
        Difference_from_Original=('Difference from Original', 'mean'),
    )
    summary['Predicted Score'] = summary['Predicted_Score'].map(lambda value: round(value, 2))
    summary['Difference from Original'] = summary['Difference_from_Original'].map(lambda value: f'{value:+.2f}')
    return summary[['Variant', 'Predicted Score', 'Difference from Original']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--sanity-check', type=int, default=0)
    parser.add_argument('--sensitivity-samples', type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_info()
    result_dir = resolve(cfg, 'results')
    result_dir.mkdir(parents=True, exist_ok=True)

    test_path = resolve(cfg, 'data/test.csv')
    if not test_path.exists():
        raise FileNotFoundError('Test split missing. Run train.py first.')
    test_frame = pd.read_csv(test_path)

    models, ensemble_enabled, loaded_paths = _load_models(Path(args.checkpoint), device)
    print(f'Evaluating {len(loaded_paths)} checkpoint(s):')
    for path in loaded_paths:
        print(f'  - {path}')
    if not ensemble_enabled:
        print('Ensemble not trained — uncertainty unavailable')

    split_metrics = _evaluate_split(models, cfg, device, test_frame, 'test', result_dir)

    detector = FaceDetector(cfg)
    sensitivity_samples = test_frame.head(max(1, args.sensitivity_samples)).copy()
    ablation_tables = []
    background_tables = []
    quality_tables = []
    occlusion_maps = []
    gradcam_maps = []
    correlations = []

    for index, (_, sample) in enumerate(sensitivity_samples.iterrows()):
        source_path = Path(sample.get('source_path', sample.crop_path))
        source_image = plt.imread(source_path)
        if source_image.ndim == 2:
            source_image = np.repeat(source_image[..., None], 3, axis=2)
        if source_image.dtype != np.uint8:
            source_image = (np.clip(source_image, 0, 1) * 255).astype(np.uint8)
        if source_image.shape[-1] == 4:
            source_image = source_image[..., :3]

        crop_path = Path(sample.crop_path)
        crop_image = plt.imread(crop_path)
        if crop_image.ndim == 2:
            crop_image = np.repeat(crop_image[..., None], 3, axis=2)
        if crop_image.dtype != np.uint8:
            crop_image = (np.clip(crop_image, 0, 1) * 255).astype(np.uint8)
        if crop_image.shape[-1] == 4:
            crop_image = crop_image[..., :3]

        box = _parse_box(sample.get('face_box'))
        if box is None:
            boxes = detector.detect(source_image)
            if not boxes:
                continue
            box = boxes[0]
        landmarks = read_landmarks(sample.get('landmark_path'))

        ablation_tables.append(predict_ablation_table(models, detector, cfg, device, source_image, box, landmarks=landmarks))
        background_tables.append(predict_background_table(models, detector, cfg, device, source_image, box, landmarks=landmarks))
        quality_tables.append(predict_quality_table(models, cfg, device, crop_image))

        crop_box = detector.crop(source_image, box)
        summary, gradcam_overlay, gradcam_heatmap = predict_face(crop_box, models, cfg, device)
        occlusion_grid, occlusion_heatmap, base_score = occlusion_sensitivity(models, cfg, device, crop_box)
        gradcam_resized = cv2.resize(gradcam_heatmap, (occlusion_heatmap.shape[1], occlusion_heatmap.shape[0]), interpolation=cv2.INTER_CUBIC)
        correlations.append(float(np.corrcoef(gradcam_resized.reshape(-1), occlusion_heatmap.reshape(-1))[0, 1]))
        occlusion_maps.append(occlusion_heatmap)
        gradcam_maps.append(gradcam_resized)

        sample_dir = result_dir / 'sanity'
        sample_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(source_image)
        axes[0].set_title(f'Original\nactual {sample.score_1_10:.2f}')
        axes[0].axis('off')
        axes[1].imshow(crop_box)
        axes[1].set_title(f'Face crop\npred {summary.score:.2f}')
        axes[1].axis('off')
        axes[2].imshow(gradcam_overlay)
        axes[2].set_title('Grad-CAM')
        axes[2].axis('off')
        fig.tight_layout()
        fig.savefig(sample_dir / f'sanity_{index + 1}.png', dpi=160)
        plt.close(fig)

    if ablation_tables:
        ablation_summary = _aggregate_tables(ablation_tables)
        background_summary = _aggregate_tables(background_tables)
        quality_summary = _aggregate_tables(quality_tables)
        ablation_summary.to_csv(result_dir / 'ablation_summary.csv', index=False)
        background_summary.to_csv(result_dir / 'background_sensitivity_summary.csv', index=False)
        quality_summary.to_csv(result_dir / 'quality_sensitivity_summary.csv', index=False)

        print('\nABLATION SUMMARY')
        print(ablation_summary.to_string(index=False))
        print('\nBACKGROUND SUMMARY')
        print(background_summary.to_string(index=False))
        print('\nQUALITY SUMMARY')
        print(quality_summary.to_string(index=False))
        print(f'\nGrad-CAM vs occlusion mean correlation: {float(np.mean(correlations)):.2f}')

        target_size = (cfg["image_size"], cfg["image_size"])
        occlusion_resized = [cv2.resize(m, target_size, interpolation=cv2.INTER_LINEAR) for m in occlusion_maps]
        gradcam_resized_list = [cv2.resize(m, target_size, interpolation=cv2.INTER_LINEAR) for m in gradcam_maps]
        occlusion_mean = np.mean(np.stack(occlusion_resized, axis=0), axis=0)
        gradcam_mean = np.mean(np.stack(gradcam_resized_list, axis=0), axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(gradcam_mean, cmap='turbo')
        axes[0].set_title('Average Grad-CAM')
        axes[0].axis('off')
        axes[1].imshow(occlusion_mean, cmap='turbo')
        axes[1].set_title('Average occlusion sensitivity')
        axes[1].axis('off')
        fig.tight_layout()
        fig.savefig(result_dir / 'average_interpretability_maps.png', dpi=160)
        plt.close(fig)

    if args.sanity_check > 0:
        print(f'\nSanity-check images saved under {result_dir / "sanity"}')

    return split_metrics


if __name__ == '__main__':
    main()
