from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import FaceDetector, load_checkpoint, load_config, make_transform


@dataclass(frozen=True)
class PredictionSummary:
    score: float
    scores: list[float]
    uncertainty: float | None
    agreement: str


def score_agreement_label(std: float | None) -> str:
    if std is None:
        return "Unavailable"
    if std < 0.35:
        return "High"
    if std < 0.75:
        return "Moderate"
    return "Low"


def unit_to_score(score_unit: np.ndarray | torch.Tensor | float) -> np.ndarray | torch.Tensor | float:
    return 1.0 + 9.0 * score_unit


def load_model_bundle(checkpoint_paths: Iterable[Path], device: torch.device, minimum_ensemble: int = 5):
    paths = [Path(path) for path in checkpoint_paths]
    if len(paths) >= minimum_ensemble:
        models = [load_checkpoint(path, device)[0] for path in paths]
        return models, True, paths
    if paths:
        models = [load_checkpoint(paths[0], device)[0]]
        return models, False, [paths[0]]
    return [], False, []


def _tensor_from_rgb(image_rgb: np.ndarray, size: int, train: bool = False) -> torch.Tensor:
    return make_transform(size, train=train)(Image.fromarray(image_rgb)).unsqueeze(0)


def score_from_model(model: torch.nn.Module, image_rgb: np.ndarray, cfg: dict, device: torch.device) -> float:
    model.eval()
    tensor = _tensor_from_rgb(image_rgb, cfg["image_size"] ).to(device)
    with torch.no_grad():
        raw = model(tensor)
        return float(unit_to_score(torch.sigmoid(raw)).item())


def score_from_models(models: list[torch.nn.Module], image_rgb: np.ndarray, cfg: dict, device: torch.device) -> PredictionSummary:
    if not models:
        raise ValueError("No models available for prediction.")
    scores = [score_from_model(model, image_rgb, cfg, device) for model in models]
    mean_score = float(np.mean(scores))
    uncertainty = float(np.std(scores)) if len(scores) > 1 else None
    agreement = score_agreement_label(uncertainty)
    return PredictionSummary(score=mean_score, scores=scores, uncertainty=uncertainty, agreement=agreement)


def _representative_model(models: list[torch.nn.Module], scores: list[float], target_score: float):
    if len(models) == 1:
        return models[0]
    index = int(np.argmin(np.abs(np.asarray(scores) - target_score)))
    return models[index]


def gradcam_heatmap(model: torch.nn.Module, image_rgb: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    target = getattr(model, "target_layer", None)
    if target is None:
        modules = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        if not modules:
            raise RuntimeError("No convolutional layer available for Grad-CAM.")
        target = modules[-1]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    hook_forward = target.register_forward_hook(lambda module, inputs, output: activations.append(output))
    hook_backward = target.register_full_backward_hook(lambda module, grad_inputs, grad_outputs: gradients.append(grad_outputs[0]))

    tensor = _tensor_from_rgb(image_rgb, cfg["image_size"]).to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        raw = model(tensor)
        # Backpropagate on latent pre-sigmoid regression score to prevent sigmoid gradient saturation
        raw.sum().backward()
    finally:
        hook_forward.remove()
        hook_backward.remove()

    if not gradients or not activations:
        raise RuntimeError("Failed to capture gradients or activations for Grad-CAM.")

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = np.clip(cam, 0, None)
    denom = cam.max() - cam.min()
    return (cam - cam.min()) / (denom + 1e-8) if denom > 1e-8 else np.zeros_like(cam)


def gradcam_plus_plus_heatmap(model: torch.nn.Module, image_rgb: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    """Compute Grad-CAM++ saliency map for finer facial feature localization."""
    target = getattr(model, "target_layer", None)
    if target is None:
        modules = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        if not modules:
            raise RuntimeError("No convolutional layer available for Grad-CAM++.")
        target = modules[-1]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    hook_forward = target.register_forward_hook(lambda module, inputs, output: activations.append(output))
    hook_backward = target.register_full_backward_hook(lambda module, grad_inputs, grad_outputs: gradients.append(grad_outputs[0]))

    tensor = _tensor_from_rgb(image_rgb, cfg["image_size"]).to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        raw = model(tensor)
        raw.sum().backward()
    finally:
        hook_forward.remove()
        hook_backward.remove()

    if not gradients or not activations:
        raise RuntimeError("Failed to capture gradients or activations for Grad-CAM++.")

    act = activations[0]  # (1, C, H, W)
    grad = gradients[0]   # (1, C, H, W)

    # Positive gradients
    grad_pos = torch.relu(grad)
    grad_2 = grad.pow(2)
    grad_3 = grad.pow(3)

    # Alpha weights for Grad-CAM++
    denom = 2.0 * grad_2 + (act * grad_3).sum(dim=(2, 3), keepdim=True)
    denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
    alpha = grad_2 / denom

    weights = (alpha * grad_pos).sum(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = np.clip(cam, 0, None)
    denom_np = cam.max() - cam.min()
    return (cam - cam.min()) / (denom_np + 1e-8) if denom_np > 1e-8 else np.zeros_like(cam)


def normalize_heatmap(
    heatmap: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Robust percentile-based heatmap normalization to prevent hotspot blowout."""
    if heatmap.size == 0 or np.all(heatmap == 0):
        return heatmap
    v_min = float(np.percentile(heatmap, lower_percentile))
    v_max = float(np.percentile(heatmap, upper_percentile))
    if v_max <= v_min:
        return np.zeros_like(heatmap, dtype=np.float32)
    clipped = np.clip(heatmap, v_min, v_max)
    return ((clipped - v_min) / (v_max - v_min)).astype(np.float32)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.38,
    colormap: str = "turbo",
    threshold: float = 0.15,
) -> np.ndarray:
    """Blend saliency heatmap onto face image with colormap and thresholding."""
    resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    resized = normalize_heatmap(resized)

    cm_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    cv_cmap = cm_map.get(colormap.lower(), cv2.COLORMAP_TURBO)
    color = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv_cmap)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    if threshold > 0:
        # Fade out very low saliency areas so original face remains sharp and clean
        mask = (resized >= threshold).astype(np.float32)[:, :, None]
        blended = (1.0 - alpha * mask) * image_rgb + (alpha * mask) * color
        return np.clip(blended, 0, 255).astype(np.uint8)

    return cv2.addWeighted(image_rgb, 1.0 - alpha, color, alpha, 0)


def hierarchical_saliency_heatmap(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    cfg: dict,
    device: torch.device,
    use_plus_plus: bool = True,
) -> np.ndarray:
    """Multi-scale feature saliency combining fine facial geometry and high-level aesthetics."""
    heatmap_fn = gradcam_plus_plus_heatmap if use_plus_plus else gradcam_heatmap
    late_map = heatmap_fn(model, image_rgb, cfg, device)

    # Attempt to locate a mid-level feature layer
    mid_layer = None
    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "features") and len(model.backbone.features) >= 5:
            mid_layer = model.backbone.features[len(model.backbone.features) // 2]
        elif hasattr(model.backbone, "layer3"):
            mid_layer = model.backbone.layer3[-1]

    if mid_layer is None:
        return late_map

    orig_target = getattr(model, "target_layer", None)
    try:
        model.target_layer = mid_layer
        mid_map = heatmap_fn(model, image_rgb, cfg, device)
    except Exception:
        mid_map = None
    finally:
        if orig_target is not None:
            model.target_layer = orig_target

    if mid_map is None:
        return late_map

    combined = 0.65 * late_map + 0.35 * mid_map
    return normalize_heatmap(combined)


def predict_face(
    image_rgb: np.ndarray,
    models: list[torch.nn.Module],
    cfg: dict,
    device: torch.device,
    method: str = "hierarchical",
    colormap: str = "turbo",
    alpha: float = 0.38,
):
    summary = score_from_models(models, image_rgb, cfg, device)
    representative = _representative_model(models, summary.scores, summary.score)
    if method == "hierarchical":
        heatmap = hierarchical_saliency_heatmap(representative, image_rgb, cfg, device)
    elif method in {"plus_plus", "gradcam++"}:
        heatmap = gradcam_plus_plus_heatmap(representative, image_rgb, cfg, device)
    else:
        heatmap = gradcam_heatmap(representative, image_rgb, cfg, device)
    overlay = overlay_heatmap(image_rgb, heatmap, alpha=alpha, colormap=colormap)
    return summary, overlay, heatmap


def _face_bbox(box: tuple[int, int, int, int], image_shape: tuple[int, int, int], margin: float = 0.0) -> tuple[int, int, int, int]:
    x, y, width, height = box
    pad = int(round(max(width, height) * margin))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_shape[1], x + width + pad)
    y2 = min(image_shape[0], y + height + pad)
    return x1, y1, x2, y2


def _clamp_region(x1: int, y1: int, x2: int, y2: int, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    return (
        max(0, x1),
        max(0, y1),
        min(image_shape[1], x2),
        min(image_shape[0], y2),
    )


def _landmark_array(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None:
        return None
    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        return None
    return landmarks.astype(np.float32)


def _face_mask(image_shape: tuple[int, int, int], box: tuple[int, int, int, int], landmarks: np.ndarray | None = None) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = _landmark_array(landmarks)
    if points is not None and len(points) >= 4:
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
        return mask
    x, y, width, height = box
    center = (int(round(x + width / 2.0)), int(round(y + height / 2.0)))
    axes = (max(1, int(round(width * 0.48))), max(1, int(round(height * 0.60))))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _region_from_landmarks(points: np.ndarray | None, image_shape: tuple[int, int, int], fallback_box: tuple[int, int, int, int], region: str):
    if points is None or len(points) < 68:
        x, y, width, height = fallback_box
        if region == "hair":
            return _clamp_region(x + int(0.08 * width), y, x + int(0.92 * width), y + int(0.33 * height), image_shape)
        if region == "eyes":
            return _clamp_region(x + int(0.16 * width), y + int(0.23 * height), x + int(0.84 * width), y + int(0.48 * height), image_shape)
        if region == "mouth":
            return _clamp_region(x + int(0.22 * width), y + int(0.58 * height), x + int(0.78 * width), y + int(0.86 * height), image_shape)
        if region == "center":
            return _clamp_region(x + int(0.28 * width), y + int(0.28 * height), x + int(0.72 * width), y + int(0.74 * height), image_shape)
        return _clamp_region(x, y, x + width, y + height, image_shape)

    points = points.astype(np.float32)
    if region == "hair":
        face_top = int(np.min(points[:, 1]))
        eye_line = int(np.mean(points[36:48, 1]))
        face_left = int(np.min(points[:, 0]))
        face_right = int(np.max(points[:, 0]))
        return _clamp_region(face_left, face_top, face_right, max(face_top + 1, eye_line - 6), image_shape)
    if region == "eyes":
        left = points[36:42]
        right = points[42:48]
        xs = np.concatenate([left[:, 0], right[:, 0]])
        ys = np.concatenate([left[:, 1], right[:, 1]])
        return _clamp_region(int(xs.min()) - 12, int(ys.min()) - 12, int(xs.max()) + 12, int(ys.max()) + 12, image_shape)
    if region == "mouth":
        mouth = points[48:68]
        return _clamp_region(int(mouth[:, 0].min()) - 12, int(mouth[:, 1].min()) - 12, int(mouth[:, 0].max()) + 12, int(mouth[:, 1].max()) + 12, image_shape)
    if region == "center":
        face_left = int(np.min(points[:, 0]))
        face_right = int(np.max(points[:, 0]))
        face_top = int(np.min(points[:, 1]))
        face_bottom = int(np.max(points[:, 1]))
        width = face_right - face_left
        height = face_bottom - face_top
        return _clamp_region(
            face_left + int(0.27 * width),
            face_top + int(0.28 * height),
            face_left + int(0.73 * width),
            face_top + int(0.74 * height),
            image_shape,
        )
    return _clamp_region(int(points[:, 0].min()), int(points[:, 1].min()), int(points[:, 0].max()), int(points[:, 1].max()), image_shape)


def _apply_region_mask(image_rgb: np.ndarray, region_box: tuple[int, int, int, int], mode: str = "blur") -> np.ndarray:
    x1, y1, x2, y2 = region_box
    result = image_rgb.copy()
    region = result[y1:y2, x1:x2]
    if region.size == 0:
        return result
    if mode == "blur":
        replacement = cv2.GaussianBlur(region, (0, 0), sigmaX=18)
    elif mode == "mean":
        replacement = np.full_like(region, np.round(region.mean(axis=(0, 1))).astype(np.uint8))
    else:
        replacement = np.zeros_like(region)
    result[y1:y2, x1:x2] = replacement
    return result


def _composite_background(image_rgb: np.ndarray, face_mask: np.ndarray, background_mode: str) -> np.ndarray:
    result = image_rgb.copy()
    inverted = face_mask == 0
    if background_mode == "blur":
        blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=24)
        result[inverted] = blurred[inverted]
    elif background_mode == "plain":
        fill = np.round(image_rgb.mean(axis=(0, 1))).astype(np.uint8)
        result[inverted] = fill
    elif background_mode == "dark":
        result[inverted] = 0
    elif background_mode == "light":
        result[inverted] = 255
    return result


def make_ablation_variants(source_rgb: np.ndarray, box: tuple[int, int, int, int], landmarks: np.ndarray | None = None) -> dict[str, np.ndarray]:
    points = _landmark_array(landmarks)
    face_mask = _face_mask(source_rgb.shape, box, points)
    variants = {
        "Original": source_rgb,
        "Background masked": _composite_background(source_rgb, face_mask, "blur"),
        "Hair masked": _apply_region_mask(source_rgb, _region_from_landmarks(points, source_rgb.shape, box, "hair"), mode="blur"),
        "Eyes masked": _apply_region_mask(source_rgb, _region_from_landmarks(points, source_rgb.shape, box, "eyes"), mode="blur"),
        "Mouth masked": _apply_region_mask(source_rgb, _region_from_landmarks(points, source_rgb.shape, box, "mouth"), mode="blur"),
        "Central face blurred": _apply_region_mask(source_rgb, _region_from_landmarks(points, source_rgb.shape, box, "center"), mode="blur"),
    }
    return variants


def make_background_variants(source_rgb: np.ndarray, box: tuple[int, int, int, int], landmarks: np.ndarray | None = None) -> dict[str, np.ndarray]:
    points = _landmark_array(landmarks)
    face_mask = _face_mask(source_rgb.shape, box, points)
    return {
        "Original background": source_rgb,
        "Plain background": _composite_background(source_rgb, face_mask, "plain"),
        "Blurred background": _composite_background(source_rgb, face_mask, "blur"),
        "Dark background": _composite_background(source_rgb, face_mask, "dark"),
        "Light background": _composite_background(source_rgb, face_mask, "light"),
    }


def make_quality_variants(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    darker = cv2.convertScaleAbs(crop_rgb, alpha=1.0, beta=-12)
    brighter = cv2.convertScaleAbs(crop_rgb, alpha=1.0, beta=12)
    contrast = cv2.convertScaleAbs(crop_rgb, alpha=1.10, beta=0)
    blurred = cv2.GaussianBlur(crop_rgb, (0, 0), sigmaX=1.1)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), encode_params)
    compressed = crop_rgb if not success else cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return {
        "Normal quality": crop_rgb,
        "Slight blur": blurred,
        "Brightness +": brighter,
        "Brightness -": darker,
        "Contrast +": contrast,
        "Compression": compressed,
    }


def _crop_for_prediction(detector: FaceDetector, image_rgb: np.ndarray, box: tuple[int, int, int, int]):
    return detector.crop(image_rgb, box)


def score_variant_table(
    models: list[torch.nn.Module],
    detector: FaceDetector,
    cfg: dict,
    device: torch.device,
    source_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    variants: dict[str, np.ndarray],
    use_detector_crop: bool = True,
) -> pd.DataFrame:
    rows = []
    original_score: float | None = None
    for name, variant_image in variants.items():
        if use_detector_crop:
            crop = _crop_for_prediction(detector, variant_image, box)
        else:
            crop = variant_image
        summary = score_from_models(models, crop, cfg, device)
        if name == "Original":
            original_score = summary.score
        rows.append({"Variant": name, "Predicted Score": round(summary.score, 2)})

    frame = pd.DataFrame(rows)
    if original_score is None:
        original_score = score_from_models(models, _crop_for_prediction(detector, source_rgb, box), cfg, device).score
    frame["Difference from Original"] = frame["Predicted Score"] - original_score
    frame["Difference from Original"] = frame["Difference from Original"].map(lambda value: f"{value:+.2f}")
    return frame


def predict_background_table(
    models: list[torch.nn.Module],
    detector: FaceDetector,
    cfg: dict,
    device: torch.device,
    source_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    landmarks: np.ndarray | None = None,
) -> pd.DataFrame:
    variants = make_background_variants(source_rgb, box, landmarks=landmarks)
    return score_variant_table(models, detector, cfg, device, source_rgb, box, variants, use_detector_crop=True)


def predict_ablation_table(
    models: list[torch.nn.Module],
    detector: FaceDetector,
    cfg: dict,
    device: torch.device,
    source_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    landmarks: np.ndarray | None = None,
) -> pd.DataFrame:
    variants = make_ablation_variants(source_rgb, box, landmarks=landmarks)
    return score_variant_table(models, detector, cfg, device, source_rgb, box, variants, use_detector_crop=True)


def predict_quality_table(
    models: list[torch.nn.Module],
    cfg: dict,
    device: torch.device,
    crop_rgb: np.ndarray,
) -> pd.DataFrame:
    variants = make_quality_variants(crop_rgb)
    rows = []
    original_summary = score_from_models(models, variants["Normal quality"], cfg, device)
    for name, variant in variants.items():
        summary = score_from_models(models, variant, cfg, device)
        rows.append({
            "Variant": name,
            "Predicted Score": round(summary.score, 2),
            "Difference from Original": f"{summary.score - original_summary.score:+.2f}",
        })
    return pd.DataFrame(rows)


def occlusion_sensitivity(
    models: list[torch.nn.Module],
    cfg: dict,
    device: torch.device,
    crop_rgb: np.ndarray,
    grid_rows: int = 6,
    grid_cols: int = 6,
) -> tuple[np.ndarray, np.ndarray, float]:
    base = score_from_models(models, crop_rgb, cfg, device).score
    height, width = crop_rgb.shape[:2]
    cell_h = int(np.ceil(height / grid_rows))
    cell_w = int(np.ceil(width / grid_cols))
    sensitivity = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1 = row * cell_h
            y2 = min(height, (row + 1) * cell_h)
            x1 = col * cell_w
            x2 = min(width, (col + 1) * cell_w)
            variant = crop_rgb.copy()
            patch = variant[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            variant[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (0, 0), sigmaX=18)
            score = score_from_models(models, variant, cfg, device).score
            sensitivity[row, col] = abs(score - base)

    normalized = sensitivity / (sensitivity.max() + 1e-8)
    heatmap = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_CUBIC)
    return sensitivity, heatmap, base
