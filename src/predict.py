from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from analysis import analyze_feature_importance, draw_aesthetic_landmarks, load_model_bundle, predict_face
from common import FaceDetector, device_info, discover_checkpoints, load_config, read_landmarks, resolve, score_agreement_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--config', default=None)
    parser.add_argument('--face-index', type=int, default=None)
    parser.add_argument('--method', default='hierarchical', choices=['hierarchical', 'plus_plus', 'gradcam'])
    parser.add_argument('--colormap', default='turbo', choices=['turbo', 'inferno', 'jet', 'viridis', 'magma'])
    parser.add_argument('--alpha', type=float, default=0.38)
    parser.add_argument('--landmarks', default=None, help='Optional path to landmark PTS/TXT file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_info()
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f'Invalid/corrupted image: {args.image}')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = FaceDetector(cfg)
    boxes = detector.detect(image)
    if not boxes:
        print('Face detected: No')
        return
    if len(boxes) > 1 and args.face_index is None:
        print(f'Multiple faces detected: {len(boxes)}. Re-run with --face-index 0..{len(boxes)-1} (boxes are detector order).')
        return

    index = args.face_index or 0
    if not 0 <= index < len(boxes):
        raise ValueError('face-index is out of range')

    landmarks = read_landmarks(args.landmarks) if args.landmarks else None
    crop = detector.crop(image, boxes[index], landmarks=landmarks)
    if args.checkpoint:
        checkpoint_paths = [Path(args.checkpoint)]
    else:
        checkpoint_paths = discover_checkpoints(resolve(cfg, 'models'))
    if not checkpoint_paths:
        raise FileNotFoundError('No checkpoint. Train a model or pass --checkpoint.')

    models, ensemble_enabled, loaded_paths = load_model_bundle(checkpoint_paths, device, minimum_ensemble=5)
    summary, overlay, heatmap = predict_face(
        crop, models, cfg, device, method=args.method, colormap=args.colormap, alpha=args.alpha
    )

    feature_importance = analyze_feature_importance(
        heatmap, landmarks, crop.shape, fallback_box=(0, 0, crop.shape[1], crop.shape[0])
    )

    out = resolve(cfg, 'results/inference')
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / 'face_crop.png'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / 'saliency_heatmap.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if landmarks is not None:
        landmarks_overlay = draw_aesthetic_landmarks(crop, landmarks)
        cv2.imwrite(str(out / 'landmarks.png'), cv2.cvtColor(landmarks_overlay, cv2.COLOR_RGB2BGR))

    result = {
        'face_detected': True,
        'selected_face': index,
        'detected_faces': len(boxes),
        'checkpoint_count': len(loaded_paths),
        'ensemble_trained': ensemble_enabled,
        'predicted_aesthetic_score_1_10': round(summary.score, 2),
        'estimated_uncertainty': None if summary.uncertainty is None else round(summary.uncertainty, 2),
        'model_agreement': summary.agreement,
        'saliency_method': args.method,
        'feature_importance_breakdown': feature_importance,
        'uncertainty_note': 'Spread across trained checkpoints; not a calibrated probability.' if ensemble_enabled else 'Ensemble not trained — uncertainty unavailable.',
    }
    (out / 'prediction.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Face detected: Yes')
    print(f'Predicted aesthetic score: {summary.score:.2f} / 10')
    if summary.uncertainty is None:
        print('Ensemble not trained - uncertainty unavailable')
    else:
        print(f'Ensemble uncertainty: ±{summary.uncertainty:.2f}')
        print(f'Model agreement: {summary.agreement}')
    print('Regional importance breakdown:')
    for zone, fraction in feature_importance.items():
        print(f'  - {zone.capitalize()}: {fraction * 100:.1f}%')


if __name__ == '__main__':
    main()
