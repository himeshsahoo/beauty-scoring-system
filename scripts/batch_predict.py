import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import pandas as pd
from analysis import analyze_feature_importance, load_model_bundle, predict_face
from common import FaceDetector, device_info, discover_checkpoints, load_config, resolve


def batch_predict(
    input_dir: Path,
    output_csv: Path,
    checkpoint_dir: Path,
    cfg: dict,
    save_heatmaps: bool = False,
    heatmap_dir: Path | None = None,
):
    device = device_info()
    detector = FaceDetector(cfg)

    ckpt_paths = discover_checkpoints(checkpoint_dir)
    if not ckpt_paths:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")
    models, ensemble_enabled, _ = load_model_bundle(ckpt_paths, device, minimum_ensemble=5)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts]
    print(f"Found {len(image_paths)} images in {input_dir}")

    if save_heatmaps and heatmap_dir is not None:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, img_path in enumerate(image_paths, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = detector.detect(image_rgb)
        if not boxes:
            records.append({
                "image_name": img_path.name,
                "face_detected": False,
                "predicted_score": None,
                "top_influential_zone": None,
            })
            continue

        crop = detector.crop(image_rgb, boxes[0])
        summary, overlay, heatmap = predict_face(crop, models, cfg, device, method="hierarchical")
        importance = analyze_feature_importance(heatmap, None, crop.shape, fallback_box=(0, 0, crop.shape[1], crop.shape[0]))
        top_zone = max(importance.items(), key=lambda item: item[1])[0] if importance else "center"

        if save_heatmaps and heatmap_dir is not None:
            out_img_path = heatmap_dir / f"{img_path.stem}_saliency.png"
            cv2.imwrite(str(out_img_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        records.append({
            "image_name": img_path.name,
            "face_detected": True,
            "predicted_score": round(summary.score, 2),
            "uncertainty": round(summary.uncertainty, 2) if summary.uncertainty is not None else None,
            "top_influential_zone": top_zone,
            "eyes_saliency_pct": round(importance.get("eyes", 0.0) * 100, 1),
            "mouth_saliency_pct": round(importance.get("mouth", 0.0) * 100, 1),
        })

        if idx % 25 == 0 or idx == len(image_paths):
            print(f"Processed [{idx:3d}/{len(image_paths):3d}] images...")

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Batch scoring complete! Results saved to {output_csv}")
    detected_count = df["face_detected"].sum()
    if detected_count > 0:
        mean_score = df[df["face_detected"]]["predicted_score"].mean()
        print(f"Summary: {detected_count}/{len(df)} faces detected | Mean aesthetic score: {mean_score:.2f} / 10")
    return df


def main():
    parser = argparse.ArgumentParser(description="Batch score facial aesthetics across an image folder.")
    parser.add_argument("--input-dir", required=True, help="Folder containing portrait images")
    parser.add_argument("--output-csv", default="results/batch_predictions.csv", help="Output CSV path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint or models directory")
    parser.add_argument("--save-heatmaps", action="store_true", help="Save saliency heatmaps")
    args = parser.parse_args()

    cfg = load_config()
    in_dir = Path(args.input_dir)
    out_csv = resolve(cfg, args.output_csv)
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else resolve(cfg, "models")
    hm_dir = resolve(cfg, "results/batch_heatmaps") if args.save_heatmaps else None

    batch_predict(in_dir, out_csv, ckpt_dir, cfg, save_heatmaps=args.save_heatmaps, heatmap_dir=hm_dir)


if __name__ == "__main__":
    main()
