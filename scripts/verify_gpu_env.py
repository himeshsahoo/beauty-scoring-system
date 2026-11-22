import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from common import build_model, device_info, load_config, resolve
from analysis import gradcam_heatmap, gradcam_plus_plus_heatmap


def verify_environment():
    print("=" * 60)
    print("AI Facial Aesthetic Scoring - GPU Environment Verification")
    print("=" * 60)

    cfg = load_config()
    device = device_info()

    if device.type != "cuda":
        print("[WARNING] CUDA is not available. Running on CPU.")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"[OK] GPU: {props.name}")
        print(f"[OK] Total VRAM: {props.total_memory / (1024**3):.2f} GB")
        print(f"[OK] CUDA capability: {props.major}.{props.minor}")
        print(f"[OK] TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")

    print("\nVerifying model architectures with pretrained weights...")
    for model_name in ["efficientnet", "resnet18"]:
        try:
            model = build_model(model_name, pretrained=True).to(device)
            dummy_input = torch.randn(2, 3, 224, 224, device=device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(dummy_input)
            print(f"[OK] {model_name} initialized and inferred: shape {out.shape}")
        except Exception as exc:
            print(f"[ERROR] {model_name} failed: {exc}")
            return False

    print("\nVerifying Grad-CAM and Grad-CAM++ on GPU...")
    try:
        sample_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cam = gradcam_heatmap(model, sample_img, cfg, device)
        cam_pp = gradcam_plus_plus_heatmap(model, sample_img, cfg, device)
        print(f"[OK] Grad-CAM generated: shape {cam.shape}, range [{cam.min():.2f}, {cam.max():.2f}]")
        print(f"[OK] Grad-CAM++ generated: shape {cam_pp.shape}, range [{cam_pp.min():.2f}, {cam_pp.max():.2f}]")
    except Exception as exc:
        print(f"[ERROR] Heatmap failed: {exc}")
        return False

    manifest_path = resolve(cfg, cfg["data"]["processed_manifest"])
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        print(f"\n[OK] Manifest exists: {len(df)} samples loaded from {manifest_path}")
    else:
        print(f"\n[WARNING] Manifest not found at {manifest_path}")

    print("\nEnvironment verification successful!")
    return True


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
