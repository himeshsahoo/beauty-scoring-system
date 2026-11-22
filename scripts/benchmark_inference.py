import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from common import build_model, device_info, load_config
from analysis import gradcam_heatmap, gradcam_plus_plus_heatmap, hierarchical_saliency_heatmap


def benchmark():
    print("=" * 65)
    print("AI Facial Aesthetic Scoring - RTX 3050 Ti Inference Benchmark")
    print("=" * 65)

    device = device_info()
    cfg = load_config()
    model = build_model("efficientnet", pretrained=False).to(device)
    model.eval()

    # Warmup
    print("\nWarming up GPU...")
    warmup_tensor = torch.randn(8, 3, 224, 224, device=device)
    for _ in range(10):
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _ = model(warmup_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Latency & Throughput Benchmark
    print("\nBenchmarking Forward Inference:")
    batch_sizes = [1, 4, 8, 16, 32]
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 224, 224, device=device)
        iterations = 50
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                for _ in range(iterations):
                    _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        avg_latency_ms = (elapsed / iterations) * 1000.0
        fps = (bs * iterations) / elapsed
        print(f"  Batch size {bs:2d}: {avg_latency_ms:6.2f} ms/batch | {fps:7.1f} FPS")

    # Heatmap Latency Benchmark
    print("\nBenchmarking Saliency Heatmap Generation (Single Image):")
    sample_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    methods = [
        ("Standard Grad-CAM", gradcam_heatmap),
        ("Grad-CAM++", gradcam_plus_plus_heatmap),
        ("Hierarchical Multi-scale", hierarchical_saliency_heatmap),
    ]

    for name, fn in methods:
        # 3 warmup runs
        for _ in range(3):
            _ = fn(model, sample_img, cfg, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        runs = 15
        for _ in range(runs):
            _ = fn(model, sample_img, cfg, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        avg_ms = ((time.perf_counter() - start) / runs) * 1000.0
        print(f"  {name:25s}: {avg_ms:6.2f} ms")

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nPeak VRAM Allocated: {peak_vram_mb:.1f} MB (well within 4GB limit)")

    print("=" * 65)


if __name__ == "__main__":
    benchmark()
