import unittest
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import build_model
from analysis import (
    gradcam_heatmap,
    gradcam_plus_plus_heatmap,
    hierarchical_saliency_heatmap,
    normalize_heatmap,
    overlay_heatmap,
    analyze_feature_importance,
)


class TestHeatmapGeneration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = build_model("efficientnet", pretrained=False).to(self.device)
        self.cfg = {"image_size": 224}
        self.sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_gradcam_heatmap_shape_and_bounds(self):
        cam = gradcam_heatmap(self.model, self.sample_image, self.cfg, self.device)
        self.assertEqual(cam.shape, (224, 224))
        self.assertGreaterEqual(cam.min(), 0.0)
        self.assertLessEqual(cam.max(), 1.0 + 1e-6)

    def test_gradcam_plus_plus_heatmap(self):
        cam_pp = gradcam_plus_plus_heatmap(self.model, self.sample_image, self.cfg, self.device)
        self.assertEqual(cam_pp.shape, (224, 224))
        self.assertGreaterEqual(cam_pp.min(), 0.0)
        self.assertLessEqual(cam_pp.max(), 1.0 + 1e-6)

    def test_hierarchical_saliency(self):
        h_cam = hierarchical_saliency_heatmap(self.model, self.sample_image, self.cfg, self.device)
        self.assertEqual(h_cam.shape, (224, 224))

    def test_normalize_heatmap(self):
        raw = np.array([[0.0, 5.0], [10.0, 100.0]], dtype=np.float32)
        norm = normalize_heatmap(raw)
        self.assertEqual(norm.shape, raw.shape)
        self.assertGreaterEqual(norm.min(), 0.0)
        self.assertLessEqual(norm.max(), 1.0)

    def test_overlay_heatmap(self):
        cam = np.random.rand(224, 224).astype(np.float32)
        overlay = overlay_heatmap(self.sample_image, cam, alpha=0.4, colormap="turbo")
        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_analyze_feature_importance(self):
        cam = np.ones((224, 224), dtype=np.float32)
        box = (20, 20, 180, 180)
        importance = analyze_feature_importance(cam, landmarks=None, image_shape=(224, 224, 3), fallback_box=box)
        self.assertIn("eyes", importance)
        self.assertIn("mouth", importance)
        self.assertGreater(importance["eyes"], 0.0)


if __name__ == "__main__":
    unittest.main()
