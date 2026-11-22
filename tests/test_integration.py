import unittest
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import FaceDetector, build_model, load_config
from analysis import (
    load_model_bundle,
    predict_face,
    predict_ablation_table,
    predict_background_table,
    predict_quality_table,
    occlusion_sensitivity,
)


class TestFullPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = build_model("efficientnet", pretrained=False).to(cls.device)
        cls.model.eval()
        cls.models = [cls.model]
        # Synthetic face image with distinct facial pattern
        cls.image = np.ones((224, 224, 3), dtype=np.uint8) * 180
        # Draw eyes and mouth
        cls.image[60:80, 60:85] = 40
        cls.image[60:80, 140:165] = 40
        cls.image[150:170, 90:135] = 30
        cls.box = (20, 20, 184, 184)

    def test_predict_face_all_methods(self):
        for method in ["hierarchical", "plus_plus", "gradcam"]:
            summary, overlay, heatmap = predict_face(
                self.image, self.models, self.cfg, self.device, method=method, colormap="turbo"
            )
            self.assertGreaterEqual(summary.score, 1.0)
            self.assertLessEqual(summary.score, 10.0)
            self.assertEqual(overlay.shape, (224, 224, 3))
            self.assertEqual(heatmap.shape, (224, 224))

    def test_occlusion_sensitivity_vectorized(self):
        sensitivity, heatmap, base_score = occlusion_sensitivity(
            self.models, self.cfg, self.device, self.image, grid_rows=4, grid_cols=4
        )
        self.assertEqual(sensitivity.shape, (4, 4))
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertGreaterEqual(base_score, 1.0)
        self.assertLessEqual(base_score, 10.0)

    def test_sensitivity_tables(self):
        detector = FaceDetector(self.cfg)
        ablation = predict_ablation_table(self.models, detector, self.cfg, self.device, self.image, self.box)
        background = predict_background_table(self.models, detector, self.cfg, self.device, self.image, self.box)
        quality = predict_quality_table(self.models, self.cfg, self.device, self.image)

        self.assertGreater(len(ablation), 0)
        self.assertGreater(len(background), 0)
        self.assertGreater(len(quality), 0)
        self.assertIn("Predicted Score", ablation.columns)
        self.assertIn("Difference from Original", ablation.columns)


if __name__ == "__main__":
    unittest.main()
