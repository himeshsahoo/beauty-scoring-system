import unittest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import build_model, count_parameters, set_backbone_trainable


class TestFaceAestheticRegressor(unittest.TestCase):
    def test_build_model_baseline(self):
        model = build_model("baseline", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([2]))

    def test_build_model_efficientnet(self):
        model = build_model("efficientnet", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([2]))
        self.assertTrue(hasattr(model, "target_layer"))

    def test_build_model_resnet18(self):
        model = build_model("resnet18", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([2]))

    def test_backbone_freezing(self):
        model = build_model("efficientnet", pretrained=False)
        total_params = count_parameters(model)
        set_backbone_trainable(model, False)
        frozen_params = count_parameters(model)
        self.assertLess(frozen_params, total_params)

        set_backbone_trainable(model, True)
        unfrozen_params = count_parameters(model)
        self.assertEqual(unfrozen_params, total_params)

    def test_model_backward_pass(self):
        model = build_model("efficientnet", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(param.grad is not None for param in model.parameters() if param.requires_grad)
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
