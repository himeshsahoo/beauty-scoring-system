import unittest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common import build_model
from export_model import export_torchscript


class TestModelExport(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = ROOT / "tests" / "fixtures"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_model("efficientnet", pretrained=False)
        self.ckpt_path = self.tmp_dir / "test_ckpt.pth"
        self.traced_path = self.tmp_dir / "test_traced.pt"

        torch.save(
            {
                "model_name": "efficientnet",
                "model_state": self.model.state_dict(),
            },
            self.ckpt_path,
        )

    def test_export_and_load_torchscript(self):
        device = torch.device("cpu")
        out_path = export_torchscript(self.ckpt_path, self.traced_path, device)
        self.assertTrue(out_path.exists())

        loaded_traced = torch.jit.load(str(out_path))
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_eager = self.model(x)
            out_traced = loaded_traced(x)
        delta = torch.max(torch.abs(out_eager - out_traced)).item()
        self.assertLess(delta, 1e-4)


if __name__ == "__main__":
    unittest.main()
