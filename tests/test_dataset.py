import unittest
import pandas as pd
import torch
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import AestheticDataset, make_transform, score_to_unit, unit_to_score
from train import split_frame


class TestDatasetAndTransforms(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = ROOT / "tests" / "fixtures"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.img_path = self.tmp_dir / "sample_crop.jpg"
        Image.new("RGB", (100, 100), color=(120, 150, 180)).save(self.img_path)

        self.df = pd.DataFrame(
            {
                "crop_path": [str(self.img_path)] * 10,
                "score_1_10": [2.0, 3.5, 5.0, 6.5, 8.0, 4.0, 7.0, 5.5, 6.0, 7.5],
                "image_id": [f"img_{i}.jpg" for i in range(10)],
                "identity": [f"id_{i % 3}" for i in range(10)],
            }
        )

    def test_unit_conversions(self):
        self.assertAlmostEqual(score_to_unit(1.0), 0.0)
        self.assertAlmostEqual(score_to_unit(10.0), 1.0)
        self.assertAlmostEqual(score_to_unit(5.5), 0.5)
        self.assertAlmostEqual(float(unit_to_score(0.0)), 1.0)
        self.assertAlmostEqual(float(unit_to_score(1.0)), 10.0)
        self.assertAlmostEqual(float(unit_to_score(0.5)), 5.5)

    def test_aesthetic_dataset_train_and_eval(self):
        train_ds = AestheticDataset(self.df, size=224, train=True)
        eval_ds = AestheticDataset(self.df, size=224, train=False)
        self.assertEqual(len(train_ds), 10)
        self.assertEqual(len(eval_ds), 10)

        x_train, y_train, id_train = train_ds[0]
        self.assertEqual(x_train.shape, (3, 224, 224))
        self.assertTrue(0.0 <= float(y_train) <= 1.0)

        x_eval, y_eval, id_eval = eval_ds[0]
        self.assertEqual(x_eval.shape, (3, 224, 224))

    def test_dataloader_batching(self):
        ds = AestheticDataset(self.df, size=224, train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        x, y, ids = batch
        self.assertEqual(x.shape, (4, 3, 224, 224))
        self.assertEqual(y.shape, (4,))

    def test_split_frame_identity_disjoint(self):
        cfg = {"val_fraction": 0.2, "test_fraction": 0.2, "seed": 42}
        train, val, test = split_frame(self.df, cfg)
        train_ids = set(train["identity"])
        val_ids = set(val["identity"])
        test_ids = set(test["identity"])
        self.assertTrue(train_ids.isdisjoint(test_ids))


if __name__ == "__main__":
    unittest.main()
