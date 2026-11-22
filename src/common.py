from __future__ import annotations

import random
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or project_root() / "config.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve(config: dict, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def set_seed(seed: int, benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device_info() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        print(f"GPU: {gpu_name} ({total_vram:.2f} GB VRAM, CUDA {cuda_version})")
        # Enable TF32 for Ampere (RTX 30-series) GPUs for faster matrix operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def score_to_unit(score_1_10: float) -> float:
    return (float(score_1_10) - 1.0) / 9.0


def unit_to_score(score_unit: np.ndarray | torch.Tensor | float) -> np.ndarray | torch.Tensor | float:
    return 1.0 + 9.0 * score_unit


def score_agreement_label(std: float | None) -> str:
    if std is None:
        return "Unavailable"
    if std < 0.35:
        return "High"
    if std < 0.75:
        return "Moderate"
    return "Low"


def discover_checkpoints(models_dir: Path) -> list[Path]:
    seeded = sorted(models_dir.glob("*_seed*_best.pth"))
    if seeded:
        return seeded
    return sorted(models_dir.glob("*_best.pth"))


def read_landmarks(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    landmark_path = Path(path)
    if not landmark_path.exists():
        return None

    raw = landmark_path.read_bytes()
    if len(raw) >= 8:
        count = struct.unpack_from("<i", raw, 0)[0]
        coords = np.frombuffer(raw[4:], dtype="<f4")
        if count > 0 and coords.size >= count * 2:
            return coords[: count * 2].reshape(count, 2).astype(np.float32)

    try:
        lines = [line.strip() for line in landmark_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    except UnicodeDecodeError:
        return None

    values: list[float] = []
    for line in lines:
        if not line or line.startswith("version") or line.startswith("n_points") or line in {"{", "}"}:
            continue
        cleaned = line.replace("(", " ").replace(")", " ").replace(",", " ").split()
        if len(cleaned) >= 2:
            try:
                values.extend([float(cleaned[0]), float(cleaned[1])])
            except ValueError:
                continue

    if len(values) >= 4 and len(values) % 2 == 0:
        return np.asarray(values, dtype=np.float32).reshape(-1, 2)
    return None


def select_largest(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return max(boxes, key=lambda box: box[2] * box[3])


def _rotate_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogenous = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    rotated = homogenous @ matrix.T
    return rotated.astype(np.float32)


def _eye_centers(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if points.shape[0] >= 48:
        left_eye = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)
        return left_eye, right_eye
    return None


def _landmark_box(points: np.ndarray, image_shape: tuple[int, int, int], margin: float) -> tuple[int, int, int, int]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    side = max(width, height) * (1.0 + 2.0 * margin)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    half = side / 2.0
    x1 = max(0, int(round(center_x - half)))
    y1 = max(0, int(round(center_y - half)))
    x2 = min(image_shape[1], int(round(center_x + half)))
    y2 = min(image_shape[0], int(round(center_y + half)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def landmark_box(points: np.ndarray, image_shape: tuple[int, int, int], margin: float) -> tuple[int, int, int, int]:
    return _landmark_box(points, image_shape, margin)


class FaceDetector:
    """Detect a face and crop it tightly enough to reduce background shortcuts."""

    def __init__(self, config: dict):
        cascade = cv2.data.haarcascades + config["face_detector"]["cascade"]
        self.detector = cv2.CascadeClassifier(cascade)
        if self.detector.empty():
            raise RuntimeError("OpenCV face cascade could not be loaded.")
        self.margin = float(config["face_detector"]["margin"])
        self.align = bool(config["face_detector"].get("align", True))
        self.mtcnn = None
        if config["face_detector"].get("backend", "mtcnn") == "mtcnn":
            try:
                from facenet_pytorch import MTCNN

                self.mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                print("facenet-pytorch is unavailable; using OpenCV Haar fallback.")

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.mtcnn is not None:
            boxes, _ = self.mtcnn.detect(Image.fromarray(image))
            if boxes is not None:
                height, width = image.shape[:2]
                results = []
                for x1, y1, x2, y2 in boxes:
                    if x2 <= x1 or y2 <= y1:
                        continue
                    left = max(0, int(x1))
                    top = max(0, int(y1))
                    right = min(width, int(x2))
                    bottom = min(height, int(y2))
                    if right > left and bottom > top:
                        results.append((left, top, right - left, bottom - top))
                if results:
                    return results
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        boxes = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return [tuple(map(int, box)) for box in boxes]

    def _align(self, image: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eye_centers = _eye_centers(landmarks)
        if eye_centers is None:
            return image, landmarks
        left_eye, right_eye = eye_centers
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        angle = -np.degrees(np.arctan2(dy, dx))
        center = tuple(((left_eye + right_eye) / 2.0).tolist())
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        rotated_landmarks = _rotate_points(landmarks, matrix)
        return rotated, rotated_landmarks

    def crop(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
        landmarks: np.ndarray | None = None,
    ) -> np.ndarray:
        if landmarks is not None and self.align:
            image, landmarks = self._align(image, landmarks)
            box = _landmark_box(landmarks, image.shape, self.margin)
        x, y, width, height = box
        pad = int(round(max(width, height) * self.margin))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + width + pad)
        y2 = min(image.shape[0], y + height + pad)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Detected face crop was empty.")
        return crop

    def preprocess(
        self,
        image: np.ndarray,
        landmarks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        boxes = self.detect(image)
        if not boxes:
            raise ValueError("No face detected.")
        box = select_largest(boxes)
        crop = self.crop(image, box, landmarks=landmarks)
        return crop, box


def make_transform(size: int, train: bool = False):
    if train:
        steps = [
            transforms.RandomResizedCrop(size, scale=(0.88, 1.0), ratio=(0.95, 1.05), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.96, 1.04), shear=2),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
            transforms.RandomGrayscale(p=0.03),
        ]
    else:
        steps = [transforms.Resize((size, size), antialias=True)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class AestheticDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, size: int, train: bool = False, cache_in_memory: bool = False):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = make_transform(size, train)
        self.crop_paths = [str(p) for p in self.frame["crop_path"]]
        self.targets = torch.tensor([score_to_unit(float(s)) for s in self.frame["score_1_10"]], dtype=torch.float32)
        self.image_ids = [str(i) for i in self.frame["image_id"]]
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.crop_paths)

    def __getitem__(self, idx: int):
        if self.cache_in_memory and idx in self._cache:
            image = self._cache[idx]
        else:
            path = self.crop_paths[idx]
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                if self.cache_in_memory:
                    self._cache[idx] = image
            except Exception as exc:
                raise RuntimeError(f"Unreadable crop {path}: {exc}") from exc
        return self.transform(image), self.targets[idx], self.image_ids[idx]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def _replace_head(module: nn.Module, attribute: str) -> None:
    head = getattr(module, attribute)
    if isinstance(head, nn.Sequential):
        last = head[-1]
        if not isinstance(last, nn.Linear):
            raise RuntimeError(f"Unexpected classifier layout for {attribute}.")
        head[-1] = nn.Linear(last.in_features, 1)
    elif isinstance(head, nn.Linear):
        setattr(module, attribute, nn.Linear(head.in_features, 1))
    else:
        raise RuntimeError(f"Unsupported head type for {attribute}: {type(head)!r}")


class FaceAestheticRegressor(nn.Module):
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.name = name.lower()
        if self.name == "baseline":
            self.backbone = SimpleCNN()
            self.target_layer = self.backbone.features[6]
        elif self.name == "efficientnet":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None,
            )
            _replace_head(self.backbone, "classifier")
            self.target_layer = self.backbone.features[-1]
        elif self.name == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None,
            )
            _replace_head(self.backbone, "fc")
            self.target_layer = self.backbone.layer4[-1]
        elif self.name == "mobilenetv2":
            self.backbone = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None,
            )
            _replace_head(self.backbone, "classifier")
            self.target_layer = self.backbone.features[-1]
        else:
            raise ValueError("model must be baseline, efficientnet, resnet50, or mobilenetv2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        return output.squeeze(1) if output.ndim > 1 else output


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    return FaceAestheticRegressor(name, pretrained=pretrained)


def backbone_parameter_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for param_name, param in model.named_parameters():
        if any(token in param_name for token in ("classifier", ".fc", "head")):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param_name, param in model.named_parameters():
        if any(token in param_name for token in ("classifier", ".fc", "head")):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_name"], pretrained=False).to(device)
    state_dict = checkpoint["model_state"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), nn.Module):
            model.backbone.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, checkpoint
