import pytest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import build_model, count_parameters, set_backbone_trainable


def test_build_model_baseline():
    model = build_model("baseline", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == torch.Size([2])


def test_build_model_efficientnet():
    model = build_model("efficientnet", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == torch.Size([2])
    assert hasattr(model, "target_layer")


def test_build_model_resnet18():
    model = build_model("resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == torch.Size([2])


def test_backbone_freezing():
    model = build_model("efficientnet", pretrained=False)
    total_params = count_parameters(model)
    set_backbone_trainable(model, False)
    frozen_params = count_parameters(model)
    assert frozen_params < total_params

    set_backbone_trainable(model, True)
    unfrozen_params = count_parameters(model)
    assert unfrozen_params == total_params


def test_model_backward_pass():
    model = build_model("efficientnet", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            break


if __name__ == "__main__":
    test_build_model_baseline()
    test_build_model_efficientnet()
    test_build_model_resnet18()
    test_backbone_freezing()
    test_model_backward_pass()
    print("All model tests passed!")
