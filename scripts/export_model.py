import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from common import build_model, discover_checkpoints, load_checkpoint, load_config, resolve


def export_torchscript(checkpoint_path: Path, output_path: Path, device: torch.device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model.eval()

    example_input = torch.randn(1, 3, 224, 224, device=device)

    print("Tracing model with TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))
    print(f"[OK] Saved TorchScript model to {output_path}")

    # Verify parity
    with torch.no_grad():
        orig_out = model(example_input)
        traced_out = traced_model(example_input)
        diff = torch.max(torch.abs(orig_out - traced_out)).item()
        print(f"[OK] TorchScript verification parity delta: {diff:.8f}")
        assert diff < 1e-4, f"Parity check failed: delta {diff}"

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model checkpoint to TorchScript.")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint")
    parser.add_argument("--output", default=None, help="Output .pt file path")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cpu")  # Export on CPU for maximum cross-platform compatibility

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpts = discover_checkpoints(resolve(cfg, "models"))
        if not ckpts:
            raise FileNotFoundError("No checkpoints found in models directory.")
        ckpt_path = ckpts[0]

    out_path = Path(args.output) if args.output else ckpt_path.with_name(ckpt_path.stem + "_traced.pt")
    export_torchscript(ckpt_path, out_path, device)
    print("Export completed successfully!")


if __name__ == "__main__":
    main()
