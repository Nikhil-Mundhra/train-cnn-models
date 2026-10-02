#!/usr/bin/env python3
"""Measure memory and step time for the proposed anisotropic 3D backbone.

This is a capacity probe, not the production model. It intentionally mirrors
the planned two axial-only downsampling stages followed by two isotropic stages
and a symmetric decoder with skip connections.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int]):
        super().__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv3d(in_channels, out_channels, 1)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(torch.cat((self.reduce(x), skip), dim=1))


class AnisotropicUNetCapacityProbe(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 12]
        self.enc0 = ConvBlock(1, channels[0])
        self.enc1 = DownBlock(channels[0], channels[1], (1, 2, 1))
        self.enc2 = DownBlock(channels[1], channels[2], (1, 2, 1))
        self.enc3 = DownBlock(channels[2], channels[3], (2, 2, 2))
        self.bottleneck = DownBlock(channels[3], channels[4], (2, 2, 2))
        self.dec3 = UpBlock(channels[4], channels[3], channels[3])
        self.dec2 = UpBlock(channels[3], channels[2], channels[2])
        self.dec1 = UpBlock(channels[2], channels[1], channels[1])
        self.dec0 = UpBlock(channels[1], channels[0], channels[0])
        self.head = nn.Conv3d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.bottleneck(x3)
        return self.head(self.dec0(self.dec1(self.dec2(self.dec3(x4, x3), x2), x1), x0))


@dataclass
class ProfileResult:
    patch_shape: Tuple[int, int, int]
    batch_size: int
    precision: str
    status: str
    parameters: int
    peak_allocated_gib: float = 0.0
    peak_reserved_gib: float = 0.0
    mean_step_seconds: float = 0.0
    error: str = ""


def parse_shapes(raw: str) -> List[Tuple[int, int, int]]:
    shapes: List[Tuple[int, int, int]] = []
    for item in raw.split(","):
        shape = tuple(int(value) for value in item.lower().split("x"))
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise argparse.ArgumentTypeError(f"Invalid patch shape: {item}")
        if any(value % 4 for value in shape):
            raise argparse.ArgumentTypeError(f"Every patch dimension must be divisible by 4: {item}")
        shapes.append(shape)
    return shapes


def profile_one(
    shape: Tuple[int, int, int],
    batch_size: int,
    precision: str,
    base_channels: int,
    warmup: int,
    steps: int,
) -> ProfileResult:
    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    model = AnisotropicUNetCapacityProbe(base_channels=base_channels).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = ProfileResult(shape, batch_size, precision, "ok", parameter_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_amp = precision != "fp32"

    try:
        image = torch.randn((batch_size, 1, *shape), device=device)
        target = torch.empty((batch_size, 1, *shape), device=device).bernoulli_(0.1)
        torch.cuda.reset_peak_memory_stats()
        elapsed: List[float] = []
        for step in range(warmup + steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                logits = model(image)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step >= warmup:
                elapsed.append(time.perf_counter() - started)
        result.peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        result.peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
        result.mean_step_seconds = sum(elapsed) / len(elapsed)
    except torch.cuda.OutOfMemoryError as exc:
        result.status = "oom"
        result.error = str(exc).splitlines()[0]
    finally:
        del model, optimizer
        if "image" in locals():
            del image, target
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=parse_shapes, default=parse_shapes("32x768x32,48x768x48,64x768x64"))
    parser.add_argument("--batch_sizes", default="1,2")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the A100 feasibility profile")

    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    results = [
        profile_one(shape, batch_size, args.precision, args.base_channels, args.warmup, args.steps)
        for shape in args.patches
        for batch_size in batch_sizes
    ]
    payload = {
        "device": torch.cuda.get_device_name(),
        "total_device_memory_gib": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
        "torch_version": torch.__version__,
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

