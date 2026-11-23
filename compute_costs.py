"""CPU-only parameter/MAC accounting for G4-v1 adapters."""
from __future__ import annotations

import torch
from torch import nn


def conv2d_macs_per_image(module: nn.Module, height: int, width: int) -> int:
    """Count multiply-accumulates for registered Conv2d layers at H×W."""
    total = 0
    for layer in module.modules():
        if not isinstance(layer, nn.Conv2d):
            continue
        kh, kw = layer.kernel_size
        sh, sw = layer.stride
        ph, pw = layer.padding
        dh, dw = layer.dilation
        out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        total += (out_h * out_w * layer.out_channels *
                  (layer.in_channels // layer.groups) * kh * kw)
    return int(total)


def dtcwt_filter_conv_macs_per_image(adapter: nn.Module,
                                     height: int = 130,
                                     width: int = 130) -> dict:
    """Count actual fixed DTCWT forward conv MACs using CPU execution hooks.

    Counts grouped `F.conv2d` in the original vendored transform. It does not
    count abs/sqrt, tensor rearrangement, padding/indexing, interpolation, or
    backward work; those remain separately identified non-MAC overhead.
    """
    if not hasattr(adapter, "transform"):
        return {"filter_conv_macs_per_image": 0, "conv_calls": 0,
                "detail": "adapter has no DTCWT transform"}
    import torch.nn.functional as functional

    original = functional.conv2d
    calls: list[dict] = []

    def counted(input_tensor, weight, bias=None, stride=1, padding=0,
                dilation=1, groups=1):
        output = original(input_tensor, weight, bias, stride, padding,
                          dilation, groups)
        kh, kw = weight.shape[-2:]
        macs = output.numel() * weight.shape[1] * kh * kw
        calls.append({"input_shape": list(input_tensor.shape),
                      "output_shape": list(output.shape),
                      "kernel": [kh, kw], "groups": int(groups),
                      "macs": int(macs)})
        return output

    functional.conv2d = counted
    try:
        device = next(adapter.transform.buffers()).device
        sample = torch.zeros(1, adapter.bottleneck, height, width,
                             device=device, dtype=torch.float32)
        with torch.inference_mode():
            adapter.transform(sample)
    finally:
        functional.conv2d = original
    return {"filter_conv_macs_per_image": sum(x["macs"] for x in calls),
            "conv_calls": len(calls), "calls": calls,
            "input_shape": [1, adapter.bottleneck, height, width],
            "counting_boundary": "DTCWT fixed forward grouped conv2d only"}


def parameter_count(module: nn.Module) -> dict[str, int]:
    return {"total": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters()
                             if p.requires_grad),
            "buffers": sum(b.numel() for b in module.buffers())}
