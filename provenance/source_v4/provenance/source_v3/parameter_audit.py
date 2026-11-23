"""Generate model parameter, learned-adapter MAC and fixed-DTCWT cost audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from compute_costs import (conv2d_macs_per_image,
                           dtcwt_filter_conv_macs_per_image, parameter_count)
from g4_model import GROUPS, SpatialResidualAdapter, build_model
from models.dtcwt_adapter import DTCWTResidualAdapter


def main():
    torch.set_num_threads(2)
    model_rows = {}
    for group in GROUPS:
        model = build_model(group)
        count = parameter_count(model)
        adapter_params = sum(p.numel() for p in model.cif.parameters()) - sum(
            p.numel() for p in model.cif.attention.parameters()) - sum(
            p.numel() for p in model.cif.attention_1.parameters()) - sum(
            p.numel() for p in model.cif.last_layer_e.parameters()) - sum(
            p.numel() for p in model.cif.last_layer_d.parameters())
        if group == "G0":
            adapter_params = 0
        invalidated = 0
        if group == "G4-NoCross":
            invalidated = sum(p.numel() for module in
                              (model.cif.attention, model.cif.attention_1)
                              for p in module.parameters())
        model_rows[group] = {
            "whole_model_parameters": count["total"],
            "trainable_parameters": count["trainable"],
            "adapter_parameters": adapter_params,
            "registered_but_invalidated_no_cross_attention_parameters": invalidated,
            "effective_parameter_count_excluding_invalidated_attention":
                count["total"] - invalidated,
        }
        del model

    d8 = DTCWTResidualAdapter(channels=512, bottleneck=8, levels=3,
                              biort="near_sym_a", qshift="qshift_a",
                              mode="symmetric")
    s8 = SpatialResidualAdapter(8)
    s16 = SpatialResidualAdapter(16)
    learned = {
        "DTCWT-8": {"parameters": sum(p.numel() for p in d8.parameters()),
                    "conv_macs_per_image_at_130": conv2d_macs_per_image(d8, 130, 130)},
        "Spatial-8": {"parameters": sum(p.numel() for p in s8.parameters()),
                      "conv_macs_per_image_at_130": conv2d_macs_per_image(s8, 130, 130)},
        "Spatial-16": {"parameters": sum(p.numel() for p in s16.parameters()),
                       "conv_macs_per_image_at_130": conv2d_macs_per_image(s16, 130, 130)},
    }
    dtcwt_fixed = dtcwt_filter_conv_macs_per_image(d8, 130, 130)
    g4_learned_macs = (learned["DTCWT-8"]["conv_macs_per_image_at_130"] +
                       learned["Spatial-16"]["conv_macs_per_image_at_130"])
    ss_learned_macs = (learned["Spatial-8"]["conv_macs_per_image_at_130"] +
                       learned["Spatial-16"]["conv_macs_per_image_at_130"])
    report = {
        "accounting_input": {"feature_size": [130, 130], "batch_for_per_image": 1,
                             "DTCWT_bottleneck": 8, "levels": 3,
                             "orientations_per_level": 6,
                             "filters": ["near_sym_a", "qshift_a"],
                             "boundary": "symmetric"},
        "models": model_rows,
        "adapter_components": learned,
        "G4_learned_conv_macs_per_image": g4_learned_macs,
        "G4_SS_learned_conv_macs_per_image": ss_learned_macs,
        "G4_learned_conv_flops_lower_bound_per_image": 2 * g4_learned_macs,
        "G4_batch4_learned_conv_flops_lower_bound": 8 * g4_learned_macs,
        "DTCWT_fixed_transform_forward": dtcwt_fixed,
        "DTCWT_fixed_transform_batch4_filter_conv_macs":
            4 * dtcwt_fixed["filter_conv_macs_per_image"],
        "limitations": [
            "DTCWT fixed filter count is measured from actual CPU F.conv2d outputs for the vendored transform.",
            "Fixed-filter count excludes magnitude sqrt/add, interpolation, padding/indexing, and backward cost.",
            "2 FLOPs/MAC is used only for learned convolution MAC lower bounds; this is not whole-model FLOPs.",
            "No GPU or end-to-end latency measurement was performed in this phase."
        ]
    }
    out = ROOT / "reports" / "MODEL_PARAMETER_MAC_AUDIT.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
