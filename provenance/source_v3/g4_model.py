"""Independent G4-v1 model factory and task-specific CIF adapter routing.

The baseline HRNet, branch projections, bidirectional CIF attention and both
classifier heads are retained. Adapters act on the post-CIF fused 512-channel
feature, immediately before its task's existing classifier head.
"""
from __future__ import annotations

import sys
import contextlib
import io
from pathlib import Path
from typing import Any

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent
_VENDOR = _ROOT / "vendor"
_WAVELET_VENDOR_PARENT = _VENDOR / "pytorch_wavelets"
for _import_root in reversed((str(_WAVELET_VENDOR_PARENT), str(_VENDOR))):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

# The vendored pytorch_wavelets package is byte-for-byte copied from the
# previously audited implementation; its data package contains the two filters.
with contextlib.redirect_stdout(io.StringIO()):
    from baseline_code.nets.AECIF_Net import AECIF_Net
from baseline_code.nets.backbone import CIF
from baseline_code.nets.hrnet_training import weights_init
from models.dtcwt_adapter import DTCWTResidualAdapter


GROUPS = ("G0", "G-E", "G-D", "G4", "G4-SS", "G4-Swap", "G4-NoCross")


class SpatialResidualAdapter(nn.Module):
    """The G2 spatial residual adapter, parameterized for Spatial-8/Spatial-16."""

    def __init__(self, bottleneck: int):
        super().__init__()
        if bottleneck not in (8, 16):
            raise ValueError("Spatial adapter bottleneck must be 8 or 16")
        self.bottleneck = int(bottleneck)
        self.reduce = nn.Conv2d(512, self.bottleneck, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(self.bottleneck, self.bottleneck, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.bottleneck, self.bottleneck, 3, padding=1),
            nn.GELU(),
        )
        self.expand = nn.Conv2d(self.bottleneck, 512, 1)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.expand(self.fuse(self.reduce(x)))
        return x + self.alpha.to(dtype=x.dtype) * residual


_ADAPTER_SEEDS = {("dtcwt", 8): 31008, ("spatial", 8): 32008,
                  ("spatial", 16): 32016}


def _construct_adapter(kind: str, bottleneck: int) -> nn.Module:
    if kind == "identity":
        return nn.Identity()
    seed_key = (kind, int(bottleneck))
    if seed_key not in _ADAPTER_SEEDS:
        raise ValueError(f"Unsupported adapter definition: {seed_key}")
    # Keep initialization independent of route/order and do not consume the
    # caller's global CPU RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_ADAPTER_SEEDS[seed_key])
        if kind == "dtcwt":
            return DTCWTResidualAdapter(
                channels=512, bottleneck=bottleneck, levels=3,
                biort="near_sym_a", qshift="qshift_a", mode="symmetric")
        return SpatialResidualAdapter(bottleneck=bottleneck)


class RoutedCIF(CIF):
    """Original CIF with optional task-specific post-fusion adapters."""

    def __init__(self, num_classes, last_inp_channels,
                 element_adapter: tuple[str, int] = ("identity", 0),
                 corrosion_adapter: tuple[str, int] = ("identity", 0),
                 no_cross: bool = False):
        super().__init__(num_classes=num_classes,
                         last_inp_channels=last_inp_channels)
        self.element_adapter_kind = element_adapter[0]
        self.corrosion_adapter_kind = corrosion_adapter[0]
        self.no_cross = bool(no_cross)
        self.element_adapter = _construct_adapter(*element_adapter)
        self.corrosion_adapter = _construct_adapter(*corrosion_adapter)

    def forward(self, element_feature: torch.Tensor,
                corrosion_feature: torch.Tensor):
        if self.no_cross:
            # The two original attention modules stay registered for state-dict
            # compatibility, but are intentionally unused in this ablation.
            fused_element = element_feature
            fused_corrosion = corrosion_feature
        else:
            # This is the original CIF's bidirectional equation and ordering.
            corrosion_to_element = self.attention(element_feature)
            element_to_corrosion = self.attention_1(corrosion_feature)
            fused_element = element_feature + torch.mul(
                corrosion_feature, corrosion_to_element)
            fused_corrosion = corrosion_feature + torch.mul(
                element_feature, element_to_corrosion)

        fused_element = self.element_adapter(fused_element)
        fused_corrosion = self.corrosion_adapter(fused_corrosion)
        return (self.last_layer_e(fused_element),
                self.last_layer_d(fused_corrosion))


_ROUTES: dict[str, dict[str, Any]] = {
    "G0": {"element": ("identity", 0), "corrosion": ("identity", 0)},
    "G-E": {"element": ("dtcwt", 8), "corrosion": ("identity", 0)},
    "G-D": {"element": ("identity", 0), "corrosion": ("spatial", 16)},
    "G4": {"element": ("dtcwt", 8), "corrosion": ("spatial", 16)},
    "G4-SS": {"element": ("spatial", 8), "corrosion": ("spatial", 16)},
    "G4-Swap": {"element": ("spatial", 16), "corrosion": ("dtcwt", 8)},
    "G4-NoCross": {"element": ("dtcwt", 8), "corrosion": ("spatial", 16),
                   "no_cross": True},
}


def build_model(group: str, *, num_classes=(7, 2),
                backbone: str = "hrnetv2_w48", pretrained: bool = False,
                init_seed: int = 20261112) -> nn.Module:
    """Build a model from the one shared baseline constructor.

    `init_seed` fixes baseline-side random initialization. Adapter-specific
    initializers are independently seeded by type and bottleneck so exchanging
    an adapter's task route does not change that adapter's starting weights.
    """
    if group not in GROUPS:
        raise ValueError(f"Unknown G4-v1 group {group!r}; expected one of {GROUPS}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = AECIF_Net(num_classes=num_classes, backbone=backbone,
                          pretrained=pretrained)
        if not pretrained:
            # Matches the original train_200.py initialization call before its
            # partial pretrained checkpoint load.
            weights_init(model)
        if group == "G0":
            return model
        route = _ROUTES[group]
        old_cif = model.cif
        channels = int(sum(model.backbone.pre_stage_channels))
        routed = RoutedCIF(
            num_classes=num_classes, last_inp_channels=channels,
            element_adapter=route["element"],
            corrosion_adapter=route["corrosion"],
            no_cross=route.get("no_cross", False))
        incompatible = routed.load_state_dict(old_cif.state_dict(), strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected base-CIF keys: {incompatible.unexpected_keys}")
        unexpected_missing = [key for key in incompatible.missing_keys
                              if not key.startswith(("element_adapter.",
                                                     "corrosion_adapter."))]
        if unexpected_missing:
            raise RuntimeError(f"Missing original CIF keys: {unexpected_missing}")
        model.cif = routed
        return model


def load_baseline_initialization(model: nn.Module, checkpoint_path: str | Path,
                                 expected_sha256: str | None = None,
                                 expected_loaded_tensors: int = 1830) -> dict[str, Any]:
    """Apply the audited historical partial-load rule on CPU.

    The original loader strips the first seven characters from every source key
    (`module.`) and loads only exact key/shape matches. This helper keeps that
    rule, reports every skipped key, and checks the known 1830 matched tensors.
    It does not alter the source checkpoint.
    """
    import hashlib

    path = Path(checkpoint_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
        raise RuntimeError(f"Initialization SHA-256 mismatch: {actual_sha}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("state_dict", payload.get("model_state_dict", payload))
    if not isinstance(source, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    target = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in source.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        # Exact historical behavior: DataParallel prefix is removed by slicing.
        name = key[7:]
        if name in target and tuple(target[name].shape) == tuple(value.shape):
            matched[name] = value
        else:
            skipped.append(key)
    if len(matched) != expected_loaded_tensors:
        raise RuntimeError(
            f"Initialization match count changed: {len(matched)} != "
            f"{expected_loaded_tensors}")
    target.update(matched)
    model.load_state_dict(target, strict=True)
    del payload, source, target
    return {"path": str(path), "sha256": actual_sha,
            "loaded_tensors": len(matched), "skipped_source_keys": len(skipped),
            "skipped_source_key_examples": skipped[:20]}


def adapter_parameter_counts(model: nn.Module) -> dict[str, int]:
    if not isinstance(model.cif, RoutedCIF):
        return {"element": 0, "corrosion": 0, "total": 0}
    element = sum(p.numel() for p in model.cif.element_adapter.parameters())
    corrosion = sum(p.numel() for p in model.cif.corrosion_adapter.parameters())
    return {"element": element, "corrosion": corrosion,
            "total": element + corrosion}


def no_cross_invalidated_parameters(model: nn.Module) -> dict[str, int]:
    """Count retained-but-unexecuted CIF attention parameters in NoCross."""
    if not isinstance(model.cif, RoutedCIF) or not model.cif.no_cross:
        return {"invalidated": 0, "effective_total": sum(
            p.numel() for p in model.parameters())}
    invalidated = sum(p.numel() for module in
                      (model.cif.attention, model.cif.attention_1)
                      for p in module.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"invalidated": invalidated, "effective_total": total - invalidated,
            "registered_total": total}
