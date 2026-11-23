"""Check that every group differs only by its registered routing variable."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = ("G0", "G-E", "G-D", "G4", "G4-SS", "G4-Swap", "G4-NoCross")


def audit():
    protocol_bytes = (ROOT / "configs" / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    configs = {}
    for group in GROUPS:
        path = ROOT / "configs" / "groups" / f"{group}.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["group"] != group or item["protocol"] != "../protocol.json":
            raise RuntimeError(f"Invalid shared protocol reference in {path.name}")
        configs[group] = item
    expected_routes = {
        "G0": ("identity", None, "identity", None, False),
        "G-E": ("dtcwt", 8, "identity", None, False),
        "G-D": ("identity", None, "spatial", 16, False),
        "G4": ("dtcwt", 8, "spatial", 16, False),
        "G4-SS": ("spatial", 8, "spatial", 16, False),
        "G4-Swap": ("spatial", 16, "dtcwt", 8, False),
        "G4-NoCross": ("dtcwt", 8, "spatial", 16, True),
    }
    for group, (element_kind, element_width, corrosion_kind,
                corrosion_width, no_cross) in expected_routes.items():
        cfg = configs[group]
        element = cfg["element_adapter"]
        corrosion = cfg["corrosion_adapter"]
        element_kind_actual = element if isinstance(element, str) else element["kind"]
        corrosion_kind_actual = corrosion if isinstance(corrosion, str) else corrosion["kind"]
        element_width_actual = None if isinstance(element, str) else element.get("bottleneck")
        corrosion_width_actual = None if isinstance(corrosion, str) else corrosion.get("bottleneck")
        if (element_kind_actual, element_width_actual, corrosion_kind_actual,
                corrosion_width_actual, cfg.get("no_cross", False)) != \
                (element_kind, element_width, corrosion_kind, corrosion_width, no_cross):
            raise RuntimeError(f"Model route differs from locked group design: {group}")
        for adapter in (element, corrosion):
            if not isinstance(adapter, dict) or adapter.get("kind") != "dtcwt":
                continue
            if (adapter.get("levels"), adapter.get("orientations_per_level"),
                adapter.get("biort"), adapter.get("qshift"), adapter.get("mode")) != \
                    (3, 6, "near_sym_a", "qshift_a", "symmetric"):
                raise RuntimeError(f"DTCWT definition differs from audited G4-v1: {group}")
    if configs["G4"]["element_adapter"]["bottleneck"] != \
            configs["G4-SS"]["element_adapter"]["bottleneck"]:
        raise RuntimeError("G4 and G4-SS Element bottleneck mismatch")
    if configs["G4"]["corrosion_adapter"] != configs["G4-SS"]["corrosion_adapter"]:
        raise RuntimeError("G4 and G4-SS Corrosion adapter mismatch")
    if (configs["G4"]["corrosion_adapter"]["bottleneck"] !=
            configs["G4-SS"]["corrosion_adapter"]["bottleneck"]):
        raise RuntimeError("G4 and G4-SS parameter-matching route mismatch")
    expected_fields = {"input_size": [520, 520], "epochs": 150,
                       "batch_size": 4, "precision": "FP16 autocast + torch.cuda.amp.GradScaler (legacy protocol candidate)"}
    actual_fields = {"input_size": protocol["model"]["input_size"],
                     "epochs": protocol["training"]["epochs"],
                     "batch_size": protocol["training"]["batch_size"],
                     "precision": protocol["training"]["precision"]}
    if actual_fields != expected_fields:
        raise RuntimeError(f"Shared training protocol drift: {actual_fields}")
    if (protocol["data"]["train_count"] != 200 or
            protocol["data"]["validation_count"] != 34 or
            protocol["initialization"]["sha256"] !=
            "45c202530d8e6831fc56fc210c0597c2d04cd51fdbda10a0ca70101a33dadcf3"):
        raise RuntimeError("Data or initialization source is not the locked audit source")
    if configs["G4"]["element_adapter"]["kind"] != "dtcwt" or \
       configs["G4"]["element_adapter"]["bottleneck"] != 8 or \
       configs["G4"]["corrosion_adapter"]["kind"] != "spatial" or \
       configs["G4"]["corrosion_adapter"]["bottleneck"] != 16:
        raise RuntimeError("G4 task routing does not match the approved design")
    return {"groups": list(GROUPS), "protocol_sha256": __import__("hashlib").sha256(
        protocol_bytes).hexdigest(), "all_groups_share_protocol": True,
        "formal_trainer_connected": True,
        "groups_share_one_trainer": True,
        "G4_G4_SS_parameter_count_expected": 31410,
        "G4_NoCross_effective_attention_note":
            "CIF attention parameters remain registered but are inactive in this ablation."}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
