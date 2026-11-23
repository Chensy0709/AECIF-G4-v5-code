from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from compute_costs import parameter_count
from g4_model import (GROUPS, RoutedCIF, adapter_parameter_counts, build_model,
                      load_baseline_initialization)
from models.dtcwt_adapter import DTCWTResidualAdapter


PROTOCOL = json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))
INITIALIZATION = PROTOCOL["initialization"]


def tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def common_state_digests(model: torch.nn.Module) -> dict[str, str]:
    result = {}
    for key, value in model.state_dict().items():
        if key.startswith(("cif.element_adapter.", "cif.corrosion_adapter.")):
            continue
        result[key] = tensor_digest(value)
    return result


class G4CPUContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        torch.use_deterministic_algorithms(False)

    def test_split_manifests_are_exact_200_and_34(self):
        expected = [("train_200.txt", 200, PROTOCOL["data"]["train_manifest_sha256"]),
                    ("val.txt", 34, PROTOCOL["data"]["validation_manifest_sha256"])]
        for filename, count, expected_sha in expected:
            path = ROOT / "configs" / "data_splits" / filename
            import hashlib
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            ids = [line.strip().split()[0] for line in
                   path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            self.assertEqual(actual_sha, expected_sha)
            self.assertEqual(len(ids), count)
            self.assertEqual(len(set(ids)), count)
        train = {x.strip().split()[0] for x in
                 (ROOT / "configs/data_splits/train_200.txt").read_text(encoding="utf-8-sig").splitlines() if x.strip()}
        val = {x.strip().split()[0] for x in
               (ROOT / "configs/data_splits/val.txt").read_text(encoding="utf-8-sig").splitlines() if x.strip()}
        self.assertFalse(train.intersection(val))

    def test_all_groups_instantiate_have_shapes_and_expected_adapter_params(self):
        group_rows = {}
        for group in GROUPS:
            model = build_model(group)
            model.eval()
            with torch.inference_mode():
                outputs = model(torch.zeros(1, 3, 64, 64))
            self.assertEqual(tuple(outputs[0].shape), (1, 7, 64, 64), group)
            self.assertEqual(tuple(outputs[1].shape), (1, 2, 64, 64), group)
            total = parameter_count(model)["total"]
            adapters = adapter_parameter_counts(model)
            expected_added = {"G0": 0, "G-E": 9873, "G-D": 21537,
                              "G4": 31410, "G4-SS": 31410,
                              "G4-Swap": 31410, "G4-NoCross": 31410}[group]
            self.assertEqual(adapters["total"], expected_added, group)
            group_rows[group] = {"total_parameters": total,
                                 "adapter_parameters": adapters["total"]}
            del outputs, model
        self.assertEqual(group_rows["G4"]["total_parameters"],
                         group_rows["G4-SS"]["total_parameters"])
        self.assertEqual(group_rows["G4"]["total_parameters"],
                         group_rows["G4-Swap"]["total_parameters"])
        report = ROOT / "reports" / "CPU_PARAMETER_COUNTS.json"
        report.write_text(json.dumps(group_rows, indent=2), encoding="utf-8")

    def test_initialization_is_shared_and_adapter_route_seed_is_stable(self):
        reference = build_model("G0")
        reference_digest = common_state_digests(reference)
        del reference
        for group in GROUPS[1:]:
            model = build_model(group)
            self.assertEqual(common_state_digests(model), reference_digest, group)
            del model

        g4 = build_model("G4")
        swap = build_model("G4-Swap")
        self.assertEqual(common_state_digests(g4), common_state_digests(swap))
        self.assertEqual(common_state_digests(g4), reference_digest)
        dtcwt_g4 = g4.cif.element_adapter.state_dict()
        dtcwt_swap = swap.cif.corrosion_adapter.state_dict()
        spatial_g4 = g4.cif.corrosion_adapter.state_dict()
        spatial_swap = swap.cif.element_adapter.state_dict()
        self.assertEqual(set(dtcwt_g4), set(dtcwt_swap))
        self.assertEqual(set(spatial_g4), set(spatial_swap))
        for key in dtcwt_g4:
            self.assertTrue(torch.equal(dtcwt_g4[key], dtcwt_swap[key]), key)
        for key in spatial_g4:
            self.assertTrue(torch.equal(spatial_g4[key], spatial_swap[key]), key)

    def test_original_partial_checkpoint_loads_into_g0_and_g4(self):
        loaded_models = []
        base_digests = []
        for group in ("G0", "G4"):
            model = build_model(group)
            audit = load_baseline_initialization(
                model, INITIALIZATION["path"], INITIALIZATION["sha256"], 1830)
            self.assertEqual(audit["loaded_tensors"], 1830)
            self.assertEqual(audit["sha256"], INITIALIZATION["sha256"])
            base_digests.append(common_state_digests(model))
            loaded_models.append(model.eval())
        self.assertEqual(base_digests[0], base_digests[1])
        g4 = loaded_models[1]
        self.assertEqual(float(g4.cif.element_adapter.alpha.item()), 0.0)
        self.assertEqual(float(g4.cif.corrosion_adapter.alpha.item()), 0.0)
        x = torch.linspace(-1.0, 1.0, steps=3 * 520 * 520,
                           dtype=torch.float32).view(1, 3, 520, 520)
        with torch.inference_mode():
            y0 = loaded_models[0](x)
            y4 = loaded_models[1](x)
        self.assertEqual(tuple(y0[0].shape), (1, 7, 520, 520))
        self.assertEqual(tuple(y0[1].shape), (1, 2, 520, 520))
        for baseline_output, g4_output in zip(y0, y4):
            self.assertTrue(torch.equal(baseline_output, g4_output))
            self.assertEqual(float((baseline_output - g4_output).abs().max()), 0.0)

    def test_task_routes_feed_the_correct_classifier_heads(self):
        cases = [("G4", DTCWTResidualAdapter, "spatial"),
                 ("G4-Swap", "spatial", DTCWTResidualAdapter)]
        for group, expected_element, expected_corrosion in cases:
            model = build_model(group)
            cif = model.cif
            if expected_element == "spatial":
                self.assertEqual(cif.element_adapter.__class__.__name__,
                                 "SpatialResidualAdapter")
            else:
                self.assertIsInstance(cif.element_adapter, expected_element)
            if expected_corrosion == "spatial":
                self.assertEqual(cif.corrosion_adapter.__class__.__name__,
                                 "SpatialResidualAdapter")
            else:
                self.assertIsInstance(cif.corrosion_adapter, expected_corrosion)
            seen = {"e_adapter": None, "d_adapter": None, "e_head": None,
                    "d_head": None}
            handles = [
                cif.element_adapter.register_forward_hook(
                    lambda _m, _i, out: seen.__setitem__("e_adapter", out.detach().clone())),
                cif.corrosion_adapter.register_forward_hook(
                    lambda _m, _i, out: seen.__setitem__("d_adapter", out.detach().clone())),
                cif.last_layer_e.register_forward_pre_hook(
                    lambda _m, args: seen.__setitem__("e_head", args[0].detach().clone())),
                cif.last_layer_d.register_forward_pre_hook(
                    lambda _m, args: seen.__setitem__("d_head", args[0].detach().clone())),
            ]
            cif.eval()
            with torch.inference_mode():
                outputs = cif(torch.randn(1, 512, 32, 32),
                              torch.randn(1, 512, 32, 32))
            for handle in handles:
                handle.remove()
            self.assertTrue(torch.equal(seen["e_adapter"], seen["e_head"]))
            self.assertTrue(torch.equal(seen["d_adapter"], seen["d_head"]))
            self.assertEqual(tuple(outputs[0].shape), (1, 7, 32, 32))
            self.assertEqual(tuple(outputs[1].shape), (1, 2, 32, 32))
            del model

    def test_cif_matches_original_equations_and_no_cross_is_marked(self):
        from baseline_code.nets.backbone import CIF
        original = CIF(num_classes=(7, 2), last_inp_channels=720).eval()
        routed = RoutedCIF(num_classes=(7, 2), last_inp_channels=720).eval()
        self.assertEqual(routed.load_state_dict(original.state_dict(), strict=False).missing_keys, [])
        x = torch.randn(1, 512, 16, 16)
        d = torch.randn(1, 512, 16, 16)
        with torch.inference_mode():
            expected = original(x, d)
            actual = routed(x, d)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        no_cross = build_model("G4-NoCross")
        accounting = sum(p.numel() for m in
                         (no_cross.cif.attention, no_cross.cif.attention_1)
                         for p in m.parameters())
        self.assertEqual(accounting, 4719616)
        self.assertTrue(no_cross.cif.no_cross)

    def test_dtcwt_cpu_forward_and_finite_differentiable_path(self):
        adapter = DTCWTResidualAdapter(channels=512, bottleneck=8, levels=3,
                                       biort="near_sym_a", qshift="qshift_a",
                                       mode="symmetric")
        adapter.alpha.data.fill_(0.1)
        x = torch.randn(1, 512, 32, 32, requires_grad=True)
        y = adapter(x)
        self.assertEqual(tuple(y.shape), (1, 512, 32, 32))
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(adapter.reduce.weight.grad).all())
        self.assertTrue(torch.isfinite(adapter.alpha.grad).all())

    def test_spatial16_matches_the_audited_g2_adapter(self):
        from models.matched_spatial_adapter import (
            SpatialResidualAdapter as AuditedSpatial16)
        from g4_model import SpatialResidualAdapter as G4Spatial
        audited = AuditedSpatial16().eval()
        g4_spatial = G4Spatial(16).eval()
        incompatible = g4_spatial.load_state_dict(audited.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        x = torch.randn(1, 512, 24, 24)
        with torch.inference_mode():
            expected = audited(x)
            actual = g4_spatial(x)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(sum(p.numel() for p in g4_spatial.parameters()), 21537)

    def test_g4_no_cross_attention_is_registered_but_not_used(self):
        model = build_model("G4-NoCross")
        seen = {"e": 0, "d": 0}
        handles = [model.cif.attention.register_forward_hook(
                       lambda *_: seen.__setitem__("e", seen["e"] + 1)),
                   model.cif.attention_1.register_forward_hook(
                       lambda *_: seen.__setitem__("d", seen["d"] + 1))]
        model.cif.eval()
        with torch.inference_mode():
            model.cif(torch.randn(1, 512, 16, 16),
                       torch.randn(1, 512, 16, 16))
        for handle in handles:
            handle.remove()
        self.assertEqual(seen, {"e": 0, "d": 0})
        self.assertEqual(sum(p.numel() for m in
                             (model.cif.attention, model.cif.attention_1)
                             for p in m.parameters()), 4719616)


if __name__ == "__main__":
    unittest.main(verbosity=2)
