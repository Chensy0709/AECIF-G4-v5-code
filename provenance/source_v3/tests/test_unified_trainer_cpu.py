from __future__ import annotations

import json
import hashlib
import os
import random
import subprocess
import sys
import tempfile
import time
import textwrap
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
import numpy as np
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from checkpointing import (capture_rng_state, restore_latest_or_previous,
                           save_epoch_checkpoint, sha256_file,
                           state_dict_sha256, write_best_model_artifact,
                           collect_unreferenced_best_artifacts, SCHEMA_VERSION)
from source_lock import verify_source_lock
from trainer_core import (DWAWeights, get_original_cosine_schedule,
                          original_dwa_weights, run_train_epoch,
                          run_validation, should_update_best)
from unified_trainer import load_locked_configs
import train_entry
from baseline_code.nets.hrnet_training import CE_Loss


class _Adapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x + self.alpha * self.conv(x)


class _ToyCIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.element_adapter = _Adapter(4)
        self.corrosion_adapter = _Adapter(4)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Conv2d(3, 4, 1)
        self.cif = _ToyCIF()
        self.element_head = nn.Conv2d(4, 7, 1)
        self.corrosion_head = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        shared = self.shared(x)
        return (self.element_head(self.cif.element_adapter(shared)),
                self.corrosion_head(self.cif.corrosion_adapter(shared)))


def _batch(n=2):
    return (torch.randn(n, 3, 5, 5),
            torch.randint(0, 8, (n, 5, 5)),
            torch.randint(0, 3, (n, 5, 5)))


class UnifiedTrainerCPUContract(unittest.TestCase):
    def test_dwa_matches_legacy_formula_and_epoch_boundary(self):
        w1 = original_dwa_weights(0, {"losses_e": [], "losses_d": []})
        self.assertEqual((w1.a_element, w1.a_corrosion), (1.0, 1.0))
        history = {"losses_e": [2.0, 1.0], "losses_d": [4.0, 2.0]}
        w3 = original_dwa_weights(2, history)
        self.assertAlmostEqual(w3.ratio_element, 0.5)
        self.assertAlmostEqual(w3.ratio_corrosion, 0.5)
        self.assertAlmostEqual(w3.a_element + w3.a_corrosion, 2.0)

    def test_original_cosine_schedule_endpoints_and_tail(self):
        schedule = get_original_cosine_schedule(3e-4, 3e-6, 150)
        from baseline_code.nets.hrnet_training import get_lr_scheduler
        legacy = get_lr_scheduler("cos", 3e-4, 3e-6, 150)
        self.assertEqual([schedule(i) for i in range(150)],
                         [legacy(i) for i in range(150)])
        self.assertAlmostEqual(schedule(0), 3e-5, places=10)
        self.assertAlmostEqual(schedule(3), 3e-4, places=10)
        self.assertAlmostEqual(schedule(135), 3e-6, places=10)
        self.assertAlmostEqual(schedule(149), 3e-6, places=10)

    def test_best_selection_rule_preserves_minimum_and_later_tie(self):
        self.assertTrue(should_update_best(.5, None))
        self.assertTrue(should_update_best(.5, .5))
        self.assertTrue(should_update_best(.49, .5))
        self.assertFalse(should_update_best(.51, .5))

    def test_cpu_training_updates_model_and_records_adapter_gradients(self):
        torch.manual_seed(31)
        model = _ToyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = {"losses_e": [], "losses_d": []}
        before = model.shared.weight.detach().clone()
        result = run_train_epoch(model, [_batch(), _batch()], optimizer,
                                 epoch_index=0, dwa_history=history, device="cpu",
                                 collect_gradient_details=True)
        self.assertEqual(result["steps"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["train_loss"])))
        self.assertEqual(len(history["losses_e"]), 1)
        self.assertFalse(torch.equal(before, model.shared.weight.detach()))
        for task in ("element", "corrosion"):
            record = result["step_records"][0]["gradients"]["adapter_gradients"][task]
            self.assertIsNotNone(record["alpha_grad"])
            self.assertGreater(record["alpha_grad"], 0)

    def test_validation_uses_sample_weighted_batch_means(self):
        torch.manual_seed(8)
        model = _ToyModel().eval()
        data = [_batch(2), _batch(1)]
        weights = DWAWeights(1.2, 0.8, 1.0, 1.0)
        actual = run_validation(model, data, device="cpu", dwa=weights)
        expected_total = 0.0
        for batch in data:
            outputs = model(batch[0])
            e = CE_Loss(outputs[0], batch[1], torch.ones(7), num_classes=7)
            c = CE_Loss(outputs[1], batch[2], torch.ones(2), num_classes=2)
            expected_total += (weights.a_element * e.item() +
                               weights.a_corrosion * c.item()) * len(batch[0])
        self.assertEqual(actual["validation_samples"], 3)
        self.assertEqual(actual["validation_batches"], 2)
        self.assertAlmostEqual(actual["val_loss"], expected_total / 3, places=6)

    def test_checkpoint_roundtrip_and_corrupt_latest_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model = _ToyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Initialize Adam state so the saved optimizer has realistic state.
            model(_batch()[0])[0].sum().backward(); optimizer.step(); optimizer.zero_grad()
            artifact = write_best_model_artifact(out, model.state_dict())
            expected = {"trainer_id": "cpu-test", "group": "G0",
                        "precision_mode": "fp32_cpu_test", "protocol_sha256": "p",
                        "config_sha256": "c", "source_manifest_sha256": "s",
                        "initialization_sha256": "i", "runtime_sha256": "r"}
            row = {"epoch": 1, "train_loss": 1.0, "train_loss_element": .5,
                   "train_loss_corrosion": .5, "val_loss": .75,
                   "val_loss_element": .4, "val_loss_corrosion": .35,
                   "dwa_element": 1., "dwa_corrosion": 1., "learning_rate": .001,
                   "overflow_count": 0, "optimizer_step_skipped_count": 0,
                   "epoch_seconds": .1, "batch_records": [{"step": 0}]}
            payload = {"schema_version": SCHEMA_VERSION, **expected, "completed_epoch": 0,
                       "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict(),
                       "scaler_state_dict": None,
                       "scheduler_state": {"type": "cos", "completed_epoch": 1,
                                           "last_lr": [.001]},
                       "dwa_state": {"losses_e": [.5], "losses_d": [.5]},
                       "rng_state": capture_rng_state("cpu"),
                       "sampler_state": {"epoch": 0},
                       "best_state": {"epoch": 1, "validation_loss": .75,
                                      **artifact},
                       "metrics_history": [row], "runtime": {"test": True},
                       "training_mode": True}
            save_epoch_checkpoint(out, payload)
            # Create previous as a valid older checkpoint, then corrupt latest.
            payload2 = dict(payload); payload2["completed_epoch"] = 1
            payload2["scheduler_state"] = {"type": "cos", "completed_epoch": 2,
                                            "last_lr": [.0009]}
            payload2["optimizer_state_dict"] = dict(payload["optimizer_state_dict"])
            payload2["optimizer_state_dict"]["param_groups"] = [
                dict(group, lr=.0009)
                for group in payload["optimizer_state_dict"]["param_groups"]]
            row2 = dict(row); row2["epoch"] = 2; row2["val_loss"] = .7
            payload2["metrics_history"] = [row, row2]
            payload2["dwa_state"] = {"losses_e": [.5, .4], "losses_d": [.5, .4]}
            payload2["best_state"] = dict(payload["best_state"])
            save_epoch_checkpoint(out, payload2)
            (out / "checkpoint_latest.pth").write_bytes(b"broken")
            random.setstate(payload["rng_state"]["python"])
            expected_python_random = random.random()
            np.random.set_state(payload["rng_state"]["numpy"])
            expected_numpy_random = float(np.random.random())
            torch.set_rng_state(payload["rng_state"]["torch_cpu"])
            expected_torch_random = torch.rand(4)
            restored_model = _ToyModel()
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
            restored = restore_latest_or_previous(
                out, expected=expected, model=restored_model,
                optimizer=restored_optimizer, device="cpu")
            self.assertEqual(random.random(), expected_python_random)
            self.assertEqual(float(np.random.random()), expected_numpy_random)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch_random))
            self.assertEqual(restored["payload"]["completed_epoch"], 0)
            self.assertIn("checkpoint_latest.pth", str(restored["fallback_errors"]))
            for key, tensor in payload["model_state_dict"].items():
                self.assertTrue(torch.equal(restored_model.state_dict()[key], tensor), key)
            self.assertEqual(restored_optimizer.state_dict()["param_groups"],
                             payload["optimizer_state_dict"]["param_groups"])
            for parameter_id, saved_state in payload["optimizer_state_dict"]["state"].items():
                loaded_state = restored_optimizer.state_dict()["state"][parameter_id]
                self.assertEqual(set(loaded_state), set(saved_state))
                for key, value in saved_state.items():
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(loaded_state[key], value),
                                        f"optimizer state {parameter_id}/{key}")
                    else:
                        self.assertEqual(loaded_state[key], value)
            self.assertEqual(restored["payload"]["dwa_state"], payload["dwa_state"])
            self.assertEqual(restored["payload"]["scheduler_state"],
                             payload["scheduler_state"])
            self.assertEqual(restored["payload"]["best_state"], payload["best_state"])
            self.assertTrue((out / "batch_diagnostics.jsonl").is_file())
            self.assertEqual(len((out / "batch_diagnostics.jsonl").read_text().splitlines()), 1)

    def test_content_addressed_best_gc_preserves_latest_previous_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_best_model_artifact(root, _ToyModel().state_dict())
            model2 = _ToyModel()
            with torch.no_grad():
                model2.shared.weight.add_(1)
            second = write_best_model_artifact(root, model2.state_dict())
            legacy = root / "best_epoch_001.pth"
            legacy.write_bytes(b"historical-artifact")
            removed = collect_unreferenced_best_artifacts(root, [second, first])
            self.assertEqual(removed, [])
            removed = collect_unreferenced_best_artifacts(root, [second])
            self.assertEqual(removed, [first["path"]])
            self.assertTrue((root / second["path"]).is_file())
            self.assertTrue(legacy.is_file())

    def test_latest_restore_preserves_both_best_fallbacks_and_collects_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            expected = {"trainer_id": "gc-test", "group": "G0",
                        "precision_mode": "fp32_cpu_test", "protocol_sha256": "p",
                        "config_sha256": "c", "source_manifest_sha256": "s",
                        "initialization_sha256": "i", "runtime_sha256": "r"}
            model = _ToyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model(_batch()[0])[0].sum().backward(); optimizer.step(); optimizer.zero_grad()
            artifact1 = write_best_model_artifact(out, model.state_dict())
            row1 = {"epoch": 1, "val_loss": .8, "batch_records": []}
            base = {"schema_version": SCHEMA_VERSION, **expected,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": None,
                    "dwa_state": {"losses_e": [.5], "losses_d": [.4]},
                    "rng_state": capture_rng_state("cpu"),
                    "sampler_state": {"epoch": 0},
                    "runtime": {"cpu_test": True}, "training_mode": True}
            first = {**base, "completed_epoch": 0,
                     "scheduler_state": {"type": "cos", "completed_epoch": 1,
                                         "last_lr": [.001]},
                     "best_state": {"epoch": 1, "validation_loss": .8, **artifact1},
                     "metrics_history": [row1]}
            save_epoch_checkpoint(out, first)

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(.01)
            artifact2 = write_best_model_artifact(out, model.state_dict())
            row2 = {"epoch": 2, "val_loss": .7, "batch_records": []}
            second = {**base, "completed_epoch": 1,
                      "model_state_dict": model.state_dict(),
                      "scheduler_state": {"type": "cos", "completed_epoch": 2,
                                          "last_lr": [.001]},
                      "dwa_state": {"losses_e": [.5, .45], "losses_d": [.4, .36]},
                      "best_state": {"epoch": 2, "validation_loss": .7, **artifact2},
                      "metrics_history": [row1, row2]}
            save_epoch_checkpoint(out, second)
            orphan = write_best_model_artifact(out, _ToyModel().state_dict())
            self.assertTrue((out / orphan["path"]).is_file())
            restored_model = _ToyModel()
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
            restore_latest_or_previous(out, expected=expected,
                                       model=restored_model,
                                       optimizer=restored_optimizer, device="cpu")
            self.assertTrue((out / artifact1["path"]).is_file())
            self.assertTrue((out / artifact2["path"]).is_file())
            self.assertFalse((out / orphan["path"]).exists())

    def test_checkpoint_temp_write_process_kill_recovers_committed_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "run"
            out.mkdir()
            model = _ToyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model(_batch()[0])[0].sum().backward()
            optimizer.step(); optimizer.zero_grad()
            artifact = write_best_model_artifact(out, model.state_dict())
            expected = {"trainer_id": "cpu-process-test", "group": "G0",
                        "precision_mode": "fp32_cpu_test", "protocol_sha256": "p",
                        "config_sha256": "c", "source_manifest_sha256": "s",
                        "initialization_sha256": "i", "runtime_sha256": "r"}
            row = {"epoch": 1, "train_loss": 1., "train_loss_element": .5,
                   "train_loss_corrosion": .5, "val_loss": .75,
                   "val_loss_element": .4, "val_loss_corrosion": .35,
                   "dwa_element": 1., "dwa_corrosion": 1., "learning_rate": .001,
                   "overflow_count": 0, "optimizer_step_skipped_count": 0,
                   "epoch_seconds": .1, "batch_records": []}
            payload = {"schema_version": SCHEMA_VERSION, **expected,
                       "completed_epoch": 0, "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict(),
                       "scaler_state_dict": None,
                       "scheduler_state": {"type": "cos", "completed_epoch": 1,
                                           "last_lr": [.001]},
                       "dwa_state": {"losses_e": [.5], "losses_d": [.5]},
                       "rng_state": capture_rng_state("cpu"),
                       "sampler_state": {"epoch": 0}, "best_state": {
                           "epoch": 1, "validation_loss": .75, **artifact},
                       "metrics_history": [row], "runtime": {"cpu_test": True},
                       "training_mode": True}
            save_epoch_checkpoint(out, payload)
            latest_before = sha256_file(out / "checkpoint_latest.pth")
            next_payload = dict(payload)
            next_payload["completed_epoch"] = 1
            next_payload["scheduler_state"] = {"type": "cos", "completed_epoch": 2,
                                                "last_lr": [.0009]}
            next_payload["optimizer_state_dict"] = dict(payload["optimizer_state_dict"])
            next_payload["optimizer_state_dict"]["param_groups"] = [
                dict(group, lr=.0009)
                for group in payload["optimizer_state_dict"]["param_groups"]]
            row2 = dict(row, epoch=2, val_loss=.7)
            next_payload["metrics_history"] = [row, row2]
            next_payload["dwa_state"] = {"losses_e": [.5, .4],
                                          "losses_d": [.5, .4]}
            staged_payload = root / "next_payload.pth"
            torch.save(next_payload, staged_payload)
            marker = root / "child_failpoint.txt"
            script = root / "interrupt_writer.py"
            script.write_text(textwrap.dedent("""
                import sys, time, torch
                from pathlib import Path
                sys.path.insert(0, sys.argv[4])
                from checkpointing import save_epoch_checkpoint
                out, payload_path, marker = map(Path, sys.argv[1:4])
                payload = torch.load(payload_path, map_location="cpu", weights_only=False)
                def stop(stage):
                    if stage == "after_temp_fsync":
                        marker.write_text(stage, encoding="utf-8")
                        time.sleep(120)
                save_epoch_checkpoint(out, payload, failpoint=stop)
            """), encoding="utf-8")
            child_log = root / "child.log"
            with child_log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    [sys.executable, str(script), str(out), str(staged_payload),
                     str(marker), str(ROOT)],
                    cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline and process.poll() is None and not marker.exists():
                        time.sleep(.05)
                    self.assertTrue(marker.exists(), "child never reached post-fsync failpoint")
                    self.assertEqual(marker.read_text(encoding="utf-8"), "after_temp_fsync")
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=15)
            self.assertEqual(sha256_file(out / "checkpoint_latest.pth"), latest_before)
            self.assertTrue(list(out.glob("checkpoint_latest.pth.*.tmp")))
            restored_model = _ToyModel()
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=.001)
            restored = restore_latest_or_previous(
                out, expected=expected, model=restored_model,
                optimizer=restored_optimizer, device="cpu")
            self.assertEqual(restored["payload"]["completed_epoch"], 0)
            self.assertFalse(list(out.glob("*.tmp")))
            self.assertEqual(sha256_file(out / "checkpoint_latest.pth"), latest_before)

    def test_best_weight_temp_write_process_kill_preserves_committed_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "run"
            out.mkdir()
            model = _ToyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model(_batch()[0])[0].sum().backward(); optimizer.step(); optimizer.zero_grad()
            artifact = write_best_model_artifact(out, model.state_dict())
            expected = {"trainer_id": "cpu-best-interrupt", "group": "G0",
                        "precision_mode": "fp32_cpu_test", "protocol_sha256": "p",
                        "config_sha256": "c", "source_manifest_sha256": "s",
                        "initialization_sha256": "i", "runtime_sha256": "r"}
            row = {"epoch": 1, "val_loss": .75, "batch_records": []}
            payload = {"schema_version": SCHEMA_VERSION, **expected,
                       "completed_epoch": 0, "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict(),
                       "scaler_state_dict": None,
                       "scheduler_state": {"type": "cos", "completed_epoch": 1,
                                           "last_lr": [.001]},
                       "dwa_state": {"losses_e": [.5], "losses_d": [.5]},
                       "rng_state": capture_rng_state("cpu"),
                       "sampler_state": {"epoch": 0},
                       "best_state": {"epoch": 1, "validation_loss": .75, **artifact},
                       "metrics_history": [row], "runtime": {"cpu_test": True},
                       "training_mode": True}
            save_epoch_checkpoint(out, payload)
            original_latest_sha = sha256_file(out / "checkpoint_latest.pth")
            original_best_sha = sha256_file(out / artifact["path"])
            candidate_state = {key: value.detach().clone()
                               for key, value in model.state_dict().items()}
            candidate_state["shared.weight"].add_(5)
            staged = root / "candidate_state.pth"
            torch.save(candidate_state, staged)
            marker = root / "best_failpoint.txt"
            script = root / "interrupt_best_writer.py"
            script.write_text(textwrap.dedent("""
                import sys, time, torch
                from pathlib import Path
                sys.path.insert(0, sys.argv[4])
                from checkpointing import write_best_model_artifact
                out, state_path, marker = map(Path, sys.argv[1:4])
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                def stop(stage):
                    if stage == "after_temp_fsync":
                        marker.write_text(stage, encoding="utf-8")
                        time.sleep(120)
                write_best_model_artifact(out, state, failpoint=stop)
            """), encoding="utf-8")
            with (root / "best_child.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    [sys.executable, str(script), str(out), str(staged),
                     str(marker), str(ROOT)], cwd=str(ROOT), stdout=stream,
                    stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline and process.poll() is None and not marker.exists():
                        time.sleep(.05)
                    self.assertTrue(marker.exists(), "best writer missed post-fsync failpoint")
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=15)
            self.assertEqual(sha256_file(out / "checkpoint_latest.pth"), original_latest_sha)
            self.assertEqual(sha256_file(out / artifact["path"]), original_best_sha)
            # A terminated writer may leave its staged temp behind; if the OS
            # removes it, the already committed content-addressed artifact must
            # still remain the only visible best model.
            visible_best = {path.name for path in out.glob("best_model_*.pth")}
            self.assertEqual(visible_best, {artifact["path"]})
            restored_model = _ToyModel()
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
            restored = restore_latest_or_previous(
                out, expected=expected, model=restored_model,
                optimizer=restored_optimizer, device="cpu")
            self.assertEqual(restored["payload"]["best_state"],
                             {"epoch": 1, "validation_loss": .75, **artifact})
            self.assertFalse(list(out.glob("*.tmp")))
            self.assertEqual(sha256_file(out / artifact["path"]), original_best_sha)

    def test_source_lock_verifies_each_file_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = root / "locked.py"
            item.write_text("value = 1\n", encoding="utf-8")
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            lock_name = "SOURCE_LOCK.v2.json"
            payload = {"schema": "g4-v1-source-lock-v2", "hash_algorithm": "SHA-256",
                       "files": [{"relative_path": "locked.py", "sha256": digest,
                                  "size_bytes": item.stat().st_size}]}
            lock_path = root / lock_name
            lock_path.write_text(json.dumps(payload), encoding="utf-8")
            lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            (root / "SOURCE_LOCK.v2.sha256").write_text(
                f"{lock_sha}  {lock_name}\n", encoding="ascii")
            self.assertTrue(verify_source_lock(root, lock_name)["verified"])
            item.write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_source_lock(root, lock_name)

    def test_formal_entry_default_is_inert_and_wrong_budget_is_rejected(self):
        with patch("sys.argv", ["train_entry.py", "--group", "G0"]):
            self.assertEqual(train_entry.main(), 0)
        with patch("sys.argv", ["train_entry.py", "--group", "G0",
                                 "--execute-formal-training", "--epoch-budget", "149",
                                 "--data-root", str(ROOT), "--run-dir", str(ROOT / "runs" / "unused"),
                                 "--runtime-lock", str(ROOT / "configs" / "unified_trainer.json")]):
            with self.assertRaises(SystemExit):
                train_entry.main()

    def test_group_config_protocol_and_g4_ss_matching_shape(self):
        configs = ROOT / "configs" / "groups"
        g4 = json.loads((configs / "G4.json").read_text(encoding="utf-8"))
        ss = json.loads((configs / "G4-SS.json").read_text(encoding="utf-8"))
        for group in ("G0", "G-E", "G-D", "G4", "G4-SS", "G4-Swap", "G4-NoCross"):
            cfg = json.loads((configs / f"{group}.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["protocol"], "../protocol.json")
        self.assertEqual(g4["element_adapter"]["bottleneck"],
                         ss["element_adapter"]["bottleneck"])
        self.assertEqual(g4["corrosion_adapter"], ss["corrosion_adapter"])
        self.assertNotEqual(g4["element_adapter"]["kind"], ss["element_adapter"]["kind"])
        for group in ("G0", "G-E", "G-D", "G4", "G4-SS", "G4-Swap", "G4-NoCross"):
            loaded = load_locked_configs(group)
            self.assertEqual(loaded[2]["group"], group)


if __name__ == "__main__":
    unittest.main(verbosity=2)
