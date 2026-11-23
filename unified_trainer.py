"""One epoch-resumable trainer shared by every G4-v1 model group.

This module defines the formal run path but is not executed during the current
CPU-only preparation stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from g4_model import GROUPS, build_model, load_baseline_initialization
from trainer_core import (get_original_cosine_schedule, original_dwa_weights,
                          should_update_best,
                          run_train_epoch, run_validation)
from checkpointing import (SCHEMA_VERSION, capture_rng_state, restore_latest_or_previous,
                           restore_rng_state, save_epoch_checkpoint, sha256_file,
                           state_dict_sha256, write_best_model_artifact,
                           collect_unreferenced_best_artifacts)
from source_lock import verify_source_lock

ROOT = Path(__file__).resolve().parent
ACTIVE_SOURCE_LOCK = "SOURCE_LOCK.v5.json"
VENDOR = ROOT / "vendor"
if str(VENDOR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(VENDOR))
from baseline_code.nets.hrnet_training import set_optimizer_lr
from baseline_code.utils.dataloader_multi import SegmentationDataset


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value) -> str:
    return _sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))


class SeededDataset(Dataset):
    """Preserve baseline augmentation ops but isolate RNG by epoch and sample."""
    def __init__(self, base, ids: list[str], seed: int, train: bool):
        self.base = base
        self.ids = list(ids)
        self.seed = int(seed)
        self.epoch = 0
        self.train = bool(train)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        if not self.train:
            item = self.base[index]
            # The baseline dataset also returns two one-hot label tensors;
            # the locked trainer consumes only RGB and the two class-index masks.
            return (*item[:3], self.ids[index])
        previous = np.random.get_state()
        sample_seed = self.augmentation_seed(index)
        try:
            np.random.seed(sample_seed)
            item = self.base[index]
        finally:
            np.random.set_state(previous)
        return (*item[:3], self.ids[index], sample_seed)

    def augmentation_seed(self, index: int) -> int:
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{self.ids[index]}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little")


class EpochPermutationSampler(Sampler):
    def __init__(self, size: int, seed: int, epoch: int):
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + 1_000_003 * self.epoch) % (2**63 - 1))
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self):
        return self.size


def load_locked_configs(group: str):
    if group not in GROUPS:
        raise ValueError(f"Unsupported group: {group}")
    protocol_path = ROOT / "configs" / "protocol.json"
    trainer_path = ROOT / "configs" / "unified_trainer.json"
    group_path = ROOT / "configs" / "groups" / f"{group}.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    trainer = json.loads(trainer_path.read_text(encoding="utf-8"))
    group_config = json.loads(group_path.read_text(encoding="utf-8"))
    _validate_group_route(group, group_config)
    for split_key, count_key, sha_key in (("train_manifest", "train_count", "train_manifest_sha256"),
                                          ("validation_manifest", "validation_count", "validation_manifest_sha256")):
        manifest = ROOT / protocol["data"][split_key]
        if sha256_file(manifest) != protocol["data"][sha_key]:
            raise RuntimeError(f"Locked {split_key} SHA-256 mismatch")
        ids = [line.strip().split()[0] for line in
               manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(ids) != protocol["data"][count_key] or len(set(ids)) != len(ids):
            raise RuntimeError(f"Locked {split_key} has wrong count or duplicate IDs")
    config_sha = _canonical_sha({"protocol": protocol, "trainer": trainer,
                                 "group": group_config})
    return protocol, trainer, group_config, config_sha


def _validate_group_route(group: str, config: dict) -> None:
    dtcwt = {"kind": "dtcwt", "bottleneck": 8, "levels": 3,
             "orientations_per_level": 6, "biort": "near_sym_a",
             "qshift": "qshift_a", "mode": "symmetric"}
    spatial8 = {"kind": "spatial", "bottleneck": 8}
    spatial16 = {"kind": "spatial", "bottleneck": 16}
    routes = {
        "G0": ("identity", "identity", False),
        "G-E": (dtcwt, "identity", False),
        "G-D": ("identity", spatial16, False),
        "G4": (dtcwt, spatial16, False),
        "G4-SS": (spatial8, spatial16, False),
        "G4-Swap": (spatial16, dtcwt, False),
        "G4-NoCross": (dtcwt, spatial16, True),
    }
    expected = routes[group]
    observed = (config.get("element_adapter"), config.get("corrosion_adapter"),
                config.get("no_cross", False))
    if (config.get("group") != group or
            config.get("protocol") != "../protocol.json" or observed != expected):
        raise RuntimeError(f"Group config does not match model-factory route: {group}")


def runtime_fingerprint(device) -> dict:
    device = torch.device(device)
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def validate_runtime_lock(lock: dict, device) -> dict:
    actual = runtime_fingerprint(device)
    keys = ("python", "torch", "cuda", "gpu")
    if not isinstance(lock, dict) or any(not lock.get(key) for key in keys):
        raise RuntimeError("A reviewed runtime lock with python/torch/cuda/gpu is required")
    mismatches = {key: {"expected": lock[key], "actual": actual[key]}
                  for key in keys if str(lock[key]) != str(actual[key])}
    if mismatches:
        raise RuntimeError(f"Runtime lock mismatch: {mismatches}")
    return actual


def _read_ids(path: Path):
    return [line.strip().split()[0] for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _atomic_json(path: Path, data):
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _write_metrics_csv(output_dir: Path, rows: list[dict]):
    from checkpointing import _rewrite_history_csv
    _rewrite_history_csv(output_dir / "epoch_metrics.csv", rows)


def _write_batch_jsonl(output_dir: Path, rows: list[dict]):
    import tempfile
    path = output_dir / "batch_diagnostics.jsonl"
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                epoch = row["epoch"]
                for batch in row.get("batch_records", []):
                    stream.write(json.dumps({"epoch": epoch, **batch}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class UnifiedTrainer:
    def __init__(self, group: str, data_root: Path, run_dir: Path, *,
                 runtime_lock: dict, device="cuda:0", resume=False,
                 run_scope="formal"):
        self.group = group
        self.data_root = Path(data_root)
        if run_scope not in ("formal", "diagnostic"):
            raise ValueError("run_scope must be formal or diagnostic")
        self.run_scope = run_scope
        runs_root = ((ROOT / "runs") if run_scope == "formal" else
                     (ROOT / "diagnostics" / "runs")).resolve()
        self.run_dir = Path(run_dir).resolve()
        if not self.run_dir.is_relative_to(runs_root):
            raise RuntimeError(f"{run_scope} output must be beneath {runs_root}")
        if resume:
            if not self.run_dir.is_dir() or not any(self.run_dir.iterdir()):
                raise FileNotFoundError("Epoch-boundary resume requires an existing run directory")
        elif self.run_dir.exists():
            raise FileExistsError("A formal run must use a never-used output directory")
        self.source_lock_audit = verify_source_lock(ROOT, ACTIVE_SOURCE_LOCK)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("The formal protocol is FP16+GradScaler on CUDA; CPU is test-only")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing to substitute another precision")
        self.runtime = validate_runtime_lock(runtime_lock, self.device)
        self.protocol, self.trainer_config, self.group_config, self.config_sha = load_locked_configs(group)
        if self.protocol["training"]["precision"] != "FP16 autocast + torch.cuda.amp.GradScaler (legacy protocol candidate)":
            raise RuntimeError("Shared protocol precision changed")
        self.init_meta = self.protocol["initialization"]
        if sha256_file(Path(self.init_meta["path"])) != self.init_meta["sha256"]:
            raise RuntimeError("Initialization checkpoint hash mismatch")
        self.source_sha = self.source_lock_audit["lock_sha256"]
        self.protocol_sha = sha256_file(ROOT / "configs" / "protocol.json")
        self.expected_checkpoint = {
            "trainer_id": self.trainer_config["trainer_id"],
            "group": group, "precision_mode": "fp16_grad_scaler",
            "protocol_sha256": self.protocol_sha,
            "config_sha256": self.config_sha,
            "source_manifest_sha256": self.source_sha,
            "initialization_sha256": self.init_meta["sha256"],
            "runtime_sha256": _canonical_sha(self.runtime),
        }
        seed = int(self.trainer_config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_model(group, num_classes=tuple(self.protocol["model"]["num_classes"]),
                                 backbone=self.protocol["model"]["backbone"],
                                 pretrained=False, init_seed=self.protocol["training"]["seed"])
        init_audit = load_baseline_initialization(
            self.model, self.init_meta["path"], self.init_meta["sha256"], 1830)
        self.initial_model_state_sha256 = state_dict_sha256(self.model.state_dict())
        self.init_audit = init_audit
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.protocol["training"]["learning_rate_peak"],
            betas=tuple(self.protocol["training"]["adam_betas"]),
            weight_decay=self.protocol["training"]["weight_decay"])
        self.scaler = torch.cuda.amp.GradScaler()
        self.lr_schedule = get_original_cosine_schedule(
            self.protocol["training"]["learning_rate_peak"],
            self.protocol["training"]["learning_rate_minimum"],
            self.protocol["training"]["epochs"])
        self.dwa_history = {"losses_e": [], "losses_d": []}
        self.best_state = {"epoch": None, "validation_loss": None,
                           "path": None, "sha256": None}
        self.metrics_history = []
        self.completed_epoch = -1
        self.start_epoch = 0
        resume_rng_state = None
        if resume:
            restored = restore_latest_or_previous(
                self.run_dir, expected=self.expected_checkpoint,
                model=self.model, optimizer=self.optimizer, scaler=self.scaler,
                device=self.device, source_lock_root=ROOT,
                source_lock_name=ACTIVE_SOURCE_LOCK)
            payload = restored["payload"]
            self.completed_epoch = payload["completed_epoch"]
            self.start_epoch = self.completed_epoch + 1
            self.dwa_history = payload["dwa_state"]
            self.best_state = payload["best_state"]
            self.metrics_history = payload["metrics_history"]
            self.sampler_state = payload["sampler_state"]
            resume_rng_state = payload["rng_state"]
            self.resume_audit = {"restored_from": restored["restored_from"],
                                 "fallback_errors": restored["fallback_errors"]}
            _write_batch_jsonl(self.run_dir, self.metrics_history)
        else:
            self.sampler_state = {"policy": self.trainer_config["train_sampler"],
                                  "seed": self.trainer_config["seed"],
                                  "augmentation_rng": self.trainer_config["augmentation_rng"],
                                  "last_epoch_order_sha256": None}
            self.resume_audit = {"restored_from": None, "fallback_errors": []}
        self._prepare_datasets()
        # Restore after all constructor/setup work so it cannot consume the
        # checkpoint RNG stream before the first resumed DataLoader iterator.
        if resume_rng_state is not None:
            restore_rng_state(resume_rng_state, self.device)

    def _prepare_datasets(self):
        train_manifest = ROOT / self.protocol["data"]["train_manifest"]
        val_manifest = ROOT / self.protocol["data"]["validation_manifest"]
        train_ids = _read_ids(train_manifest)
        val_ids = _read_ids(val_manifest)
        if set(train_ids) & set(val_ids):
            raise RuntimeError("Train/validation manifest overlap")
        for ids, branch in ((train_ids, "train"), (val_ids, "val")):
            annotation = ids
            original = SegmentationDataset(
                annotation_lines=annotation,
                input_shape=self.protocol["model"]["input_size"],
                num_classes=self.protocol["model"]["num_classes"],
                train=(branch == "train"), dataset_path=str(self.data_root))
            wrapped = SeededDataset(original, ids,
                                    self.trainer_config["seed"], branch == "train")
            setattr(self, f"{branch}_dataset", wrapped)
        self.train_ids = train_ids
        self.val_ids = val_ids

    def _loaders(self, epoch):
        self.train_dataset.set_epoch(epoch)
        sampler = EpochPermutationSampler(len(self.train_dataset),
                                          self.trainer_config["seed"], epoch)
        order = list(iter(sampler))
        order_sha = _sha_bytes("\n".join(self.train_ids[i] for i in order).encode("utf-8"))
        self.sampler_state["last_epoch_order_sha256"] = order_sha
        self.sampler_state["last_epoch"] = epoch
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.protocol["training"]["batch_size"],
            sampler=order, shuffle=False, num_workers=0, pin_memory=True,
            drop_last=True)
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.protocol["training"]["batch_size"],
            shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        return train_loader, val_loader

    def _checkpoint_payload(self, completed_epoch, scheduler_state, row):
        return {
            "schema_version": SCHEMA_VERSION,
            **self.expected_checkpoint,
            "completed_epoch": int(completed_epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state": scheduler_state,
            "dwa_state": self.dwa_history,
            "rng_state": capture_rng_state(self.device),
            "sampler_state": dict(self.sampler_state),
            "best_state": dict(self.best_state),
            "metrics_history": list(self.metrics_history),
            "runtime": dict(self.runtime),
            "training_mode": bool(self.model.training),
            "last_epoch_metrics": dict(row),
        }

    def fit(self):
        if self.run_scope != "formal":
            raise RuntimeError("Diagnostic trainer cannot enter the formal 150-Epoch loop")
        total_epochs = int(self.protocol["training"]["epochs"])
        return self._fit_until_epoch(total_epochs)

    def fit_diagnostic(self, epoch_limit: int):
        if self.run_scope != "diagnostic":
            raise RuntimeError("Short diagnostic budget is unavailable to formal runs")
        if epoch_limit not in (1, 2):
            raise ValueError("Diagnostic epoch limit must be exactly 1 or 2")
        if epoch_limit > int(self.protocol["training"]["epochs"]):
            raise ValueError("Diagnostic limit exceeds the locked total-epoch protocol")
        return self._fit_until_epoch(epoch_limit)

    def _fit_until_epoch(self, epoch_limit: int):
        total_epochs = int(self.protocol["training"]["epochs"])
        if not 0 <= epoch_limit <= total_epochs:
            raise ValueError("Epoch limit must remain within the locked 150-Epoch protocol")
        if self.start_epoch > epoch_limit:
            raise ValueError("Checkpoint is already beyond the requested diagnostic epoch limit")
        for epoch in range(self.start_epoch, epoch_limit):
            epoch_start = time.perf_counter()
            set_optimizer_lr(self.optimizer, self.lr_schedule, epoch)
            lr = float(self.optimizer.param_groups[0]["lr"])
            train_loader, val_loader = self._loaders(epoch)
            train = run_train_epoch(
                self.model, train_loader, self.optimizer, epoch_index=epoch,
                dwa_history=self.dwa_history, device=self.device,
                scaler=self.scaler, fp16=True,
                audit_input_hashes=(self.run_scope == "diagnostic"))
            # Training uses weights computed at epoch entry; validation must
            # use that exact same pair, as in the legacy fit_one_epoch.
            dwa_values = train["dwa"]
            from trainer_core import DWAWeights
            dwa = DWAWeights(**dwa_values)
            val = run_validation(self.model, val_loader, device=self.device, dwa=dwa)
            epoch_seconds = time.perf_counter() - epoch_start
            row = {"epoch": epoch + 1, "train_loss": train["train_loss"],
                   "train_loss_element": train["train_loss_element"],
                   "train_loss_corrosion": train["train_loss_corrosion"],
                   **val, "dwa_element": dwa.a_element,
                   "dwa_corrosion": dwa.a_corrosion, "learning_rate": lr,
                   "overflow_count": train["overflow_count"],
                   "optimizer_step_skipped_count": train["optimizer_step_skipped_count"],
                   "mean_scaler_scale": train["mean_scaler_scale"],
                   "sampler_order_sha256": self.sampler_state["last_epoch_order_sha256"],
                   "epoch_seconds": epoch_seconds,
                   "batch_records": train["step_records"]}
            if not math_isfinite(row["val_loss"]):
                raise FloatingPointError(f"Non-finite validation loss at epoch {epoch + 1}")
            previous_best_state = dict(self.best_state)
            if should_update_best(row["val_loss"],
                                  self.best_state["validation_loss"]):
                best_artifact = write_best_model_artifact(
                    self.run_dir, self.model.state_dict())
                self.best_state = {"epoch": epoch + 1,
                                   "validation_loss": row["val_loss"],
                                   **best_artifact}
            self.metrics_history.append(row)
            scheduler_state = {
                "type": "cos", "total_epochs": total_epochs,
                "completed_epoch": epoch + 1,
                "initial_lr": self.protocol["training"]["learning_rate_peak"],
                "min_lr": self.protocol["training"]["learning_rate_minimum"],
                "last_lr": [lr],
                "schedule_source": "vendor/baseline_code/nets/hrnet_training.py",
            }
            payload = self._checkpoint_payload(epoch, scheduler_state, row)
            save_epoch_checkpoint(self.run_dir, payload)
            collect_unreferenced_best_artifacts(self.run_dir,
                                                [self.best_state, previous_best_state])
            self.completed_epoch = epoch
            _write_metrics_csv(self.run_dir, self.metrics_history)
            _write_batch_jsonl(self.run_dir, self.metrics_history)
        return self.metrics_history


def math_isfinite(value):
    return bool(np.isfinite(float(value)))
