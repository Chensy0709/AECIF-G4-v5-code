"""Epoch-boundary checkpointing for the isolated G4-v1 trainer."""
from __future__ import annotations

import csv
import hashlib
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = 2
REQUIRED = {
    "schema_version", "trainer_id", "group", "precision_mode",
    "protocol_sha256", "config_sha256", "source_manifest_sha256",
    "initialization_sha256", "completed_epoch", "model_state_dict",
    "optimizer_state_dict", "scaler_state_dict", "scheduler_state",
    "dwa_state", "rng_state", "sampler_state", "best_state",
    "metrics_history", "runtime", "training_mode",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Non-Tensor in model state: {key}")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_best_model_artifact(output_dir: Path, state: dict[str, torch.Tensor],
                              failpoint=None) -> dict:
    """Write immutable content-addressed best weights in this run only."""
    output_dir = Path(output_dir)
    state_sha = state_dict_sha256(state)
    name = f"best_model_{state_sha}.pth"
    path = output_dir / name
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if state_dict_sha256(saved) != state_sha:
            raise RuntimeError("Existing content-addressed best artifact is corrupt")
    else:
        cpu_state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
        atomic_torch_save(cpu_state, path, failpoint=failpoint)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if state_dict_sha256(saved) != state_sha:
            raise RuntimeError("New best artifact failed tensor verification")
    return {"path": name, "sha256": sha256_file(path),
            "state_sha256": state_sha}


def collect_unreferenced_best_artifacts(output_dir: Path, keep_states) -> list[str]:
    """Remove only this trainer's unreferenced content-addressed files."""
    keep = {Path(item["path"]).name for item in keep_states
            if isinstance(item, dict) and item.get("path")}
    removed = []
    for path in Path(output_dir).glob("best_model_*.pth"):
        if path.name not in keep:
            path.unlink()
            removed.append(path.name)
    return removed


def atomic_torch_save(payload, destination: Path, failpoint=None) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".",
                                     suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if failpoint is not None:
            failpoint("after_temp_fsync")
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".",
                                     suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as target, Path(source).open("rb") as src:
            shutil.copyfileobj(src, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def capture_rng_state(device) -> dict:
    device = torch.device(device)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all()
            if device.type == "cuda" and torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict, device) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Checkpoint RNG state is incomplete")
    def as_cpu_byte_state(value, name):
        if not torch.is_tensor(value):
            raise TypeError(f"Checkpoint {name} RNG state must be a Tensor")
        tensor = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        if tensor.ndim != 1 or tensor.numel() == 0:
            raise ValueError(f"Checkpoint {name} RNG state must be a non-empty 1D ByteTensor")
        return tensor

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(as_cpu_byte_state(state["torch_cpu"], "CPU"))
    device = torch.device(device)
    if state["torch_cuda_all"] is not None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG but current device is not CUDA")
        cuda_states = state["torch_cuda_all"]
        if not isinstance(cuda_states, (tuple, list)) or not cuda_states:
            raise ValueError("Checkpoint CUDA RNG state must be a non-empty sequence")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Checkpoint CUDA RNG device count differs from the current runtime: "
                f"saved={len(cuda_states)}, current={torch.cuda.device_count()}")
        torch.cuda.set_rng_state_all([
            as_cpu_byte_state(value, f"CUDA[{index}]")
            for index, value in enumerate(cuda_states)])


def validate_checkpoint(payload: dict, expected: dict, model: torch.nn.Module,
                        scaler=None) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint root must be a mapping")
    missing = sorted(REQUIRED - set(payload))
    if missing:
        raise ValueError(f"Missing checkpoint fields: {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Checkpoint schema version mismatch")
    for key in ("trainer_id", "group", "precision_mode", "protocol_sha256",
                "config_sha256", "source_manifest_sha256", "initialization_sha256",
                "runtime_sha256"):
        if payload[key] != expected[key]:
            raise ValueError(f"Checkpoint {key} mismatch")
    if not isinstance(payload["completed_epoch"], int) or payload["completed_epoch"] < 0:
        raise ValueError("Checkpoint is not a completed-epoch checkpoint")
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state",
                "dwa_state", "rng_state", "sampler_state", "best_state"):
        if payload[key] is None:
            raise ValueError(f"Checkpoint field {key} is null")
    if not isinstance(payload["scheduler_state"], dict) or not payload["scheduler_state"]:
        raise ValueError("Scheduler checkpoint state is incomplete")
    scheduler_state = payload["scheduler_state"]
    if (scheduler_state.get("type") != "cos" or
            scheduler_state.get("completed_epoch") != payload["completed_epoch"] + 1 or
            not isinstance(scheduler_state.get("last_lr"), list)):
        raise ValueError("Scheduler state does not match the committed Epoch")
    optimizer_state = payload["optimizer_state_dict"]
    if (not isinstance(optimizer_state, dict) or
            not isinstance(optimizer_state.get("param_groups"), list) or
            not isinstance(optimizer_state.get("state"), dict)):
        raise ValueError("Optimizer checkpoint state is incomplete")
    optimizer_lrs = [float(group["lr"]) for group in optimizer_state["param_groups"]]
    scheduler_lrs = [float(value) for value in scheduler_state["last_lr"]]
    if optimizer_lrs != scheduler_lrs:
        raise ValueError("Scheduler last_lr differs from optimizer checkpoint state")
    dwa = payload["dwa_state"]
    if (not isinstance(dwa, dict) or
            not isinstance(dwa.get("losses_e"), list) or
            not isinstance(dwa.get("losses_d"), list) or
            len(dwa["losses_e"]) != payload["completed_epoch"] + 1 or
            len(dwa["losses_d"]) != payload["completed_epoch"] + 1):
        raise ValueError("DWA history does not match completed Epoch count")
    history = payload["metrics_history"]
    if (not isinstance(history, list) or
            len(history) != payload["completed_epoch"] + 1 or
            [row.get("epoch") for row in history] !=
            list(range(1, payload["completed_epoch"] + 2))):
        raise ValueError("Metric history is incomplete or non-contiguous")
    best = payload["best_state"]
    if not isinstance(best, dict):
        raise ValueError("Best state is not a mapping")
    if best.get("epoch") is None or not (1 <= best["epoch"] <= payload["completed_epoch"] + 1):
        raise ValueError("Best epoch is outside the committed history")
    best_rows = [row for row in history if row.get("epoch") == best["epoch"]]
    if len(best_rows) != 1 or best_rows[0].get("val_loss") != best.get("validation_loss"):
        raise ValueError("Best state does not match its committed validation row")
    if (not best.get("path") or not best.get("sha256") or
            not best.get("state_sha256") or
            Path(best["path"]).name != best["path"] or
            best["path"] != f"best_model_{best['state_sha256']}.pth"):
        raise ValueError("Best artifact identity is incomplete")
    live = model.state_dict()
    saved = payload["model_state_dict"]
    if set(live) != set(saved):
        raise ValueError("Checkpoint model key set differs from current group")
    for key, tensor in live.items():
        if tuple(tensor.shape) != tuple(saved[key].shape):
            raise ValueError(f"Checkpoint shape mismatch at {key}")
    if payload["precision_mode"] == "fp16_grad_scaler":
        if scaler is None or payload["scaler_state_dict"] is None:
            raise ValueError("FP16 checkpoint lacks a GradScaler state")
    elif payload["scaler_state_dict"] is not None:
        raise ValueError("Non-FP16 checkpoint unexpectedly contains GradScaler state")


def save_epoch_checkpoint(output_dir: Path, payload: dict, failpoint=None) -> dict:
    output_dir = Path(output_dir)
    latest = output_dir / "checkpoint_latest.pth"
    previous = output_dir / "checkpoint_previous.pth"
    had_latest = latest.exists()
    if had_latest:
        _atomic_copy(latest, previous)
    if failpoint is not None:
        failpoint("after_previous_copy")
    atomic_torch_save(payload, latest, failpoint=failpoint)
    if not had_latest:
        # Seed the fallback slot after the first committed epoch.
        _atomic_copy(latest, previous)
    return {"latest": str(latest), "sha256": sha256_file(latest),
            "previous": str(previous) if previous.exists() else None}


def _load_payload(path: Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def _best_reference_from_checkpoint(path: Path, output_dir: Path):
    """Read checkpoint metadata via mmap so recovery GC preserves both fallbacks."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        best = payload.get("best_state")
        if not isinstance(best, dict):
            return None
        name = best.get("path")
        if (not isinstance(name, str) or Path(name).name != name or
                name != f"best_model_{best.get('state_sha256')}.pth"):
            return None
        artifact = Path(output_dir) / name
        if not artifact.is_file() or sha256_file(artifact) != best.get("sha256"):
            return None
        return best
    except Exception:
        return None


def restore_latest_or_previous(output_dir: Path, *, expected: dict,
                               model: torch.nn.Module,
                               optimizer: torch.optim.Optimizer,
                               scaler=None, device="cpu",
                               source_lock_root: Path | None = None,
                               source_lock_name: str | None = None) -> dict:
    """Validate candidates before loading; a failed latest tries previous."""
    if (source_lock_root is None) != (source_lock_name is None):
        raise ValueError("Source lock root and name must be provided together")
    if source_lock_root is not None:
        from source_lock import verify_source_lock
        lock = verify_source_lock(source_lock_root, source_lock_name)
        if lock["lock_sha256"] != expected.get("source_manifest_sha256"):
            raise RuntimeError("Active source lock differs from checkpoint identity")
    output_dir = Path(output_dir)
    errors = []
    for name in ("checkpoint_latest.pth", "checkpoint_previous.pth"):
        candidate = output_dir / name
        if not candidate.is_file():
            errors.append({"candidate": name, "error": "missing"})
            continue
        try:
            payload = _load_payload(candidate, map_location=device)
            validate_checkpoint(payload, expected, model, scaler)
            best = payload["best_state"]
            if best.get("path"):
                best_path = output_dir / best["path"]
                if not best_path.is_file() or sha256_file(best_path) != best.get("sha256"):
                    raise ValueError("Best-model artifact is missing or has a SHA-256 mismatch")
                best_tensors = torch.load(best_path, map_location="cpu", weights_only=True)
                if state_dict_sha256(best_tensors) != best["state_sha256"]:
                    raise ValueError("Best-model tensor digest differs from checkpoint metadata")
                if best["epoch"] == payload["completed_epoch"] + 1 and \
                        state_dict_sha256(payload["model_state_dict"]) != best["state_sha256"]:
                    raise ValueError("Current-epoch best weights differ from checkpoint model")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            if scaler is not None:
                scaler.load_state_dict(payload["scaler_state_dict"])
            restore_rng_state(payload["rng_state"], device)
            restored_from_previous = name == "checkpoint_previous.pth"
            if restored_from_previous:
                # Promote the validated candidate before the next save rotates
                # latest into previous; otherwise a corrupt latest could erase
                # the only valid fallback on the following interrupted save.
                _atomic_copy(candidate, output_dir / "checkpoint_latest.pth")
                keep_best_refs = [best]
            else:
                previous_best = _best_reference_from_checkpoint(
                    output_dir / "checkpoint_previous.pth", output_dir)
                keep_best_refs = [best] + ([previous_best] if previous_best else [])
            collect_unreferenced_best_artifacts(output_dir, keep_best_refs)
            _rewrite_history_csv(output_dir / "epoch_metrics.csv",
                                 payload["metrics_history"])
            _rewrite_batch_jsonl(output_dir / "batch_diagnostics.jsonl",
                                 payload["metrics_history"])
            _cleanup_temporary_files(output_dir)
            return {"payload": payload, "restored_from": str(candidate),
                    "fallback_errors": errors}
        except Exception as exc:
            errors.append({"candidate": name, "error": f"{type(exc).__name__}: {exc}"})
    raise RuntimeError(f"No valid G4-v1 checkpoint candidate: {errors}")


def _cleanup_temporary_files(output_dir: Path) -> list[str]:
    removed = []
    for path in Path(output_dir).glob("*.tmp"):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def _rewrite_history_csv(path: Path, rows: list[dict]) -> None:
    fields = ["epoch", "train_loss", "train_loss_element", "train_loss_corrosion",
              "val_loss", "val_loss_element", "val_loss_corrosion", "dwa_element",
              "dwa_corrosion", "learning_rate", "overflow_count",
              "optimizer_step_skipped_count", "epoch_seconds"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _rewrite_batch_jsonl(path: Path, rows: list[dict]) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                for batch in row.get("batch_records", []):
                    stream.write(json.dumps({"epoch": row["epoch"], **batch}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
