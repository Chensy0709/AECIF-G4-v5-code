"""Prepared paired G0/G4 FP16 diagnostic. It is inert unless explicitly gated."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))
from g4_model import build_model, load_baseline_initialization
from source_lock import verify_source_lock
from baseline_code.nets.hrnet_training import CE_Loss
from baseline_code.utils.dataloader_multi import SegmentationDataset


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "diagnostics" / "fp16_probe")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Required with --execute-gpu-diagnostic; maximum 1..3 paired train batches")
    parser.add_argument("--execute-gpu-diagnostic", action="store_true",
                        help="Explicitly enables training-split image reads and CUDA")
    return parser.parse_args()


def validate_runtime(lock_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    keys = ("python", "torch", "cuda", "gpu")
    if any(not lock.get(key) for key in keys):
        raise RuntimeError("Runtime lock must specify python/torch/cuda/gpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing precision substitution")
    actual = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    mismatch = {k: {"expected": lock[k], "actual": actual[k]}
                for k in keys if str(lock[k]) != str(actual[k])}
    if mismatch:
        raise RuntimeError(f"Runtime lock mismatch: {mismatch}")
    return {"expected": lock, "actual": actual}


def _model_and_optimizer(group, protocol, device):
    init = protocol["initialization"]
    model = build_model(group, init_seed=protocol["training"]["seed"])
    init_audit = load_baseline_initialization(model, init["path"], init["sha256"], 1830)
    model.to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=protocol["training"]["learning_rate_peak"],
        betas=tuple(protocol["training"]["adam_betas"]),
        weight_decay=protocol["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()
    return model, optimizer, scaler, init_audit


def _adapter_gradients(model):
    result = {}
    def safe_scalar(tensor):
        value = float(tensor.detach().float().cpu())
        return value if math.isfinite(value) else None
    for name in ("element", "corrosion"):
        adapter = getattr(model.cif, f"{name}_adapter", None)
        if adapter is None or not hasattr(adapter, "alpha"):
            continue
        result[name] = {
            "alpha": float(adapter.alpha.detach().float().cpu()),
            "alpha_grad": (None if adapter.alpha.grad is None else
                           safe_scalar(adapter.alpha.grad)),
            "alpha_grad_finite": (None if adapter.alpha.grad is None else
                                  bool(torch.isfinite(adapter.alpha.grad).all().item())),
            "adapter_grad_abs_sum": safe_scalar(sum(
                (p.grad.detach().float().abs().sum()
                 for p in adapter.parameters() if p.grad is not None),
                torch.zeros((), device=adapter.alpha.device))),
            "adapter_grad_nonzero_count": int(sum(
                torch.count_nonzero(p.grad).item()
                for p in adapter.parameters() if p.grad is not None)),
        }
    return result


def _probe_batch_to_device(batch, device):
    """Cast images to FP32 while preserving dataset label tensor dtypes."""
    images, target_e, target_c = batch[:3]
    return (
        images.to(device=device, dtype=torch.float32, non_blocking=True),
        target_e.to(device=device, non_blocking=True),
        target_c.to(device=device, non_blocking=True),
    )


def _detach_task_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise TypeError("AECIF-Net probe expects (Element, Corrosion) outputs")
    return [output.detach() for output in outputs]


def _step(model, optimizer, scaler, batch, batch_index):
    images, target_e, target_c = _probe_batch_to_device(batch, "cuda:0")
    optimizer.zero_grad()
    before_versions = [p._version for p in model.parameters()]
    scale_before = float(scaler.get_scale())
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output_e, output_c = model(images)
        loss_e = CE_Loss(output_e, target_e, torch.ones(7, device="cuda:0"), num_classes=7)
        loss_c = CE_Loss(output_c, target_c, torch.ones(2, device="cuda:0"), num_classes=2)
        total = loss_e + loss_c  # DWA is exactly 1/1 in the first two epochs.
    finite_loss = bool(torch.isfinite(total).all().item())
    if not finite_loss:
        return {"batch_index": batch_index, "losses_finite": False,
                "scale_before": scale_before, "update_attempted": False}
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    nonfinite = [name for name, p in model.named_parameters()
                 if p.grad is not None and not bool(torch.isfinite(p.grad).all().item())]
    adapter_gradients = _adapter_gradients(model)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(0)
    scale_after = float(scaler.get_scale())
    changed = any(version != p._version
                  for version, p in zip(before_versions, model.parameters()))
    return {
        "batch_index": batch_index,
        "losses_finite": finite_loss,
        "element_loss": float(loss_e.detach().float().cpu()),
        "corrosion_loss": float(loss_c.detach().float().cpu()),
        "total_loss": float(total.detach().float().cpu()),
        "output_shapes": [list(output_e.shape), list(output_c.shape)],
        "all_gradients_finite": not nonfinite,
        "nonfinite_parameter_names": nonfinite[:30],
        "scale_before": scale_before,
        "scale_after": scale_after,
        "scale_reduced_overflow_signal": scale_after < scale_before,
        "optimizer_step_skipped": scale_after < scale_before,
        "grad_scaler_unscale_called": True,
        "grad_scaler_step_called": True,
        "grad_scaler_update_called": True,
        "parameters_changed": changed,
        "adapter_gradients": adapter_gradients,
        "elapsed_seconds": time.perf_counter() - started,
        "torch_cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
        "torch_cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(0),
    }


def main():
    args = parse_args()
    if not args.execute_gpu_diagnostic:
        print(json.dumps({"status": "PREPARED_NOT_RUN", "groups": ["G0", "G4"],
                          "precision": "FP16+GradScaler", "training_image_access": False,
                          "cuda_access": False,
                          "next_gate": "Requires separate user authorization and reviewed runtime lock"},
                         ensure_ascii=False, indent=2))
        return 0
    if args.max_batches is None or not 1 <= args.max_batches <= 3:
        raise ValueError("Explicit --max-batches 1..3 is required for the GPU probe")
    lock_audit = verify_source_lock(ROOT, "SOURCE_LOCK.v5.json")
    if args.runtime_lock is None:
        raise RuntimeError("--runtime-lock is required")
    runtime = validate_runtime(args.runtime_lock)
    protocol_path = ROOT / "configs" / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seed = int(protocol["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    manifest = ROOT / protocol["data"]["train_manifest"]
    if sha256(manifest) != protocol["data"]["train_manifest_sha256"]:
        raise RuntimeError("Training manifest identity changed")
    ids = [line.strip().split()[0] for line in
           manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(ids) != 200 or len(set(ids)) != 200:
        raise RuntimeError("Expected exactly 200 unique training IDs")
    init = protocol["initialization"]
    if sha256(Path(init["path"])) != init["sha256"]:
        raise RuntimeError("Initialization checkpoint SHA-256 changed")
    dataset = SegmentationDataset(
        annotation_lines=ids, input_shape=protocol["model"]["input_size"],
        num_classes=protocol["model"]["num_classes"], train=True,
        dataset_path=str(args.data_root or protocol["data"]["data_root"]))
    loader = DataLoader(dataset, batch_size=protocol["training"]["batch_size"],
                        shuffle=False, num_workers=0, drop_last=False)
    # Materialize once so G0 and G4 consume byte-identical augmented tensors.
    batches = []
    batch_ids = []
    for index, batch in enumerate(loader):
        if index >= args.max_batches:
            break
        batches.append(tuple(t.clone() for t in batch))
        batch_ids.append(ids[index * 4:index * 4 + int(batch[0].shape[0])])
    if not batches:
        raise RuntimeError("No probe batch was materialized from training split")

    comparison_batch = _probe_batch_to_device(batches[0], "cuda:0")
    records = {"G0": [], "G4": []}
    init_audits = {}
    initial_outputs_cpu = {}
    initial_diff = None
    scaler_roundtrip = {}
    for group in ("G0", "G4"):
        model, optimizer, scaler, init_audit = _model_and_optimizer(group, protocol, "cuda:0")
        init_audits[group] = init_audit
        model.eval()
        with torch.no_grad():
            initial_outputs = model(comparison_batch[0])
            initial = _detach_task_outputs(initial_outputs)
            if group == "G0":
                initial_outputs_cpu[group] = [x.float().cpu() for x in initial]
            else:
                initial_diff = [float((a.float().cpu() - b).abs().max())
                                for a, b in zip(initial, initial_outputs_cpu["G0"])]
        model.train()
        for batch_index, batch in enumerate(batches):
            row = _step(model, optimizer, scaler, batch, batch_index)
            row["sample_ids"] = batch_ids[batch_index]
            records[group].append(row)
            if not row["losses_finite"] or not row.get("all_gradients_finite", False):
                break
        scaler_state = scaler.state_dict()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scaler_state.pth"
            torch.save(scaler_state, path)
            restored = torch.cuda.amp.GradScaler()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            scaler_roundtrip[group] = restored.state_dict() == scaler_state
        del model, optimizer, scaler
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to((ROOT / "diagnostics").resolve()):
        raise RuntimeError("FP16 probe output must remain beneath diagnostics/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"paired_G0_G4_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report = {
        "status": "ISOLATED_DIAGNOSTIC_ONLY", "formal_training": False,
        "training_split_only": True, "validation_test_access": False,
        "groups": ["G0", "G4"], "precision": "FP16+GradScaler",
        "dwa_weights": {"element": 1.0, "corrosion": 1.0,
                        "reason": "first epoch legacy DWA"},
        "runtime": runtime, "protocol_sha256": sha256(protocol_path),
        "source_lock": lock_audit,
        "train_manifest_sha256": sha256(manifest), "initialization": init_audits,
        "same_input_batch_ids": batch_ids,
        "initial_eval_max_abs_output_difference_G0_vs_G4": initial_diff,
        "batches": records, "scaler_state_roundtrip": scaler_roundtrip,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if output.exists():
        raise FileExistsError(output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
