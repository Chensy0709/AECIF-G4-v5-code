"""Shared AECIF/G4 training math, kept CPU-testable and group agnostic."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Callable, Iterable
from pathlib import Path
import sys

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_VENDOR = _ROOT / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from baseline_code.nets.hrnet_training import CE_Loss


@dataclass(frozen=True)
class DWAWeights:
    a_element: float
    a_corrosion: float
    ratio_element: float
    ratio_corrosion: float

    def as_dict(self):
        return asdict(self)


def original_dwa_weights(epoch_index: int, history: dict[str, list[float]]) -> DWAWeights:
    """Exact FP16-path DWA formula from the old fit_one_epoch.

    First two epochs use equal task weights. Later, each task's last two
    weighted train-CE epoch averages are divided directly, followed by
    exp(ratio/2), with weights normalized to sum to two. No clipping or
    stabilization is applied, so numerical behavior remains observable.
    """
    if epoch_index + 1 <= 2:
        r_e = r_c = 1.0
    else:
        e_hist = history["losses_e"]
        c_hist = history["losses_d"]
        if len(e_hist) < 2 or len(c_hist) < 2:
            raise ValueError("DWA requires two completed weighted-loss epochs")
        r_e = float(e_hist[-1]) / float(e_hist[-2])
        r_c = float(c_hist[-1]) / float(c_hist[-2])
    z_e = np.exp(r_e / 2.0)
    z_c = np.exp(r_c / 2.0)
    denominator = z_e + z_c
    a_e = 2.0 * z_e / denominator
    a_c = 2.0 * z_c / denominator
    return DWAWeights(float(a_e), float(a_c), float(r_e), float(r_c))


def get_original_cosine_schedule(peak_lr=3e-4, min_lr=3e-6, epochs=150):
    """Use the copied original helper and its default warmup/tail behavior."""
    from baseline_code.nets.hrnet_training import get_lr_scheduler
    return get_lr_scheduler("cos", peak_lr, min_lr, epochs)


def should_update_best(validation_loss: float, best_validation_loss) -> bool:
    """Keep legacy rule: lower total validation CE wins; ties select later epoch."""
    return best_validation_loss is None or validation_loss <= best_validation_loss


def _finite_gradient_summary(model: torch.nn.Module, detailed: bool = False) -> dict:
    flags = []
    names = []
    max_values = []
    l2_values = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        flags.append(torch.isfinite(grad).all())
        names.append(name)
        if detailed and grad.numel():
            max_values.append(grad.detach().abs().max().float())
            l2_values.append(torch.sum(grad.detach().float() ** 2))
    finite = True if not flags else bool(torch.stack(flags).all().item())
    nonfinite_names = []
    if not finite:
        nonfinite_names = [name for name, flag in zip(names, flags)
                           if not bool(flag.item())]
    max_abs = (float(torch.stack(max_values).max().item())
               if detailed and max_values else None)
    l2_norm = (float(torch.sqrt(torch.stack(l2_values).sum()).item())
               if detailed and l2_values else None)
    adapter = {}
    if detailed:
        for task in ("element", "corrosion"):
            module = getattr(getattr(model, "cif", None), f"{task}_adapter", None)
            if module is None:
                continue
            alpha = getattr(module, "alpha", None)
            adapter[task] = {
                "alpha": None if alpha is None else float(alpha.detach().item()),
                "alpha_grad": None if alpha is None or alpha.grad is None else
                    float(alpha.grad.detach().float().abs().item()),
                "grad_abs_sum": float(sum(
                    p.grad.detach().float().abs().sum().item()
                    for p in module.parameters() if p.grad is not None)),
            }
    return {"finite": finite, "nonfinite_parameter_names": nonfinite_names,
            "max_abs": max_abs, "l2_norm": l2_norm,
            "adapter_gradients": adapter}


def _batch_parts(batch, device):
    images, target_e, target_c = batch[:3]
    return (images.to(device=device, dtype=torch.float32, non_blocking=True),
            target_e.to(device=device, dtype=torch.long, non_blocking=True),
            target_c.to(device=device, dtype=torch.long, non_blocking=True))


def _sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def run_train_epoch(model: torch.nn.Module, loader: Iterable,
                    optimizer: torch.optim.Optimizer, *, epoch_index: int,
                    dwa_history: dict[str, list[float]], device,
                    scaler=None, fp16: bool = False,
                    collect_gradient_details: bool = False,
                    on_step: Callable[[dict], None] | None = None) -> dict:
    """One epoch using old CE/DWA aggregation and GradScaler sequencing."""
    device = torch.device(device)
    if fp16 and (device.type != "cuda" or scaler is None):
        raise ValueError("FP16 training requires CUDA and an enabled GradScaler")
    weights = original_dwa_weights(epoch_index, dwa_history)
    classes_e = torch.ones(7, device=device, dtype=torch.float32)
    classes_c = torch.ones(2, device=device, dtype=torch.float32)
    model.train()
    sums = {"total": 0.0, "element": 0.0, "corrosion": 0.0}
    step_records = []
    steps = 0
    for step, batch in enumerate(loader):
        images, target_e, target_c = _batch_parts(batch, device)
        optimizer.zero_grad()  # same default clearing semantics as old trainer
        scale_before = float(scaler.get_scale()) if fp16 else None
        start = time.perf_counter()
        if fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss_e = CE_Loss(outputs[0], target_e, classes_e, num_classes=7)
                loss_c = CE_Loss(outputs[1], target_c, classes_c, num_classes=2)
                total = weights.a_element * loss_e + weights.a_corrosion * loss_c
        else:
            outputs = model(images)
            loss_e = CE_Loss(outputs[0], target_e, classes_e, num_classes=7)
            loss_c = CE_Loss(outputs[1], target_c, classes_c, num_classes=2)
            total = weights.a_element * loss_e + weights.a_corrosion * loss_c
        forward_end = time.perf_counter()
        losses_finite = bool(torch.isfinite(total).all().item())
        if not losses_finite:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite loss at train batch {step}; update refused")
        if fp16:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
        else:
            total.backward()
        if collect_gradient_details or not fp16:
            grad_summary = _finite_gradient_summary(
                model, detailed=collect_gradient_details)
        else:
            # GradScaler's unscale/step/update path performs the authoritative
            # non-finite check. Avoid thousands of per-parameter GPU reductions
            # in every formal FP16 batch; the isolated probe does the full scan.
            grad_summary = {"finite": None, "checked_by_gradscaler": True,
                            "nonfinite_parameter_names": [], "max_abs": None,
                            "l2_norm": None, "adapter_gradients": {}}
        backward_end = time.perf_counter()
        if fp16:
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            scale_reduced = scale_after < scale_before
            if grad_summary["finite"] is None:
                grad_summary["finite"] = not scale_reduced
            skipped = scale_reduced or not grad_summary["finite"]
        else:
            optimizer.step()
            scale_after = None
            scale_reduced = False
            skipped = False
        _sync_if_cuda(device)
        end = time.perf_counter()
        record = {
            "step": step, "batch_size": int(images.shape[0]),
            "loss_element": float(loss_e.detach().float().item()),
            "loss_corrosion": float(loss_c.detach().float().item()),
            "weighted_total_loss": float(total.detach().float().item()),
            "losses_finite": losses_finite,
            "gradients": grad_summary,
            "scale_before": scale_before, "scale_after": scale_after,
            "scaler_reduced": scale_reduced,
            "optimizer_step_skipped": skipped,
            "forward_seconds": forward_end - start,
            "backward_seconds": backward_end - forward_end,
            "step_seconds": end - start,
        }
        step_records.append(record)
        if on_step is not None:
            on_step(record)
        sums["element"] += weights.a_element * record["loss_element"]
        sums["corrosion"] += weights.a_corrosion * record["loss_corrosion"]
        sums["total"] += record["weighted_total_loss"]
        steps += 1
    if steps == 0:
        raise RuntimeError("Training loader produced no batches")
    metrics = {"train_loss": sums["total"] / steps,
               "train_loss_element": sums["element"] / steps,
               "train_loss_corrosion": sums["corrosion"] / steps,
               "steps": steps, "dwa": weights.as_dict(),
               "overflow_count": sum(bool(x["scaler_reduced"]) for x in step_records),
               "optimizer_step_skipped_count": sum(
                   bool(x["optimizer_step_skipped"]) for x in step_records),
               "mean_scaler_scale": (float(np.mean([
                   x["scale_after"] for x in step_records if x["scale_after"] is not None]))
                   if fp16 else None),
               "step_records": step_records}
    # Original DWA history stores the already task-weighted epoch CE averages.
    dwa_history["losses_e"].append(metrics["train_loss_element"])
    dwa_history["losses_d"].append(metrics["train_loss_corrosion"])
    return metrics


@torch.no_grad()
def run_validation(model: torch.nn.Module, loader: Iterable, *, device,
                   dwa: DWAWeights) -> dict:
    """FP32 validation, matching old batch-mean CE and sample-weighted batches."""
    device = torch.device(device)
    model.eval()
    classes_e = torch.ones(7, device=device, dtype=torch.float32)
    classes_c = torch.ones(2, device=device, dtype=torch.float32)
    sums = {"total": 0.0, "element": 0.0, "corrosion": 0.0}
    sample_count = 0
    batches = 0
    for batch in loader:
        images, target_e, target_c = _batch_parts(batch, device)
        outputs = model(images)
        loss_e = CE_Loss(outputs[0], target_e, classes_e, num_classes=7)
        loss_c = CE_Loss(outputs[1], target_c, classes_c, num_classes=2)
        size = int(images.shape[0])
        sums["element"] += dwa.a_element * float(loss_e.item()) * size
        sums["corrosion"] += dwa.a_corrosion * float(loss_c.item()) * size
        sums["total"] += (dwa.a_element * float(loss_e.item()) +
                          dwa.a_corrosion * float(loss_c.item())) * size
        sample_count += size
        batches += 1
    if sample_count == 0:
        raise RuntimeError("Validation loader produced no samples")
    return {"val_loss": sums["total"] / sample_count,
            "val_loss_element": sums["element"] / sample_count,
            "val_loss_corrosion": sums["corrosion"] / sample_count,
            "validation_samples": sample_count, "validation_batches": batches}
