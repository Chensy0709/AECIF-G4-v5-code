"""A10 continuity-vs-resume experiment. Inert unless explicitly authorized."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from checkpointing import sha256_file, state_dict_sha256
from checkpointing import REQUIRED as CHECKPOINT_REQUIRED, SCHEMA_VERSION
from source_lock import verify_source_lock


def _command(group, data_root, runtime_lock, run_dir, limit, resume):
    cmd = [sys.executable, str(ROOT / "diagnostic_entry.py"), "--group", group,
           "--data-root", str(data_root), "--runtime-lock", str(runtime_lock),
           "--run-dir", str(run_dir), "--epoch-limit", str(limit),
           "--execute-diagnostic"]
    if resume:
        cmd.append("--resume")
    child_phrase = f"APPROVE_G4_V5_{group}_DIAGNOSTIC_UP_TO_EPOCH_{limit}\n"
    return cmd, child_phrase


def _run_child(label, command, phrase, log_dir):
    started = time.time()
    result = subprocess.run(command, input=phrase, text=True, capture_output=True,
                             cwd=ROOT, check=False)
    ended = time.time()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{label}.stderr.log").write_text(result.stderr, encoding="utf-8")
    record = {"label": label, "command": command, "returncode": result.returncode,
              "started_unix": started, "ended_unix": ended,
              "elapsed_seconds": ended - started,
              "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode != 0:
        raise RuntimeError(f"Child {label} failed; see {log_dir}")
    return record


def _compare_tensor_states(left, right):
    left_keys, right_keys = set(left), set(right)
    if left_keys != right_keys:
        return {"keys_match": False, "left_only": sorted(left_keys-right_keys),
                "right_only": sorted(right_keys-left_keys), "tensor_count": 0,
                "exact": False, "max_abs": None, "relative_l2": None}
    max_abs, sum_diff2, sum_ref2, exact, count = 0.0, 0.0, 0.0, True, 0
    for key in sorted(left_keys):
        a, b = left[key], right[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            if a != b:
                exact = False
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            return {"keys_match": True, "shape_or_dtype_mismatch": key,
                    "tensor_count": count, "exact": False,
                    "max_abs": None, "relative_l2": None}
        af, bf = a.detach().cpu().double(), b.detach().cpu().double()
        delta = af - bf
        local = float(delta.abs().max()) if delta.numel() else 0.0
        max_abs = max(max_abs, local)
        sum_diff2 += float((delta * delta).sum())
        sum_ref2 += float((af * af).sum())
        exact = exact and torch.equal(a.detach().cpu(), b.detach().cpu())
        count += 1
    return {"keys_match": True, "tensor_count": count, "exact": exact,
            "max_abs": max_abs,
            "relative_l2": math.sqrt(sum_diff2) / max(math.sqrt(sum_ref2), 1e-30)}


def _flatten_tensors(value, prefix=""):
    found = {}
    if torch.is_tensor(value):
        found[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            found.update(_flatten_tensors(child, f"{prefix}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.update(_flatten_tensors(child, f"{prefix}/{index}"))
    return found


def _recursive_equal(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_recursive_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_recursive_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _load_checkpoint(run_dir):
    path = run_dir / "checkpoint_latest.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return path, payload


def _verify_best(run_dir, payload):
    best = payload.get("best_state", {})
    if best.get("epoch") is None:
        return {"valid": False, "reason": "checkpoint has no committed best epoch"}
    path = run_dir / best["path"]
    if not path.is_file() or sha256_file(path) != best.get("sha256"):
        return {"valid": False, "reason": "best artifact missing or SHA mismatch"}
    state = torch.load(path, map_location="cpu", weights_only=True)
    state_sha = state_dict_sha256(state)
    return {"valid": state_sha == best.get("state_sha256"),
            "epoch": best["epoch"], "validation_loss": best["validation_loss"],
            "file_sha256": sha256_file(path), "state_sha256": state_sha}


def _compare_initialization_summaries(a_dir, b_dir):
    summaries = {}
    for label, path in (("continuous", a_dir / "diagnostic_summary.json"),
                        ("resumed", b_dir / "diagnostic_summary.json")):
        if not path.is_file():
            summaries[label] = {"available": False}
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            summaries[label] = {"available": True,
                                "initialization_sha256": data.get("initialization_sha256"),
                                "initial_model_state_sha256": data.get("initial_model_state_sha256"),
                                "initialization_audit": data.get("initialization_audit")}
    a, b = summaries["continuous"], summaries["resumed"]
    summaries["exact_match"] = bool(a.get("available") and b.get("available") and
        a.get("initialization_sha256") == b.get("initialization_sha256") and
        a.get("initial_model_state_sha256") == b.get("initial_model_state_sha256"))
    return summaries


def _compare_runs(a_dir, b_dir):
    a_path, a = _load_checkpoint(a_dir)
    b_path, b = _load_checkpoint(b_dir)
    a_model = _compare_tensor_states(a["model_state_dict"], b["model_state_dict"])
    a_optimizer = _compare_tensor_states(
        _flatten_tensors(a["optimizer_state_dict"]["state"]),
        _flatten_tensors(b["optimizer_state_dict"]["state"]))
    metrics_a, metrics_b = a["metrics_history"], b["metrics_history"]
    batch_a = [batch for row in metrics_a for batch in row.get("batch_records", [])]
    batch_b = [batch for row in metrics_b for batch in row.get("batch_records", [])]
    train_ids_a = [sid for row in batch_a for sid in row.get("sample_ids", [])]
    train_ids_b = [sid for row in batch_b for sid in row.get("sample_ids", [])]
    aug_a = [seed for row in batch_a for seed in row.get("augmentation_seeds", [])]
    aug_b = [seed for row in batch_b for seed in row.get("augmentation_seeds", [])]
    input_hash_fields = ("model_input_sha256", "target_element_sha256",
                         "target_corrosion_sha256")
    input_hash_sequences_a = [[row.get(key) for key in input_hash_fields] for row in batch_a]
    input_hash_sequences_b = [[row.get(key) for key in input_hash_fields] for row in batch_b]
    numeric_fields = ("learning_rate", "dwa_element", "dwa_corrosion", "train_loss",
                      "train_loss_element", "train_loss_corrosion", "val_loss",
                      "val_loss_element", "val_loss_corrosion")
    epoch_comparison = []
    for left, right in zip(metrics_a, metrics_b):
        differences = {}
        for field in numeric_fields:
            x, y = left.get(field), right.get(field)
            differences[field] = (None if x is None or y is None else
                                  {"continuous": x, "resumed": y,
                                   "abs_diff": abs(float(x)-float(y)),
                                   "exact": float(x) == float(y)})
        epoch_comparison.append({"epoch": left.get("epoch"),
            "learning_rate_and_losses": differences,
            "dwa_exact": all(left.get(k) == right.get(k) for k in
                             ("dwa_element", "dwa_corrosion")),
            "sampler_order_sha256_equal": left.get("sampler_order_sha256") ==
                                          right.get("sampler_order_sha256"),
            "validation_sample_ids_equal": left.get("validation_sample_ids") ==
                                           right.get("validation_sample_ids")})
    overflow_a = [i for i, row in enumerate(batch_a) if row.get("scaler_reduced")]
    overflow_b = [i for i, row in enumerate(batch_b) if row.get("scaler_reduced")]
    scaler_steps_a = [(row.get("scale_before"), row.get("scale_after")) for row in batch_a]
    scaler_steps_b = [(row.get("scale_before"), row.get("scale_after")) for row in batch_b]
    best_a, best_b = _verify_best(a_dir, a), _verify_best(b_dir, b)
    epoch_values_exact = (len(metrics_a) == len(metrics_b) and all(
        all(item["exact"] for item in row["learning_rate_and_losses"].values()
            if item is not None) and row["dwa_exact"] and
        row["sampler_order_sha256_equal"] and row["validation_sample_ids_equal"]
        for row in epoch_comparison))
    best_exact = (best_a.get("valid") is True and best_b.get("valid") is True and
                  best_a.get("epoch") == best_b.get("epoch") and
                  best_a.get("validation_loss") == best_b.get("validation_loss") and
                  best_a.get("state_sha256") == best_b.get("state_sha256"))
    identity_keys = ("trainer_id", "group", "precision_mode", "protocol_sha256",
                     "config_sha256", "source_manifest_sha256",
                     "initialization_sha256", "runtime_sha256")
    checkpoint_identity_exact = all(a.get(key) == b.get(key) for key in identity_keys)
    previous = {}
    for label, run_dir in (("continuous", a_dir), ("resumed", b_dir)):
        path = run_dir / "checkpoint_previous.pth"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            previous[label] = {"sha256": sha256_file(path),
                               "completed_epoch": payload.get("completed_epoch"),
                               "model_state_sha256": state_dict_sha256(
                                   payload["model_state_dict"]),
                               "best_artifact": _verify_best(run_dir, payload)}
        else:
            previous[label] = {"exists": False}
    latest_integrity = {}
    for label, run_dir, payload, path in (("continuous", a_dir, a, a_path),
                                           ("resumed", b_dir, b, b_path)):
        missing = sorted(CHECKPOINT_REQUIRED - set(payload))
        latest_integrity[label] = {
            "sha256": sha256_file(path), "schema_version": payload.get("schema_version"),
            "schema_valid": payload.get("schema_version") == SCHEMA_VERSION and not missing,
            "missing_fields": missing,
            "completed_epoch": payload.get("completed_epoch"),
            "model_state_sha256": state_dict_sha256(payload["model_state_dict"]),
            "best_artifact": _verify_best(run_dir, payload)}
    result = {
        "completed_epoch_equal": a.get("completed_epoch") == b.get("completed_epoch"),
        "checkpoint_identity_exact": checkpoint_identity_exact,
        "epoch_count_equal": len(metrics_a) == len(metrics_b),
        "batch_count_equal": len(batch_a) == len(batch_b),
        "model": a_model,
        "adam_state": a_optimizer,
        "optimizer_non_tensor_fields_equal": _recursive_equal(
            {k:v for k,v in a["optimizer_state_dict"].items() if k != "state"},
            {k:v for k,v in b["optimizer_state_dict"].items() if k != "state"}),
        "scaler_exact": _recursive_equal(a.get("scaler_state_dict"), b.get("scaler_state_dict")),
        "rng_exact": _recursive_equal(a.get("rng_state"), b.get("rng_state")),
        "sampler_state_exact": _recursive_equal(a.get("sampler_state"), b.get("sampler_state")),
        "training_sample_order_exact": train_ids_a == train_ids_b,
        "training_sample_order_count": len(train_ids_a),
        "augmentation_seed_sequence_exact": aug_a == aug_b,
        "augmented_input_and_label_hash_sequences_exact":
            input_hash_sequences_a == input_hash_sequences_b,
        "input_and_label_hash_sequences": {
            "continuous": input_hash_sequences_a,
            "resumed": input_hash_sequences_b},
        "overflow_batch_positions": {"continuous": overflow_a, "resumed": overflow_b,
                                      "equal": overflow_a == overflow_b},
        "grad_scaler_batch_scale_sequence": {
            "continuous": scaler_steps_a, "resumed": scaler_steps_b,
            "exact": scaler_steps_a == scaler_steps_b},
        "epoch_metrics": epoch_comparison,
        "best_artifact": {"continuous": best_a, "resumed": best_b},
        "best_state_exact": best_exact,
        "epoch_metrics_exact_ignoring_wall_clock_time": epoch_values_exact,
        "previous_checkpoint_integrity": previous,
        "latest_checkpoint_integrity": latest_integrity,
        "checkpoint_files": {"continuous_sha256": sha256_file(a_path),
                              "resumed_sha256": sha256_file(b_path),
                              "binary_hash_equal_not_required": True,
                              "reason": "checkpoint embeds wall-clock epoch_seconds"},
        "rng_streams": {"python_numpy_cpu_cuda": "compared recursively and bitwise"},
    }
    result["strict_bitwise_consistency"] = all((
        result["completed_epoch_equal"], result["epoch_count_equal"],
        result["checkpoint_identity_exact"],
        result["batch_count_equal"], result["model"]["exact"],
        result["adam_state"]["exact"],
        result["optimizer_non_tensor_fields_equal"], result["scaler_exact"],
        result["rng_exact"], result["sampler_state_exact"],
        result["training_sample_order_exact"],
        result["augmentation_seed_sequence_exact"],
        result["augmented_input_and_label_hash_sequences_exact"],
        result["overflow_batch_positions"]["equal"],
        result["grad_scaler_batch_scale_sequence"]["exact"],
        result["best_state_exact"], result["epoch_metrics_exact_ignoring_wall_clock_time"],
        all(item.get("schema_valid") and item["best_artifact"].get("valid")
            for item in result["latest_checkpoint_integrity"].values()),
        all(item.get("best_artifact", {}).get("valid", False)
            for item in result["previous_checkpoint_integrity"].values())
    ))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("G0", "G4"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute-resume-consistency", action="store_true")
    args = parser.parse_args()
    if not args.execute_resume_consistency:
        print(json.dumps({"status": "PREPARED_NOT_RUN", "gpu_accessed": False,
                          "images_accessed": False,
                          "planned_paths": ["continuous-2-epoch", "resume-1-plus-1"]}, indent=2))
        return 0
    if not all((args.data_root, args.runtime_lock, args.output_root)):
        raise SystemExit("Execution requires --data-root, --runtime-lock and --output-root")
    verify_source_lock(ROOT, "SOURCE_LOCK.v5.json")
    output_root = args.output_root.resolve()
    allowed = (ROOT / "diagnostics" / "resume_consistency").resolve()
    if not output_root.is_relative_to(allowed) or output_root.exists():
        raise SystemExit(f"Use a never-used output directory beneath {allowed}")
    if not args.data_root.is_dir() or not args.runtime_lock.is_file():
        raise SystemExit("Data root or runtime lock is unavailable")
    phrase = f"APPROVE_G4_V5_RESUME_EXPERIMENT_{args.group}"
    if input(f"Type {phrase} to run two bounded diagnostic paths: ").strip() != phrase:
        raise SystemExit("No diagnostic was run: explicit confirmation absent")
    output_root.mkdir(parents=True)
    group_root = output_root / args.group
    a_dir, b_dir = group_root / "continuous_2_epochs", group_root / "resume_1_plus_1"
    logs = group_root / "process_logs"
    common = {"data_root": args.data_root.resolve(),
              "runtime_lock": args.runtime_lock.resolve()}
    process_records = []
    commands = [
        ("A_continuous_2_epochs", a_dir, 2, False),
        ("B1_first_epoch", b_dir, 1, False),
        ("B2_resume_second_epoch", b_dir, 2, True),
    ]
    for label, run_dir, limit, resume in commands:
        cmd, child_phrase = _command(args.group, common["data_root"],
                                     common["runtime_lock"], run_dir, limit, resume)
        process_records.append(_run_child(label, cmd, child_phrase, logs))
    comparison = _compare_runs(a_dir, b_dir)
    resume_summary_path = b_dir / "diagnostic_summary.json"
    if resume_summary_path.is_file():
        comparison["actual_resume_entry_audit"] = json.loads(
            resume_summary_path.read_text(encoding="utf-8")).get("resume_audit")
    initialization_comparison = _compare_initialization_summaries(a_dir, b_dir)
    comparison["initialization_exact"] = initialization_comparison["exact_match"]
    comparison["strict_bitwise_consistency"] = (
        comparison["strict_bitwise_consistency"] and initialization_comparison["exact_match"])
    report = {"group": args.group, "status": "COMPARISON_COMPLETE_REVIEW_REQUIRED",
              "protocol_total_epochs": 150, "path_A": "2 continuous epochs",
              "path_B": "1 epoch + process restart + resume 1 epoch",
              "data_root": str(common["data_root"]),
              "runtime_lock_sha256": sha256_file(common["runtime_lock"]),
              "source_lock": verify_source_lock(ROOT, "SOURCE_LOCK.v5.json"),
              "child_processes": process_records, "comparison": comparison,
              "initialization_comparison": initialization_comparison,
              "created_unix": time.time()}
    report_path = group_root / "resume_consistency_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
