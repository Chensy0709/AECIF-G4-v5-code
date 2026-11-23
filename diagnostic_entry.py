"""Explicitly gated 1- or 2-Epoch diagnostic path; never a formal-run shortcut."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("G0", "G4"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--epoch-limit", type=int, choices=(1, 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-diagnostic", action="store_true")
    args = parser.parse_args()
    if not args.execute_diagnostic:
        print(json.dumps({"status": "PREPARED_NOT_RUN", "gpu_accessed": False,
                          "image_accessed": False, "epoch_limit": None,
                          "formal_protocol_epochs": 150}, indent=2))
        return 0
    required = (("--data-root", args.data_root), ("--run-dir", args.run_dir),
                ("--runtime-lock", args.runtime_lock), ("--epoch-limit", args.epoch_limit))
    missing = [name for name, value in required if value is None]
    if missing:
        raise SystemExit("Missing diagnostic arguments: " + ", ".join(missing))
    diagnostic_root = (ROOT / "diagnostics" / "runs").resolve()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(diagnostic_root):
        raise SystemExit(f"Diagnostic output must be beneath {diagnostic_root}")
    if args.resume:
        if not run_dir.is_dir() or not any(run_dir.iterdir()):
            raise SystemExit("--resume requires an existing, non-empty diagnostic run")
    elif run_dir.exists():
        raise SystemExit("A new diagnostic run requires a never-used output directory")
    phrase = f"APPROVE_G4_V5_{args.group}_DIAGNOSTIC_UP_TO_EPOCH_{args.epoch_limit}"
    if input(f"Type {phrase} to run this bounded diagnostic: ").strip() != phrase:
        raise SystemExit("Diagnostic not started: explicit confirmation absent")
    lock = json.loads(args.runtime_lock.read_text(encoding="utf-8"))
    from unified_trainer import UnifiedTrainer
    trainer = UnifiedTrainer(args.group, args.data_root, run_dir,
                             runtime_lock=lock, resume=args.resume,
                             run_scope="diagnostic")
    history = trainer.fit_diagnostic(args.epoch_limit)
    summary = {"status": "DIAGNOSTIC_COMPLETE", "group": args.group,
               "epoch_limit": args.epoch_limit,
               "completed_epoch": trainer.completed_epoch + 1,
               "protocol_total_epochs": trainer.protocol["training"]["epochs"],
               "history": history,
               "source_lock_sha256": trainer.source_sha,
               "protocol_sha256": trainer.protocol_sha,
               "config_sha256": trainer.config_sha,
               "initialization_sha256": trainer.init_meta["sha256"],
               "initial_model_state_sha256": trainer.initial_model_state_sha256,
               "initialization_audit": trainer.init_audit,
               "resume_audit": trainer.resume_audit}
    (run_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("status", "group", "epoch_limit", "completed_epoch",
                       "protocol_total_epochs", "source_lock_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
