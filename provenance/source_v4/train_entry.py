"""Shared formal-training entry for G4-v1. Disabled unless explicitly gated."""
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
    parser.add_argument("--group", choices=("G0", "G-E", "G-D", "G4", "G4-SS",
                                               "G4-Swap", "G4-NoCross"), required=True)
    parser.add_argument("--data-root", type=Path, required=False)
    parser.add_argument("--run-dir", type=Path, required=False)
    parser.add_argument("--runtime-lock", type=Path, required=False)
    parser.add_argument("--epoch-budget", type=int, required=False,
                        help="Must equal the locked 150-Epoch total budget")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-formal-training", action="store_true",
                        help="Explicitly execute all remaining epochs (requires separate approval)")
    args = parser.parse_args()
    if not args.execute_formal_training:
        print(json.dumps({"status": "PREPARED_NOT_RUN", "group": args.group,
                          "gpu_accessed": False, "image_accessed": False,
                          "formal_training": False,
                          "protocol": "locked 200/34, 520, batch4, 150 epochs, FP16+GradScaler",
                          "next_gate": "Requires separate approval and complete reviewed runtime lock"},
                         ensure_ascii=False, indent=2))
        return 0
    missing = [name for name, value in (("--data-root", args.data_root),
                                        ("--run-dir", args.run_dir),
                                        ("--runtime-lock", args.runtime_lock)) if value is None]
    if missing:
        raise SystemExit("Missing required formal-run arguments: " + ", ".join(missing))
    if args.epoch_budget != 150:
        raise SystemExit("Formal run requires --epoch-budget 150; no other budget is supported")
    runs_root = (ROOT / "runs").resolve()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(runs_root):
        raise SystemExit(f"Run output must be isolated beneath {runs_root}")
    if args.resume:
        if not run_dir.is_dir() or not any(run_dir.iterdir()):
            raise SystemExit("--resume requires an existing, non-empty run directory")
    elif run_dir.exists():
        raise SystemExit("A new formal run requires a never-used output directory")
    # This entry is intentionally a 150-Epoch formal path, not a short-run switch.
    approval = input("Type APPROVE_G4_V1_FORMAL_150_EPOCHS to proceed: ").strip()
    if approval != "APPROVE_G4_V1_FORMAL_150_EPOCHS":
        raise SystemExit("Formal training not started: explicit confirmation absent")
    from unified_trainer import UnifiedTrainer
    lock = json.loads(args.runtime_lock.read_text(encoding="utf-8"))
    trainer = UnifiedTrainer(args.group, args.data_root, run_dir,
                             runtime_lock=lock, resume=args.resume)
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
