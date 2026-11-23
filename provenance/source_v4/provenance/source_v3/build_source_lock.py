"""Create a deterministic hash manifest for the isolated G4-v1 source bundle."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def original_for(relative: Path):
    parts = relative.parts
    if parts[-1] == "__init__.py" and parts[0:2] in {
            ("vendor", "baseline_code"), ("vendor", "models"),
            ("vendor", "utils")}:
        return None
    if len(parts) >= 4 and parts[:3] == ("vendor", "baseline_code", "nets"):
        return PROJECT / "baseline_code" / "nets" / Path(*parts[3:])
    if len(parts) >= 4 and parts[:3] == ("vendor", "baseline_code", "utils"):
        return PROJECT / "baseline_code" / "utils" / Path(*parts[3:])
    if len(parts) == 3 and parts[:2] == ("vendor", "utils"):
        return PROJECT / "baseline_code" / "utils" / parts[2]
    if len(parts) == 3 and parts[:2] == ("vendor", "models"):
        return PROJECT / "models" / parts[2]
    if len(parts) >= 4 and parts[:3] == ("vendor", "pytorch_wavelets", "pytorch_wavelets"):
        return (PROJECT / "wavelets" / "dtcwt" / "vendor" /
                "pytorch_wavelets" / Path(*parts[3:]))
    if len(parts) >= 3 and parts[:2] == ("configs", "data_splits"):
        return PROJECT / "configs" / "data_splits" / Path(*parts[2:])
    if len(parts) >= 2 and parts[0] == "design_reference":
        return (PROJECT / "designs" / "G4_complementarity_design_v1" /
                Path(*parts[1:]))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=3,
                        help="Create a new immutable SOURCE_LOCK.vN.json; existing versions are never overwritten")
    args = parser.parse_args()
    if args.version < 2:
        raise SystemExit("New lock versions must be >= 2; SOURCE_LOCK.json is preserved")
    lock_name = f"SOURCE_LOCK.v{args.version}.json"
    lock = ROOT / lock_name
    sidecar = ROOT / lock_name.replace(".json", ".sha256")
    if lock.exists() or sidecar.exists():
        raise SystemExit(f"Refusing to overwrite existing immutable lock version {args.version}")
    candidates = []
    for root_name in ("vendor", "configs", "design_reference", "tests"):
        for path in (ROOT / root_name).rglob("*"):
            if (path.is_file() and "__pycache__" not in path.parts and
                    path.suffix.lower() in {".py", ".npz", ".json", ".txt", ".md"}):
                candidates.append(path)
    candidates += [ROOT / name for name in
                   ("g4_model.py", "compute_costs.py", "parameter_audit.py",
                    "fp16_isolated_probe.py", "audit_group_configs.py",
                    "build_source_lock.py", "trainer_core.py", "checkpointing.py",
                    "unified_trainer.py", "train_entry.py", "source_lock.py",
                    "README.md")]
    entries = []
    copy_mismatches = []
    for path in sorted(set(candidates), key=lambda p: p.relative_to(ROOT).as_posix()):
        rel = path.relative_to(ROOT)
        current_sha = digest(path)
        source = original_for(rel)
        source_sha = digest(source) if source and source.exists() else None
        if source and (not source.exists() or source_sha != current_sha):
            copy_mismatches.append({"relative_path": rel.as_posix(),
                                    "source_path": str(source),
                                    "source_exists": source.exists(),
                                    "copied_sha256": current_sha,
                                    "source_sha256": source_sha})
        entries.append({"relative_path": rel.as_posix(),
                        "origin_path": str(source) if source else "created in isolated G4-v1",
                        "size_bytes": path.stat().st_size,
                        "sha256": current_sha,
                        "origin_sha256": source_sha})
    init = json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))["initialization"]
    payload = {
        "schema": "g4-v1-source-lock-v2",
        "hash_algorithm": "SHA-256",
        "ordering": "relative_path ascending, POSIX separators",
        "excluded": ["reports", "logs", "__pycache__", "*.pyc",
                     "SOURCE_LOCK.json", "SOURCE_LOCK.sha256",
                     "SOURCE_LOCK.vN.json", "SOURCE_LOCK.vN.sha256"],
        "files": entries,
        "copied_source_mismatches": copy_mismatches,
        "external_read_only_initialization": init,
    }
    if copy_mismatches:
        print(json.dumps({"copied_source_mismatches": copy_mismatches}, indent=2))
        raise SystemExit(2)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    import os
    import tempfile
    fd, temp_name = tempfile.mkstemp(prefix=lock_name + ".", suffix=".tmp", dir=ROOT)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, lock)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    lock_hash = digest(lock)
    sidecar.write_text(f"{lock_hash}  {lock_name}\n", encoding="ascii")
    print(json.dumps({"files": len(entries),
                      "copied_source_mismatches": len(copy_mismatches),
                      "source_lock": lock_name,
                      "source_lock_sha256": lock_hash}, indent=2))
    if copy_mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
