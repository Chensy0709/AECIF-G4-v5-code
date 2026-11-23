"""Create a new immutable SHA-256 closure for the active G4-v5 package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "SOURCE_LOCK.v5.json"
SIDECAR = ROOT / "SOURCE_LOCK.v5.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_files():
    paths = set(ROOT.glob("*.py"))
    paths.update(p for folder in ("configs", "vendor", "tests")
                 for p in (ROOT / folder).rglob("*") if p.is_file())
    paths.update((ROOT / "README_DSW.md", ROOT / "requirements_dsw.txt"))
    excluded = {LOCK, SIDECAR}
    return sorted((p for p in paths if p.is_file() and p not in excluded and
                   "__pycache__" not in p.parts and p.suffix != ".pyc"),
                  key=lambda p: p.relative_to(ROOT).as_posix())


def main():
    if LOCK.exists() or SIDECAR.exists():
        raise FileExistsError("Refusing to overwrite an existing v5 source lock")
    v4_lock = ROOT / "provenance" / "source_v4" / "SOURCE_LOCK.v4.json"
    v3_lock = ROOT / "provenance" / "source_v3" / "SOURCE_LOCK.v3.json"
    manifest = {
        "schema": "g4-v1-source-lock-v2", "hash_algorithm": "SHA-256",
        "ordering": "relative_path ascending, POSIX separators",
        "deployment_version": "AECIF_G4_v1_DSW_Linux_v5",
        "derived_from_SOURCE_LOCK_v4_sha256": sha256(v4_lock),
        "preserved_SOURCE_LOCK_v3_sha256": sha256(v3_lock),
        "changes": {
            "configs/protocol.json": "external initialization checkpoint path updated to the v5 deployment root; checkpoint SHA, protocol values and training math unchanged",
            "checkpointing.py": "normalize CPU and CUDA RNG state tensors to CPU uint8 before restoration",
            "fp16_isolated_probe.py": "handle tuple model outputs, image float32 conversion, preserve label dtype, v5 lock and diagnostic-only outputs",
            "unified_trainer.py": "isolated bounded diagnostic scope; restore checkpoint RNG after dataset/setup initialization; record sample IDs and augmentation seeds",
            "trainer_core.py": "record train sample IDs, augmentation seeds, validation IDs, per-step scaler state and diagnostic-only CPU hashes of actual normalized input/label batches",
            "diagnostic_entry.py": "new explicit 1/2 epoch diagnostic gate; locked protocol remains 150 epochs",
            "resume_consistency_experiment.py": "new serial A/B continuity-vs-restart comparison driver including initialization state hashes and actual input hashes; inert by default",
            "capture_runtime_lock.py": "new explicit A10 runtime identity capture helper",
            "tests": "CPU contract tests for RNG restore, dtypes, per-sample seeds, sampler, diagnostic gates, and G4 structure"
        },
        "excluded": ["images", "labels", "checkpoints", "weights", "caches",
                     "diagnostics output", "runs", "logs", "independent test set",
                     "provenance snapshots"],
        "files": []
    }
    for path in active_files():
        manifest["files"].append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    LOCK.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    lock_sha = sha256(LOCK)
    SIDECAR.write_text(f"{lock_sha}  {LOCK.name}\n", encoding="ascii")
    print(json.dumps({"lock": LOCK.name, "sha256": lock_sha,
                      "file_count": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
