"""Strict verifier for the active, versioned source lock."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_lock(root: Path, lock_name: str = "SOURCE_LOCK.v3.json") -> dict:
    root = Path(root).resolve()
    lock_path = root / lock_name
    sidecar_path = root / lock_name.replace(".json", ".sha256")
    if not lock_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"Versioned source lock is incomplete: {lock_name}")
    actual_lock_sha = sha256_file(lock_path)
    sidecar_fields = sidecar_path.read_text(encoding="ascii").strip().split()
    if len(sidecar_fields) != 2 or sidecar_fields[0].lower() != actual_lock_sha:
        raise RuntimeError("Source lock sidecar SHA-256 mismatch")
    if sidecar_fields[1] != lock_name:
        raise RuntimeError("Source lock sidecar names a different manifest")
    manifest = json.loads(lock_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "g4-v1-source-lock-v2" or \
            manifest.get("hash_algorithm") != "SHA-256":
        raise RuntimeError("Unsupported source lock schema or hash algorithm")
    if manifest.get("copied_source_mismatches"):
        raise RuntimeError("Source lock records copied-source hash mismatches")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Source lock has no file entries")
    seen = set()
    failures = []
    for item in files:
        rel_text = item.get("relative_path")
        if not isinstance(rel_text, str):
            failures.append({"path": rel_text, "reason": "invalid relative path"})
            continue
        rel = PurePosixPath(rel_text)
        if rel.is_absolute() or ".." in rel.parts or "\\" in rel_text:
            failures.append({"path": rel_text, "reason": "unsafe relative path"})
            continue
        normalized = rel.as_posix()
        if normalized in seen:
            failures.append({"path": normalized, "reason": "duplicate entry"})
            continue
        seen.add(normalized)
        file_path = root.joinpath(*rel.parts)
        if not file_path.is_file():
            failures.append({"path": normalized, "reason": "missing"})
            continue
        if file_path.stat().st_size != item.get("size_bytes"):
            failures.append({"path": normalized, "reason": "size mismatch"})
            continue
        if sha256_file(file_path) != item.get("sha256"):
            failures.append({"path": normalized, "reason": "SHA-256 mismatch"})
    if failures:
        raise RuntimeError(f"Source files differ from immutable lock: {failures[:20]}")
    return {"lock_path": str(lock_path), "lock_sha256": actual_lock_sha,
            "file_count": len(files), "verified": True}
