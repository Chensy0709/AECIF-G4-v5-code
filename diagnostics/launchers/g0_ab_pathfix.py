import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/workspace/AECIF_G4_v5_independent")
SOURCE = ROOT / "resume_consistency_experiment.py"

original_bytes = SOURCE.read_bytes()
original = original_bytes.decode("utf-8")

old = 'allowed = (ROOT / "diagnostics" / "resume_consistency").resolve()'
new = 'allowed = (ROOT / "diagnostics" / "runs" / "resume_consistency").resolve()'

if original.count(old) != 1:
    raise RuntimeError("源码与预期不符，拒绝执行")

patched = original.replace(old, new, 1)
compiled = compile(patched, str(SOURCE), "exec")

tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

OUT = ROOT / "diagnostics/runs/resume_consistency" / f"A10_G0_{tag}"

if OUT.exists():
    raise RuntimeError("输出目录已存在，拒绝覆盖")

audit = {
    "purpose": "G0 A/B output path compatibility diagnostic",
    "formal_training": False,
    "original_source_sha256": hashlib.sha256(original_bytes).hexdigest(),
    "executed_source_sha256": hashlib.sha256(
        patched.encode("utf-8")
    ).hexdigest(),
    "replacement": {"old": old, "new": new},
    "source_files_modified": False,
    "output_root": str(OUT),
}

audit_path = ROOT / "diagnostics/launchers" / f"G0_pathfix_audit_{tag}.json"

with audit_path.open("x", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)

print("===== G0恢复一致性实验 =====", flush=True)
print("临时修复: 统一输出目录", flush=True)
print("v5锁定源码: 未修改", flush=True)
print("修复审计:", audit_path, flush=True)
print("实验输出:", OUT, flush=True)

sys.path.insert(0, str(ROOT))

sys.argv = [
    str(SOURCE),
    "--group", "G0",
    "--data-root",
    "/mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit",
    "--runtime-lock",
    str(ROOT / "diagnostics/runtime_lock_a10.json"),
    "--output-root",
    str(OUT),
    "--execute-resume-consistency"
]

exec(compiled, {
    "__name__": "__main__",
    "__file__": str(SOURCE)
})
