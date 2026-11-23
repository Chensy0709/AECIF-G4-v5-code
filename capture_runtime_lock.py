"""Capture an explicit Linux runtime identity; inert unless --capture is given."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.capture:
        print(json.dumps({"status": "PREPARED_NOT_RUN", "cuda_accessed": False}, indent=2))
        return 0
    if not args.output:
        raise SystemExit("--output is required with --capture")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; runtime lock not written")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    gpu_name = torch.cuda.get_device_name(device)
    if "a10" not in gpu_name.lower():
        raise SystemExit(f"Expected an NVIDIA A10; got {gpu_name!r}. Runtime lock not written")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,uuid,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"], check=True, capture_output=True,
            text=True, timeout=15).stdout.strip().splitlines()[device]
    except Exception as exc:
        driver = None
        driver_error = f"{type(exc).__name__}: {exc}"
    else:
        driver_error = None
    row = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda,
           "gpu": gpu_name,
           "device_index": device, "gpu_uuid_and_driver_nvidia_smi": driver,
           "nvidia_smi_error": driver_error,
           "cudnn": torch.backends.cudnn.version(),
           "compute_capability": list(properties.major_minor) if hasattr(properties, "major_minor")
                                else [properties.major, properties.minor],
           "total_memory_bytes": total_bytes, "free_memory_bytes": free_bytes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
