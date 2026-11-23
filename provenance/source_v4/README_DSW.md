# AECIF-Net G4-v1 DSW deployment

Runnable Linux deployment copy is at this directory root and uses SOURCE_LOCK.v4.json. The exact original v3 lock and its full 78-file closure are preserved under provenance/source_v3/; that archival tree is not the active runtime.

## External files (not included)
- Initialization checkpoint must be copied to `/mnt/workspace/AECIF_G4_v1_independent/resources/initialization/best_checkpoint_86.76_PSA_s.pth`; expected SHA-256 `45c202530d8e6831fc56fc210c0597c2d04cd51fdbda10a0ca70101a33dadcf3`, size 621699742 bytes.
- Dataset root is configured as `/mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit`; loader expects `VOC2007/JPEGImages`, `VOC2007/SegmentationClass/element`, `VOC2007/SegmentationClass/defect`. Check availability on DSW. No images/labels are packaged.
- Train/validation ID lists are packaged (200/34). No test list/data is packaged.

Vendored AECIF-Net/DTCWT sources and coefficient assets are present. Extra Python packages are listed in requirements_dsw.txt. Torch, CUDA and the NVIDIA driver are intentionally not pinned or installed; use the DSW image and lock its actual runtime before GPU use.

## CPU-only source verification
```bash
python - <<'PY'
from pathlib import Path
from source_lock import verify_source_lock
print(verify_source_lock(Path('.'), 'SOURCE_LOCK.v4.json'))
PY
cd provenance/source_v3
python - <<'PY'
from pathlib import Path
from source_lock import verify_source_lock
print(verify_source_lock(Path('.'), 'SOURCE_LOCK.v3.json'))
PY
```

The original Windows config exists only in provenance. Do not replace the Linux protocol.json with it. Extracting this ZIP does not run CUDA or training. The formal entry remains gated; the probe requires explicit execution authorization and `--max-batches 1..3`.
