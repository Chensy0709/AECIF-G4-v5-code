# AECIF-Net G4-v5 — Aliyun DSW preparation

This is a new immutable deployment version derived from the v4 Linux package. The complete v3 and v4 source snapshots are retained under `provenance/source_v3/` and `provenance/source_v4/`. No image, label, initialization checkpoint, run checkpoint, or independent test data is packaged.

## Locked experiment protocol

The shared protocol remains 200 train / 34 validation IDs, 520×520, batch 4, Adam, original CE + DWA, the 150-epoch cosine schedule (including the original warmup and tail), and FP16 + GradScaler. G0/G-E/G-D/G4/G4-SS/G4-Swap/G4-NoCross model routing is unchanged. The independent 100-image test set is not used. Do not resume v3/v4 checkpoints in v5.

## External files required on DSW

- Initialization checkpoint: `/mnt/workspace/AECIF_G4_v5_independent/resources/initialization/best_checkpoint_86.76_PSA_s.pth`; expected SHA-256 `45c202530d8e6831fc56fc210c0597c2d04cd51fdbda10a0ca70101a33dadcf3` (621,699,742 bytes).
- Dataset root: `/mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit`; the loader uses `VOC2007/JPEGImages`, `VOC2007/SegmentationClass/element`, and `VOC2007/SegmentationClass/defect`.
- The locked `configs/data_splits/train_200.txt` and `val.txt` are included. No 100-image test data/list is included.
- Use the pre-existing DSW Python packages; do not install or change Torch/CUDA/driver as part of this package.

## Verify after extraction

```bash
cd /mnt/workspace/AECIF_G4_v5_independent
python - <<'PY'
from pathlib import Path
from source_lock import verify_source_lock
print(verify_source_lock(Path('.'), 'SOURCE_LOCK.v5.json'))
PY
sha256sum resources/initialization/best_checkpoint_86.76_PSA_s.pth
```

The checkpoint must match the locked SHA above. Check the external dataset folders and split counts before any authorized diagnostic. The source lock does not include these external files.

## Capture the A10 runtime lock (preparation only)

The helper is inert unless `--capture` is passed. Run it on the intended A10 after confirming no competing GPU process. It records Python, Torch, CUDA, GPU, driver/UUID, cuDNN, compute capability, and current free/total memory.

```bash
python capture_runtime_lock.py --capture --output diagnostics/runtime_lock_a10.json
```

The trainer checks the Python/Torch/CUDA/GPU identity against this file. Keep the full file with the experiment records; do not edit it by hand.

## Bounded 1–2 epoch diagnostic entry

`diagnostic_entry.py` is inert by default. The diagnostic uses the same locked 150-epoch protocol and stops at a requested completed-epoch boundary of 1 or 2; it does not rewrite the protocol's total epoch count. Every diagnostic run is isolated under `diagnostics/runs/` and requires an explicit phrase. It cannot enter the formal `fit()` path. The formal `train_entry.py` remains inert by default and requires the separate 150-epoch authorization and `--epoch-budget 150`.

Do not run the following commands until the user separately authorizes GPU diagnostics. Example for a fresh G0 two-epoch diagnostic:

```bash
python diagnostic_entry.py --group G0 \
  --data-root /mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit \
  --runtime-lock diagnostics/runtime_lock_a10.json \
  --run-dir diagnostics/runs/A10_G0_continuous_2ep \
  --epoch-limit 2 --execute-diagnostic
```

The script asks for `APPROVE_G4_V5_G0_DIAGNOSTIC_UP_TO_EPOCH_2`. The analogous G4 phrase names `G4`. For a 1+1 resume path, run a fresh directory with `--epoch-limit 1`, then restart Python and run the same directory with `--resume --epoch-limit 2`.

## Prepared A10 resume-consistency experiment (not run)

`resume_consistency_experiment.py` starts three serial child processes for one selected group: A runs two continuous epochs; B runs one epoch, exits, then a new process resumes and runs epoch two. It requires an unused output root under `diagnostics/resume_consistency/`, verifies the v5 source lock, and prompts for a separate group-specific approval phrase. It compares the initial model-state SHA; train sample order, augmentation seeds, and CPU-side SHA-256 of actual float32 model-input and both class-index label batches; per-epoch LR/DWA/losses; per-batch GradScaler scales and overflow positions; validation IDs; model and Adam tensor differences; scaler/RNG/sampler states; best artifacts; and latest/previous checkpoint integrity. It reports differences; it never declares numerical equivalence solely because restoration succeeded. Run G0 and G4 serially, reviewing G0 before starting G4.

Example commands (prepared only; do not execute without separate authorization):

```bash
python resume_consistency_experiment.py --group G0 \
  --data-root /mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit \
  --runtime-lock diagnostics/runtime_lock_a10.json \
  --output-root diagnostics/resume_consistency/A10_G0_20261120 \
  --execute-resume-consistency

# Review the G0 report before separately authorizing the G4 command.
python resume_consistency_experiment.py --group G4 \
  --data-root /mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit \
  --runtime-lock diagnostics/runtime_lock_a10.json \
  --output-root diagnostics/resume_consistency/A10_G4_20261120 \
  --execute-resume-consistency
```

The parent asks `APPROVE_G4_V5_RESUME_EXPERIMENT_G0` or the corresponding G4 phrase. Each child separately asks the bounded diagnostic phrase. Outputs go only under the new diagnostic directories. Any CUDA error, lock/config mismatch, missing external resource, or child-process failure stops that group's sequence. This is a 2-epoch reproducibility diagnostic, not a performance result or a formal training run.

## CPU-only checks

```bash
python -m py_compile checkpointing.py fp16_isolated_probe.py trainer_core.py unified_trainer.py diagnostic_entry.py resume_consistency_experiment.py capture_runtime_lock.py train_entry.py source_lock.py build_source_lock_v5.py
python -m unittest discover -s tests -v
```

The CPU-only original-weight-loading test is skipped when the external initialization checkpoint is not staged locally. No GPU call, image load, or training occurs in the default diagnostic/runtime-lock entrypoints.
