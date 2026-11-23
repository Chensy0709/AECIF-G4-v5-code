# 阿里云A10断点恢复一致性实验准备

状态：已准备，未执行。运行A/B实验需要用户后续单独授权。

## 外部资源检查

- 运行目录：`/mnt/workspace/AECIF_G4_v5_independent`
- 数据根目录：`/mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit`（仅200训练ID用于短诊断；验证集用于训练流程中每轮的既有验证；100张测试集不读取）
- 初始化Checkpoint：`/mnt/workspace/AECIF_G4_v5_independent/resources/initialization/best_checkpoint_86.76_PSA_s.pth`
- 预期Checkpoint SHA-256：`45c202530d8e6831fc56fc210c0597c2d04cd51fdbda10a0ca70101a33dadcf3`
- A10 runtime lock：在云端显式运行 `capture_runtime_lock.py --capture` 生成；必须核对 Python、Torch、CUDA、GPU名，且GPU名含A10。

## 实验路径

对G0和G4分别执行且不并发：

| 路径 | 进程 | 执行 |
|---|---|---|
| A | 子进程A | 从同一初始化和锁定种子连续完成Epoch 1–2 |
| B | 子进程B1 | 从相同初始化和种子完成Epoch 1，提交完整Checkpoint后退出 |
| B | 新子进程B2 | 通过实际 `UnifiedTrainer(resume=True)` 从Epoch 1状态恢复，仅完成Epoch 2 |

正式协议中的total Epoch始终为150，诊断只运行边界1或2。每组实际完成2+1+1个诊断Epoch工作，产物只写到新建 `diagnostics/` 子目录，不修改正式历史结果。

## 对照字段

- 样本顺序：每批sample IDs、每轮sampler顺序SHA-256。
- 增强：每样本派生的augmentation seed，以及进入模型前实际float32图像张量和两张long类别索引标签的CPU-side SHA-256；原始增强操作仍按锁定代码执行。
- 初始化：逐进程记录加载指定Checkpoint后的整网state SHA，A/B必须一致。
- 优化过程：逐轮LR、DWA、两路train/validation loss；逐批scale前后值、scale下降与optimizer跳步/overflow批次位置。
- 状态：模型所有state tensor逐项比较；Adam所有state tensor及param group比较；GradScaler、Python/NumPy/CPU/CUDA RNG、sampler状态逐项比较。
- 完整性：Best文件SHA和state SHA；latest与previous checkpoint可读取性、schema、完成Epoch、模型state SHA及关联Best有效性。

Checkpoint内含 `epoch_seconds`，因此不要求整份checkpoint二进制SHA相同；报告同时保存两边文件SHA，严格比较训练状态字段。只有所有定义的字段都逐位一致时才标记 `strict_bitwise_consistency=true`。不一致时提供张量最大绝对差和相对L2差，并保留逐批/逐轮记录；不自动把容差差异解释为通过。

## 云端准备及命令

上传并解压 `AECIF_G4_v5_deploy.zip` 到上述运行目录，再执行：

```bash
cd /mnt/workspace/AECIF_G4_v5_independent
python - <<'PY'
from pathlib import Path
from source_lock import verify_source_lock
print(verify_source_lock(Path('.'), 'SOURCE_LOCK.v5.json'))
PY
sha256sum resources/initialization/best_checkpoint_86.76_PSA_s.pth
python capture_runtime_lock.py --capture --output diagnostics/runtime_lock_a10.json
```

确认SOURCE_LOCK逐文件通过、权重哈希正确、名单200/34及无GPU并发后，等待明确授权再运行下面的命令。每个命令均会再次询问确认口令：

```bash
python resume_consistency_experiment.py --group G0 \
  --data-root /mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit \
  --runtime-lock diagnostics/runtime_lock_a10.json \
  --output-root diagnostics/resume_consistency/A10_G0_<unique-id> \
  --execute-resume-consistency

# 审查G0后再单独授权G4，且保持串行
python resume_consistency_experiment.py --group G4 \
  --data-root /mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit \
  --runtime-lock diagnostics/runtime_lock_a10.json \
  --output-root diagnostics/resume_consistency/A10_G4_<unique-id> \
  --execute-resume-consistency
```

若需要单独运行1或2轮短程诊断，可调用 `diagnostic_entry.py --epoch-limit 1|2 --execute-diagnostic`，其输出必须使用 `diagnostics/runs/` 下的新目录。任何资源/哈希/配置异常、OOM或子进程失败均停止该组；不得切换FP16、Batch、输入尺寸、损失或优化器。
