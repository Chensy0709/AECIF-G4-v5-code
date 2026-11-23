# G4-v5 修复与封版审计

## 结论

建立独立 `g4_v5_independent`，未改动 v3/v4 快照或原项目。v3 的 78/78 与 v4 的 63/63 清单分别在只读快照中重新验证通过。v5 的 `SOURCE_LOCK.v5.json` 覆盖72个活动源码、配置、vendor资产和CPU测试文件，sidecar及逐文件 SHA-256 验证通过。

本阶段没有启动GPU、没有读取真实训练/验证/测试图片，也没有运行模型训练。A10报告中的先前结果来自用户提供的数据，本报告没有重新测量或作性能推断。

## 正式修复

1. `fp16_isolated_probe.py` 将模型二元组输出分别 detach，明确把输入图像转为 float32，并保持标签张量的原数据类型。探针输出限制在 `diagnostics/`。
2. `checkpointing.py` 恢复 Python/NumPy/CPU/CUDA RNG 时，将CPU和CUDA RNG状态统一整理为CPU上的一维 `torch.uint8` 连续张量，再交给PyTorch恢复接口。
3. `SeededDataset` 使用 `(seed, epoch, sample_id)` 派生每图增强种子，在调用原始 NumPy 增强时临时设置随机状态并恢复原状态。源码审计确认原始增强调用 NumPy RNG；DataLoader固定 `num_workers=0`。恢复后，构造数据集等准备操作完成后再次恢复Checkpoint RNG，之后才创建下一轮DataLoader迭代器。
4. 修正审计元数据布局：基线Dataset原始返回图像、两张类别索引掩码和两份未被训练器使用的one-hot标签。v5包装器保留前三项，并把 `sample_id` 和 `augmentation_seed` 放到批次第4/5项，避免把one-hot张量误读成元数据；图像增强、标签值、CE/DWA和模型不变。
5. 新增 `diagnostic_entry.py`：默认只打印 `PREPARED_NOT_RUN`；执行时限定1或2个完成Epoch，配置总预算仍为150。它使用 `diagnostics/runs/` 独立目录和单独确认口令。正式入口仍要求显式授权与 `--epoch-budget 150`，不能进入短程路径。
6. 新增 `resume_consistency_experiment.py`，为G0/G4各自串行执行 A=连续2轮、B=1轮后退出并由新进程恢复第2轮。诊断模式逐批记录规范化模型输入和两路标签张量SHA-256、样本ID/增强种子/GradScaler scale及溢出位置；逐轮比较LR/DWA/Train/Validation Loss；比较模型与Adam张量差异、RNG/采样状态、Best文件及 latest/previous checkpoint完整性。输入哈希仅在诊断模式计算，不增加正式训练开销。脚本仅准备，默认不访问GPU或图像；比较结果不会因恢复成功而自动判定等价。

## 实验协议保护

四组路由配置、训练/验证名单、初始化权重SHA、FP16+GradScaler、CE/DWA、Adam、warmup/tail及150轮调度均保留v4锁定值。`protocol.json` 中初始化Checkpoint仅将绝对目录从v1部署根改为v5部署根，文件名、SHA和加载规则不变。G0/G-E/G-D/G4/G4-SS/G4-Swap/G4-NoCross模型定义未改。v3/v4训练状态不可由v5恢复；v5新建独立run目录。

## CPU检查

- `py_compile`：活动顶层Python文件及测试文件通过。
- unittest：30项通过，1项跳过，0项失败。跳过项是必须读取外部初始化Checkpoint并做真实权重加载的测试；该约621.7 MB权重未包含在离线包、本机也未暂存，因此不伪称已经核验初始化文件。
- CPU测试覆盖7组模型路由、形状/参数量、DTCWT CPU可微路径、CIF输出、数据划分身份、RNG状态CPU化、probe元组输出/输入类型、按图增强种子和采样顺序、诊断入口默认惰性、断点恢复与对照器合成数据。
- `pkg_resources`弃用警告来自vendored `pytorch_wavelets` 的既有导入，不影响测试结果；本阶段未安装或升级依赖。

完整逐文件修改前后摘要见 `V5_FILE_SHA256_AUDIT.csv`：72项当前活动文件中，9项相对v3/v4快照修改、4项为新增、59项保持不变。完整活动源清单见 `SOURCE_LOCK.v5.json` 与 `SOURCE_LOCK.v5.sha256`。

## 尚未验证

- 本地CPU未加载云端初始化权重，G0/G4的v5初始化兼容性需在阿里云外部权重路径和SHA匹配后验证。
- A10环境锁需在目标实例现场生成并复核；用户此前提供的Python 3.12.12、PyTorch 2.10.0+cu126、CUDA 12.6、A10信息未在本次执行中访问云端重新确认。
- 连续2 Epoch与1+1跨进程一致性实验尚未执行；此时不能宣称RNG恢复在A10上逐位一致。
- GradScaler溢出位置、参数/Adam差异、耗时、显存必须留待获准的A10诊断。
