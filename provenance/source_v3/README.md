# AECIF-Net-Wavelet-Corrosion：G4-v1独立实现

本目录是独立版本，含经SHA-256锁定的AECIF-Net骨干/CIF、DTCWT代码及滤波器、空间适配器副本、统一模型工厂、组配置、CPU测试与资源准入文档。旧实验源码、配置、checkpoint、日志和结果均未被修改。没有复制数据图像或标签，100张独立测试集没有被访问。

## 模型构建

```python
from g4_model import build_model

model = build_model("G4", init_seed=20261112)
```

组名：`G0`、`G-E`、`G-D`、`G4`、`G4-SS`、`G4-Swap`、`G4-NoCross`。全部共享一个AECIF-Net构造器；G4保留原HRNet-W48、双投影、双向CIF注意力和两个分类Head。两个适配器均在CIF融合后、对应分类Head前执行，alpha初始为0。

权重初始化按旧`train_200.py`的`module.`前缀移除和键/形状匹配规则执行。唯一源checkpoint为配置中锁定的`best_checkpoint_86.76_PSA_s.pth`；本目录只引用，不复制或修改它。

## CPU验证

在本版本目录运行：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
python parameter_audit.py
python -m unittest discover -s tests -p "test_g4_v1_cpu.py" -v
```

CPU测试会校验模型构建、520输出一致性、DTCWT CPU反向、任务路由、共享初始化和参数配平。完整测试运行记录保存在`reports/`。

## 统一训练器与FP16隔离入口

`train_entry.py --group G4`默认仅显示准备状态。正式训练需单独显式开启`--execute-formal-training`、提供 runtime lock/数据/新 run 目录并键入确认短语；入口只会执行锁定的完整 150 Epoch 协议，不提供隐式短跑选项。

`fp16_isolated_probe.py`默认不实例化模型、不读取图像、不访问 CUDA。获得单独 GPU 探针授权后才允许使用`--execute-gpu-diagnostic`；探针对比 G0 与 G4，最多 3 个训练 batch，且两组顺序运行、使用相同物化输入。它不是训练实验，也不会恢复历史 checkpoint。当前阶段没有运行 GPU 探针或正式训练。

训练器和探针的规则、停止条件、未验证项见`reports/FP16_PROBE_AND_GPU_GATE.md`。CPU训练器测试：`python -m unittest discover -s tests -p "test_unified_trainer_cpu.py" -v`。旧本地 FP16 诊断中第 2 个 batch 出现非有限梯度的证据仍然有效；本实现不假定 FP16 稳定。
