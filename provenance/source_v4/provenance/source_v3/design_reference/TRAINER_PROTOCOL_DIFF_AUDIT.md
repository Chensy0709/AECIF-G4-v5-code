# 原150轮基线与统一训练器差异审计

## 范围与结论

只读检查了 `基线实验（200张训练集）/train_200.py`、其依赖的 `AECIF-Net` 源码、旧150轮状态checkpoint，以及本地 Kaggle 迁移包中的四组统一训练器/配置。没有修改原文件，没有访问验证或测试图像，没有运行训练、模型前向/反向或GPU。

`/mnt/workspace/AECIF_Kaggle_bundle` 在当前本机不可访问。下文“统一训练器”指本地迁移包中的代码，不声称已独立检查云端实际执行版本。用户给出的30 Epoch A10验证指标记录为用户提供、未独立复核，不能替代150 Epoch结论。

## 原始200/34、150 Epoch实际配置

| 项目 | 代码/状态证据 | 核验结果 |
|---|---|---|
| 数据划分 | `train_200.py`读取 `train_200.txt` 和 `val.txt`；既有数据名单审计 | 200/34；名单SHA与本地统一配置一致 |
| 输入 | `input_shape=[520,520]` | 520×520 |
| Batch | `Freeze_Train=False`、`Unfreeze_batch_size=4`、`drop_last=True` | 实际Batch=4；每训练Epoch 50步；200可整除4，无训练尾样本丢弃 |
| 骨干/双路 | `AECIF_Net(backbone='hrnetv2_w48')`；HRNet四级特征拼接后分别进入 `conv3x3` 和 `conv3x3_1` | 共享HRNet-W48，两个512通道任务特征流，CIF双向交互 |
| Optimizer | `optim.Adam(..., betas=(0.9,0.999), weight_decay=0)` | Adam；无权重衰减 |
| LR | `Init_lr=5e-4`, `Min_lr=5e-6`; batch比例4/16；Adam限值 | 调度最大3e-4、最低3e-6；Cosine helper包含3轮warmup，index 0从3e-5开始，之后升至3e-4，最后15轮到3e-6 |
| Scheduler | 原始和统一版调用同一 `nets/hrnet_training.py` 的Cosine函数；源码哈希一致 | total=150、warmup最多3、no-aug tail最多15；旧状态记录末LR约3e-6 |
| Loss | 两路 `CE_Loss`；类别权重全1；`dice_loss=False`, `focal_loss=False` | Element 7类CE + Corrosion/Defect 2类CE；ignore index分别7/2 |
| DWA | 前两轮等权；之后按两任务前两次历史loss比率，温度2，归一化权重和为2 | 新版使用减最大值的稳定softmax，数学公式等价；新版给历史loss分母加1e-12保护 |
| 增强/验证 | 新旧均引用完全同哈希的 `utils/dataloader_multi.py` | 增强变换定义一致；验证是等比例缩放/灰色填充，不做随机增强 |
| 初始化 | `pretrained=False`，权重初始化后部分匹配加载 `best_checkpoint_86.76_PSA_s.pth`；四组配置引用同一文件SHA | 不是随机从零训练，也不是HRNet ImageNet `pretrained=True`；基础权重路径、大小和SHA由配置锁定 |
| 最优模型 | 原始 `validation_loss <= best_val_loss` 更新best | 新版锁定为weighted validation total CE最小，平局取较晚Epoch；与原比较方向/平局处理一致 |
| Checkpoint进度 | 只读 `training_state_latest.pth` | `completed_epoch=149`是零基最后Epoch索引；Scheduler `completed_epoch=150`、`total_epochs=150`、`last_lr≈3e-6`，表示150轮已完成 |

实际LR需要准确表述为“最大3e-4、最低3e-6，warmup首个索引从3e-5开始”，不能把首轮实际LR写成3e-4。

旧状态含 `scaler_state_dict`（含scale/growth/backoff字段）及模型、Adam、Scheduler、DWA、RNG状态；状态载荷没有Python、PyTorch、CUDA、cuDNN版本字段。因此历史运行版本仍是UNKNOWN。当前机器的Python/PyTorch版本不能反推为历史版本。旧历史CSV标注为 `reconstructed_canonical`，不可当作原始逐批训练日志。

## AMP/FP16与云端BF16的真实差异

| 方面 | 旧200/34基线 | 本地统一四组训练器 | 研究影响 |
|---|---|---|---|
| 精度 | `fp16=True`，`torch.cuda.amp.autocast()` + `GradScaler.scale/backward/step/update` | `torch.autocast('cuda', dtype=torch.bfloat16)`；没有GradScaler；每步检查有限性 | 数值路径不同，不是同协议复跑 |
| 验证 | 原始验证前向在autocast外 | 新版验证同样在autocast外、使用`inference_mode` | 基本目标相同，no_grad与inference_mode实现不同 |
| BatchNorm | 普通BatchNorm，无SyncBN（`distributed=False`） | 普通BatchNorm | 单卡上目标一致；DataParallel多卡下BN行为/有效批次可能不同 |
| 数据顺序 | DataLoader `shuffle=True`；没有训练起始统一seed | epoch派生CPU randperm，seed=20261112；四组顺序固定一致 | 新版更可复现，但不重现旧随机顺序 |
| 增强随机数 | 同一全局NumPy随机流，训练运行未显式固定seed | 按 seed/epoch/sample ID隔离NumPy增强流 | 增强操作和分布保持，具体增强实现/序列不同；组间配对更公平 |
| GPU使用 | `distributed=False`；普通路径包装`DataParallel(model)`，可使用所有可见卡 | 明确 `device='cuda:0'`，无DP/DDP | 多卡运行时旧版可能跨可见卡，新版只用第一卡；单卡时差异较小 |
| Grad清零 | `optimizer.zero_grad()` | `zero_grad(set_to_none=True)` | 更新公式不变，内存/执行细节不同 |
| 验证节奏 | 每Epoch验证CE；EvalCallback的额外指标周期性（默认脚本设置） | 每Epoch输出逐图与汇总指标 | 记录粒度不同；Best仍以验证loss为主 |
| 输出/恢复 | latest/previous完整训练状态与Scaler；旧兼容恢复入口 | HEAD/manifest事务，Epoch边界恢复；BF16明确Scaler不适用 | 恢复可靠性实现不同；历史状态不能直接用于新Trainer的BF16实验 |

训练增强、验证预处理、模型定义、CE与Scheduler依赖中，旧源码和迁移包副本SHA相同（AECIF_Net、backbone、hrnet_training、dataloader_multi、utils_fit_DWA）。统一训练器额外增加确定性Sampler/按样本增强种子和四组适配器。

## FP16稳定性证据的边界

本地已有的独立数值诊断显示：G0/G2/G3在当前诊断环境的FP16+GradScaler第2个真实Batch出现非有限梯度；Scaler降scale并跳过该步，首个按参数扫描顺序发现异常的是共享HRNet的`backbone.conv1.weight`，G0也出现，所以不是DTCWT专属。具体autograd起因UNKNOWN，且诊断环境不等同于历史运行环境或云端A10。它证明需要复核，不证明原150轮作业必然失败。

旧150轮作业有完成记录和Scaler状态，证明代码按FP16路径运行并完成了150 Epoch；没有逐步Scaler日志，不能证明全程从未跳步或完全确定性复现。

## G1/G2实际数据流核对（重要）

- 本地统一代码的G1-new调用 `DTCWTCIF`，在双向CIF融合后把 `x_1_fuse`（第二路Defect/Corrosion特征）送入DTCWT adapter，再进 `last_layer_d`。Element logits来自未经过DTCWT adapter的 `x_fuse -> last_layer_e`。
- G2调用 `SpatialCIF`，在同一第二路 `x_1 + x * attention_1(x_1)` 特征上接SpatialResidualAdapter，再进Defect head。Element head不经过这个Adapter。
- 所以用户提供的30轮G1 Element mIoU增加，不能归因为G1直接在Element支路做了DTCWT增强；它可能经共享HRNet和CIF反馈间接影响。G2 Corrosion提升与空间适配器直接作用于Corrosion流一致，但这仍是待正式同环境对照的解释。

## 证据清单

- 旧配置：`基线实验（200张训练集）/train_200.py`、`AECIF-Net/nets/hrnet_training.py`、`AECIF-Net/utils/utils_fit_DWA.py`。
- 旧checkpoint：`logs_200_150epochs/training_state_latest.pth`只读元数据：completed_epoch 149；scheduler completed 150/150，last LR≈3e-6；保存FP16 GradScaler state；无runtime version字段。
- 新协议：本地迁移包 `remediation/unified_training/unified_trainer.py` 和四组JSON；所有组声明Python3.12/PyTorch2.10/CUDA12.6/BF16/Batch4/520/150Epoch。
- 云端A10 30Epoch结果：仅来自用户本轮提供的表格，未连接`/mnt/workspace`，未检查原始运行日志、checkpoint或设备元数据。

本报告是设计依据，不授权代码修改或训练。

## 本次静态检查文件SHA-256

| 文件 | SHA-256 |
|---|---|
| `基线实验（200张训练集）/train_200.py` | `edd3cdf0b8722ca4212b50fa427ac884a2515dbc79534630fb3b9b691fee31bd` |
| `AECIF-Net/utils/utils_fit_DWA.py` | `5b74c0f30f5f4117240ca64d0c10e62aac6e9bec80d75435123edc1d2a382451` |
| `AECIF-Net/nets/AECIF_Net.py` | `32c12a0f23b906ec98f32278376655cdec2b9d7262e41ba9f014c0c58a049626` |
| `AECIF-Net/nets/backbone.py` | `b2d1e8f4641430e79316b7044a48112bc954fa086f04b3d4a4cac775485ff5a2` |
| `AECIF-Net/nets/hrnet_training.py` | `43a91fa76a94e7bf9b3d3b6f47229c3d539d5ce3a3ff74766bf521e310c8c0ab` |
| `AECIF-Net/utils/dataloader_multi.py` | `c1027c3c5eb278b8320f8d56286fe5a2a66963394a9db8fb8f553faa522e21d3` |
| local bundle `unified_trainer.py` | `b2fb6d9de4034faf76443441e5fe5e068142622047d266e3f675ce45703f4a66` |
| local bundle `aecif_dtcwt.py` | `0f545514d89d88348482e3ea728ecefab46ea9173f704af0908dd2deaf4f6f7d` |
| local bundle `dtcwt_adapter.py` | `15f8ef94c433ccb8198137e10ef1e18b4e3ec20f305c9ca0c0c91d9ab32b5eba` |
| local bundle `matched_spatial_adapter.py` | `bcb536b019ba4aeb15d43aeeff3a5fe97159b4a4bfb7929e745328aa33c170f0` |

同路径旧数据管线、CE/Scheduler、AECIF主体和backbone文件与迁移包`baseline_code`副本哈希相同。以上只是审计时文件指纹，不是云端执行源码指纹；云端部署版本仍须独立比对manifest。
