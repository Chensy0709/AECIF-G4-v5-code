# G4消融实验矩阵

## 分组定义

所有正式组必须使用同一训练入口、同一初始化SHA、数据名单、预处理增强、seed、运行环境、训练预算、loss、Best规则和评估代码。用户提供的A10 30 Epoch结果为早期筛选参考，未独立核验，也不是150 Epoch因果证据。

| 组 | Element融合后 | Corrosion融合后 | 新增参数 | 用途 |
|---|---|---|---:|---|
| G0-new | 无Adapter | 无Adapter | 0 | 同协议AECIF基线 |
| G-E | DTCWT-8 | 无Adapter | 9,873 | “仅Element使用DTCWT”主效应 |
| G-D / 复用G2 | 无Adapter | Spatial-16 | 21,537 | “仅Corrosion使用空间特征”主效应 |
| G4 | DTCWT-8 | Spatial-16 | 31,410 | 双任务优势互补主模型 |
| G4-SS | Spatial-8 | Spatial-16 | 31,410 | 与G4新增参数完全匹配的双空间对照；Spatial-8参数/MAC匹配DTCWT-8 |
| G4-Swap（可选优先级二） | Spatial-16 | DTCWT-8 | 31,410 | 与G4相同模块总量，交换任务路由，检验“模块×任务”匹配 |
| G4-NoCross（必要交互消融） | DTCWT-8 | Spatial-16 | 31,410 | 保留Adapter与预算，移除CIF双向交叉注入，检验CIF交互作用 |
| G4-DonorStopGrad（冲突诊断备选） | DTCWT-8 | Spatial-16 | 31,410 | CIF前向仍双向交互，阻断交叉消息向供体流反传，检验梯度冲突 |

`G4-Swap`把DTCWT放回Corrosion分支、Spatial-16放到Element分支，正面处理“已有G1 DTCWT其实接在Defect流”这一历史事实。若算力只能跑最小矩阵，优先G0、G-E、G-D、G4、G4-SS；如果主张专属路由比反向路由重要，再加G4-Swap。NoCross是最小必要CIF消融；DonorStopGrad只在训练梯度诊断显示冲突后才运行。

## 核心统计对比

每张验证图像保留Element mIoU、Corrosion mIoU、Corrosion foreground IoU/Recall、Boundary F1及有效样本标记；Boundary F1必须继续使用已封版口径。所有对比做同图像配对差值和图像级Bootstrap CI，不能把像素当独立样本。测试集100张保持封存，直到模型、路由、选择规则全部锁定后一次性最终评估。

- `G4 − G4-SS`：在新增参数、主干、训练协议匹配下，检验把DTCWT路由给Element是否比同容量空间专家更有效；Corrosion分支完全相同。
- `G4 − G4-Swap`：模块总量、参数量和任务框架相同，检验路由到目标任务是否重要。
- `G4 − G-E − G-D + G0`：双模块组合的非加性/互补交互对比；分别对Element和Corrosion指标计算。该对比和G4-SS共同控制容量增加。
- `G-E − G0`：仅Element DTCWT的作用。
- `G-D − G0`：仅Corrosion空间Adapter（历史G2路由）的作用。
- `G4 − G4-NoCross`：同参数比较保留CIF交互是否有益。
- `G4 − G4-DonorStopGrad`：同前向交互与参数量比较，判断跨任务反传是否引入负迁移。

主要结论必须同时满足：Corrosion为主目标且无不可接受的Element退步；改善超过配对不确定性/种子波动；相对G4-SS仍有方法特异优势。仅G4高于G0不足以声称DTCWT/空间特征互补，因为可能只是容量增加。

## 多种子和选择

首个相同seed仅作为实现/效能筛选，不能宣称稳定性。进入正式结论的候选至少做3个独立seed（建议5个）；所有组使用相同seed集合与分组顺序。配置、路由和Best规则事先锁定，验证集用于模型选择和CI，不允许不断按验证结果改参数。多组比较应报告效应量、95%图像级CI与seed间离散程度；必要时为多对比校正显著性。

## 单因素原则

- G4与G4-SS只换Element Adapter类型。
- G4与G4-Swap只交换两种Adapter的任务位置。
- G4与G4-NoCross只移除CIF交互。
- G4与G4-DonorStopGrad只改跨任务梯度流。

不在这轮同时改DWA、CE、输入尺寸、训练增强、优化器、精度或分类头。
