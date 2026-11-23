# 原始FP16复现风险与验证计划

## 可确认的历史事实

旧训练脚本配置`fp16=True`，创建`torch.cuda.amp.GradScaler`，在CUDA autocast下前向和计算两路CE，然后`scaler.scale(loss).backward()`、`scaler.step(optimizer)`、`scaler.update()`。旧150轮训练状态确有`scaler_state_dict`。只读加载显示：completed_epoch=149（最后Epoch零基索引），Scheduler completed_epoch=150/150，最后LR≈3e-6，Best validation loss约0.22105。状态没有runtime版本字段，历史Python/PyTorch/CUDA仍UNKNOWN。

旧脚本的训练状态保存支持跨Epoch恢复模型、Adam、Scheduler、GradScaler、DWA和RNG。Scaler state存在不等于能证明150轮每步均未跳过更新；旧执行日志没有逐批记录scale/overflow。

## 目前数值风险证据

本机既有隔离诊断报告：在同初始化、同真实训练batch的FP16+GradScaler测试中，G0/G2/G3第2个batch出现非有限梯度，前向和双路loss有限；GradScaler降低scale并跳过该更新，第3批恢复有限。扫描首个异常参数是共享骨干`backbone.conv1.weight`，G0亦有；具体autograd算子及与历史环境的关系UNKNOWN。FP32/BF16只通过每组3个batch，不能替代150 Epoch验证。以上不是A10上的测试，也不能直接外推到云端。

## 复现门槛（先获批后执行）

1. 固定一个实际执行环境，并记录GPU型号、驱动、CUDA runtime/build、Python/PyTorch/cuDNN及依赖；不能把Windows当前版本当历史版本。
2. 在隔离副本运行原始G0 FP16训练入口；使用正式200训练名单，仅用少量训练batch，无验证/测试图像。记录每步loss、GradScaler scale、overflow/step是否跳过、梯度范数和AMP dtype；启用autograd anomaly detection做诊断副本，不改正式loss/标签。
3. 固定相同初始化、输入batch、增强结果，比较旧fit_one_epoch与统一入口移植的FP16路径：前向输出、两路loss、DWA权重、scaled/unscaled梯度、optimizer更新、GradScaler状态。版本迁移造成的容差需预先写定。
4. 若异常只出现在个别batch，先确定其在原训练和所有模型组是否同等出现、Scaler是否按原设计恢复；不能静默跳过或屏蔽有限性监控。预先约定可接受的overflow/跳步规则，再判断是否忠于原协议。
5. 只有FP16训练步稳定且G0/G4共用同一个已审核Trainer后，才进入获批的短程端到端检查。然后G0、G-E、G-D、G4、G4-SS以同一实际环境、同一FP16+Scaler、同一初始化重跑150 Epoch。

## 如果原FP16协议不能稳定复现

停止正式对比并提交证据：触发层/算子、batch身份、Scaler变化、更新是否跳过、环境与复现率。随后单独提出兼容方案：例如在同一A10环境将全部实验组和新G0统一改为BF16并重跑；这属于协议变更，需用户确认，不能把历史FP16 G0或用户提供的BF16 30轮结果当作新150轮严格对照。不得私自改为FP16以外精度、改loss、裁剪梯度或改学习率。

本阶段未运行GPU/模型，也没有开始FP16复现。
