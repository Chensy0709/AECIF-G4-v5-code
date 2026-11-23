# AECIF-Net G4双任务优势互补设计包

本目录仅含代码审计、结构设计、消融方案及阶段计划。没有修改任何封版代码/配置/权重，没有访问真实验证或测试图像，也没有运行模型或GPU任务。

交付：

- `TRAINER_PROTOCOL_DIFF_AUDIT.md`：旧150轮基线与本地统一训练器差异、证据边界。
- `G4_ARCHITECTURE_DESIGN.md`：按实际数据流设计的G4结构、公式、参数/FLOPs估算和梯度冲突诊断。
- `G4_ABLATION_MATRIX.md`：可归因的任务路由、参数量和CIF交互消融。
- `FP16_REPRODUCTION_RISK_PLAN.md`：原FP16+GradScaler协议风险和验证门槛。
- `PHASED_IMPLEMENTATION_150E_PLAN.md`：审核后实现及150 Epoch实验计划。

云端路径 `/mnt/workspace/AECIF_Kaggle_bundle` 在本地不可访问。用户提供的A10 30 Epoch验证指标只作为背景输入，尚未用云端原始日志/配置独立复核。
