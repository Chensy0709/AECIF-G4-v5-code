# G4完整结构设计：Element-DTCWT × Corrosion-Spatial

## 设计假设与数据流证据

AECIF-Net的HRNet-W48输出四尺度特征，插值到相同空间后拼接，分别通过`conv3x3`和`conv3x3_1`形成两路512通道特征。CIF在两个分支中用互相注意力做双向融合，然后分别送入Element与Defect/Corrosion head。

本地源码确认：G1-new的8通道三尺度DTCWT adapter实际接在CIF后的第二路腐蚀特征上；G2的16通道空间残差adapter也接在该第二路腐蚀特征上。G1的Element验证改善是间接效果或共享路径效果的候选解释，并非直接路径证据。G4因此是一个新假设：把已审计的DTCWT模块路由到Element融合特征，同时把已审计空间模块路由到Corrosion融合特征，再验证互补性。

## 主模型 G4

设共享HRNet与两个原有投影输出为 `E,D ∈ R^(B×512×130×130)`（130×130按520输入及HRNet/4高分辨率分支估算，正式实例需核验）。保持原CIF注意力参数和双向信息流：

```text
Image(520²)
    │
 shared HRNet-W48
    ├── multi-scale concat ── conv3x3 ── E (512×130²) ─────┐
    └── multi-scale concat ── conv3x3_1 ─ D (512×130²) ──┐ │
                                                         │ │
       M_E = sigmoid(A_E(E))  ◀──────────────────────────┘ │
       M_D = sigmoid(A_D(D))  ◀────────────────────────────┘
       F_E = E + D ⊙ M_E          F_D = D + E ⊙ M_D
               │                         │
     8ch 3-scale DTCWT adapter    16ch spatial residual adapter
               │                         │
       original Element head        original Corrosion head
               │                         │
       Element logits              Corrosion logits
```

公式：

\[
M_E=\sigma(A_E(E)),\quad M_D=\sigma(A_D(D)),
\]
\[
F_E=E+D\odot M_E,\qquad F_D=D+E\odot M_D,
\]
\[
\hat F_E=F_E+\alpha_E\,\mathcal P_E(\phi(\mathcal Q_E(\mathcal W_{3,6}(\mathcal R_E(F_E))))),
\]
\[
\hat F_D=F_D+\alpha_D\,\mathcal P_D(\phi(K_{3\times3}^{(2)}*\phi(K_{3\times3}^{(1)}*\mathcal R_D(F_D)))).
\]

`W_(3,6)`沿用已核验实现的三尺度、每尺度6方向、near_sym_a/qshift_a、symmetric边界及相位不变幅值；滤波器、方向排列和现有幅值/插值定义不变。`R_E/P_E`是512↔8的1×1投影，方向特征通道数8×6×3=144，经1×1融合到8维后再投影回512。DTCWT运算路径按当前实现用FP32，转换仍保留梯度；其它可autocast的层沿用统一训练器设置。Corrosion专家直接复用G2的SpatialResidualAdapter(16)。

两个alpha均初始化0，DTCWT滤波器固定；这使模块残差初始为0，预期初始logits与同一初始化的G0一致。alpha初始会先获得梯度；卷积适配器权重的梯度乘alpha，第一步可能为0，alpha改变后才开始获得非零梯度。这是预期行为，不应被误判为断图。实际初始输出一致性、alpha第一步梯度与后续适配器梯度必须在获批的独立整网预检验证。

## 参数量与运算量估算

依据本地封版模块的可训练参数统计；以下是新增Adapter参数，不含共享AECIF-Net主体。

| 模块 | Adapter参数 | 130×130每图卷积MAC估算 | 说明 |
|---|---:|---:|---|
| DTCWT-8 | 9,873 | 157,913,600 learned-conv MAC | 不含固定DTCWT滤波/幅值/插值 |
| Spatial-8（双3×3） | 9,873 | 157,913,600 MAC | 与DTCWT-8参数量和learned-conv MAC相同 |
| Spatial-16（G2） | 21,537 | 354,764,800 MAC | 512→16→两层3×3→512 |
| G4 | **31,410** | **512,678,400 MAC + DTCWT固定变换** | 按1 MAC=2 FLOPs，即至少约1.025 GFLOPs/图，不含DTCWT与非卷积运算；B=4为约4.10 GFLOPs+变换 |
| 参数匹配双空间控制 | **31,410** | **512,678,400 MAC** | Spatial-8(Element)+Spatial-16(Corrosion) |

MAC按 `H×W×Cin×Cout×kernel_area` 计算，仅计Adapter卷积。固定滤波器的DTCWT由自定义低层变换实现，常用静态FLOPs工具可能漏计；所以表中G4 FLOPs是卷积下界，不是整网准确FLOPs。正式报告应对固定滤波器卷积按实际滤波长度/尺度补算，并用同一工具测所有组。完整模型总参数尚未重新实例化统计，本阶段不伪造总参数数字。

## 任务冲突：风险与可检验手段

共享HRNet仍由两路loss共同更新。双向CIF使Element loss可沿`D⊙M_E`回传至腐蚀特征和共享骨干，Corrosion loss亦可沿`E⊙M_D`回传。因此“任务专属Adapter”只能隔离各自CIF融合后的直接增强，不能消除共享骨干及双向交互的梯度冲突。

首轮主模型保持原CIF、CE、DWA不变；只增加两路任务Adapter。诊断性短跑（在独立获批后）分别用同一batch计算`g_E=∇L_E`、`g_D=∇L_D`，报告共享HRNet stage/conv1、两个投影、CIF注意力的梯度范数、余弦相似度、负余弦比例以及Adapter alpha/参数梯度。只用训练batch诊断，不根据验证集调路由。

若需检验冲突来源，预注册`G4-DonorStopGrad`消融：保留原CIF前向数值，但在交叉注入项中对提供信息的另一分支做stop-gradient，保持注意力/本任务分支可训练。另设`G4-NoCross`：`F_E=E,F_D=D`，同样保留两Adapter，只移除CIF交叉注入。二者都不是G4主模型，也不在首轮自动启用；用于判断交互梯度/交互信息是否必要。PCGrad、改DWA或增加门控不纳入首轮，避免多个因素混杂。

## 目标函数

保持原双任务损失与DWA：

\[
L=\lambda_E(t)\,CE(\hat Y_E,Y_E)+\lambda_D(t)\,CE(\hat Y_D,Y_D),
\quad \lambda_E+\lambda_D=2.
\]

不新增Dice/Focal、boundary loss、辅助监督或任务专属学习率。这样G4收益可以归因于表示路由/模块，而不是同时改变目标函数。

共享参数冲突指标定义：对固定训练batch，分别计算 `g_E=∇_{θ_shared}L_E` 与 `g_D=∇_{θ_shared}L_D`，并报告

\[
\cos(g_E,g_D)=\frac{g_E^Tg_D}{\|g_E\|_2\|g_D\|_2+\epsilon},\qquad r_{conflict}=\Pr[\cos(g_E,g_D)<0].
\]

按预先选定的共享模块和batch汇总均值/中位数及负余弦比例；不把两路loss已乘DWA后的总梯度误当作独立任务梯度。
