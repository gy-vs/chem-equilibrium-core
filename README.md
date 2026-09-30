# chem-equilibrium-core

固定温度、单液相、理想溶液的反应平衡与批量配液计算内核。**只交付 Python 库**，
不提供网页或命令行界面；调用方在同一进程内传入反应网络与配液过程，拿回可由
外部程序独立核对的平衡组成。

## 设计要点

调用链上的每一层职责分明，数值求解器只负责其中一环：

```
Reaction（物种计量 + K）
   │  build_model：建模阶段检查，不做任何迭代
   ▼
ChemicalModel（不可变；system_id 标识同一个化学体系）
   │  equilibrate_totals / mix_samples / add_component
   ▼
solver（ln(c/c°) 空间多初值求解；严格正浓度，不做裁剪）
   │  候选浓度
   ▼
verification（逐组分物料收支 + 全部反应的质量作用定律，公开复核）
   ▼
EquilibriumResult / Sample（不可变、带溯源；Workspace 只登记复核通过者）
```

1. **物种 ↔ 组分是多对多关系。** 物种通过 `composition` 声明携带哪些组分、
   各携带几份（如 AB2 对 A、B 都负责：`{"A":1, "B":2}`）。物料核对逐组分进行
   `A·c = b`，不可能只让浓度“看起来合理”却吞掉某个组分。
2. **平衡常数写在活度上。** `K = ∏(cᵢ/c°)^νᵢ`，`Reaction(log_k=...)` 是
   自然对数；默认 c° = 1 mol/L。温度固定，不涉及动力学、固体沉淀或联网数据库。
3. **反应重表达不改变真实平衡。** 把整体反应拆成两步、补一条等价反应，只是同一
   体系（`system_id` 相同）的不同写法，计算结果相同。冗余反应的 K 与独立反应
   推导值不一致时，在**建模阶段**以结构化原因拒绝（关联调用方给的反应 id 与
   具体倍数），而不是等到数值迭代报“不收敛”。
4. **配液保留物料与体积。** 样品是不可变对象，携带浓度、体积（因而携带各组分
   物质的量）、T、c°、物种顺序、完整收支复核和来源（`Origin`）。混合按
   `∑Vᵢcᵢ/∑Vᵢ` 汇总物料后重新平衡；直接平均旧浓度只守恒、一般不满足平衡，
   库会明确给出新的平衡组成。旧样品不会被后续混合/补加改写。
5. **成功与否由公开复核决定。** 结果包里有每个组分的 supplied/accounted/残差
   和每条反应（含冗余反应）的 ln K、ln Q、残差与 Q/K 倍数，可序列化为 dict。
   复核不过的候选不会成为样品，也不会进入台账；一次失败计算不会污染此前确认过
   的模型与样品。

## 安装

```bash
pip install -e .
# 运行验证
pip install -e ".[test]" && pytest
# 或无需 pytest 的端到端演示
python examples/verification_demo.py
```

## 快速示例

```python
from chem_equilibrium_core import Reaction, build_model, equilibrate_totals

# 物种携带的组分
composition = {
    "A":   {"A": 1},
    "B":   {"B": 1},
    "AB":  {"A": 1, "B": 1},
    "AB2": {"A": 1, "B": 2},
}
model = build_model(
    composition,
    [Reaction("A+B=AB",   {"AB": 1, "A": -1, "B": -1}, k=10),
     Reaction("AB+B=AB2", {"AB2": 1, "AB": -1, "B": -1}, k=5)],
    temperature=298.15, standard_concentration=1.0,
)

result = equilibrate_totals(model, {"A": 1.0, "B": 1.5}, volume_l=1.0)
result.concentrations            # [0.170094 0.227997 0.387809 0.442097]
result.verification.passed       # True；to_dict() 可交给别的程序核对
result.system_id                 # 同一化学体系的身份哈希
```

等价写法（拆步 / 补整体反应）给出同一组成：

```python
overall = Reaction("A+2B=AB2", {"AB2": 1, "A": -1, "B": -2}, k=50)  # 10*5
m2 = build_model(composition, [r1, overall])     # 与 [r1, r2] 同 system_id
```

若把整体反应的 K 写成 30（与 10×5=50 矛盾）却又保留两条分步反应，
`build_model` 直接抛 `ModelRejectionError`，原因里带反应 id、给定 ln K、
推导 ln K 和相差倍数，模型不会进入任何配液计算。

混合与补加（纯函数，也可用 `Workspace` 登记）：

```python
from chem_equilibrium_core import make_sample, mix_samples, add_component

s1 = make_sample(equilibrate_totals(model, [1.0, 1.5], volume_l=2.0), "s1")
s2 = make_sample(equilibrate_totals(model, [0.5, 3.0], volume_l=1.0), "s2")

mixed = mix_samples([s1, s2], volumes_l=[2.0, 1.0])  # 2:1，重新平衡
mixed.volume_l                    # 3.0
mixed.origin.parents              # ("s1", "s2")

fed = add_component(s1, {"B": 0.5}, final_volume_l=2.0)  # 投 0.5 mol B，定容
# s1 本身不变，仍可继续查询
```

## 建模阶段会拒绝什么

| 原因码 | 含义 |
| --- | --- |
| `UNKNOWN_SPECIES_IN_REACTION` | 反应引用了未定义物种（带反应/物种 id） |
| `REACTION_NOT_CONSERVATIVE` | 反应前后某组分净变化非零（带组分与净变化量） |
| `UNDERDESCRIBED_NETWORK` | 独立反应数不足，存在守恒却不由任何反应生成的组成方向（带涉及物种） |
| `INCONSISTENT_DUPLICATE_REACTION` | 冗余反应的 ln K 与独立反应推导值不一致（带两条 ln K 与倍数） |
| `SPECIES_WITHOUT_COMPONENTS` | 物种不属于任何组分，无法纳入物料核算 |
| `DEPENDENT_COMPONENTS` | 组分之间线性相关，无法分别核对收支 |
| `EMPTY_*` / `DUPLICATE_*` / `BAD_*` | 空集合、重复标识、非正温度或 c° 等 |

## 包结构

```
src/chem_equilibrium_core/
  network.py       反应网络、公式矩阵 A、计量矩阵 S、秩/零空间与 K 一致性检查
  solver.py        ln 空间多初值数值内核（scipy.optimize.least_squares）
  verification.py  物料收支与质量作用定律的公开复核
  sample.py        EquilibriumResult/Sample 与混合、补加纯函数
  workspace.py     线程安全的进程内台账（失败条目不入库）
  units.py         温度/浓度/体积单位的最小支持集
  errors.py        分层异常与结构化拒绝原因
tests/             可运行验证（对照值、混合、补加、重表达、拒绝、并发等）
examples/verification_demo.py  无需 pytest 的端到端验证
```

## 范围限制

固定温度、单液相、理想溶液（活度系数为 1）；不含动力学、沉淀/气体相、压力
效应或热力学数据库查询。所有组分总量必须为正（严格正浓度假设）。
