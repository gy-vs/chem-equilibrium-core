# chem-equilibrium-core

批量配液计算用的化学平衡 **Python 内核**：固定温度、单一液相、理想溶液。
不含网页或命令行界面；只提供库调用链。运行期依赖仅 NumPy。

## 设计要点

* **组分（component）与物种（species）分离**：组分是独立守恒的投料要素；
  物种携带若干组分（组成矩阵 `C` 的一行）。一个物种属于几个组分，就同时
  计入这几个组分的物料收支，不可能“浓度看着合理却悄悄吃掉某个组分”。
* **反应描述**：每个反应是化学计量矩阵 `N` 的一行（产物为正、反应物为负），
  配无量纲平衡常数 `K`（以 `c_standard` 为标准浓度，活度为 `c_i / c_standard`）。
* **模型在被使用前完成接纳检查**，失败抛出带有反应/组分/物种名称的
  `ModelError`，而不是把问题留到数值迭代后叫“不收敛”：
  1. 命名与引用合法；
  2. 组分线性无关（含“某组分没有任何物种携带”的情况）；
  3. 每个反应守恒每个组分：`N C = 0`；
  4. `K` 为正有限值；
  5. 平衡常数热力学一致——被其他反应推出的反应必须带推出的 `K`；
  6. 反应张成所有“组成守恒的物种变换方向”，不存在无法确定平衡的物种。
* **重新表达反应不改变体系**：把整体反应拆成两步、补充逆反应或等价反应，
  只改变描述而不改变平衡。网络指纹（温度、标准浓度、物种/组分顺序、
  组成矩阵、规范不变的最小范数标准自由能投影）相同即判定为同一化学体系。
  等价反应的 `K` 与已有反应矛盾时，在建模阶段就被拒绝并指明“由哪几条反应
  推出、隐含的 K 是多少、声明的 K 是多少”。
* **结果要同时通过两项公开核对**才算成功（`verify.assess`，独立于求解器）：
  每个组分的物料收支闭合；每条反应满足 `Σ ν ln(c_i/c0) = ln K`
  （零投料组分下被阻断的反应按 KKT 边界不等式确认）。浓度由
  `c = c0·exp(Cλ−g)` 参数化，**结构上恒正，不做任何非负裁剪**。
* **样品不可变，携带物料量（mol）与体积（L）**。混合按取用量相加物种的量，
  组分总量精确守恒，再重新平衡；直接体积加权平均虽然组分仍守恒，但不满足
  平衡，会被公开核对判定失败。混合、补加（`spike_component`，可改终体积）
  都返回新样品，旧样品永不改变，并且新样品记录完整来源（输入样品 id、
  取用量、操作链），可沿谱系回溯。
* **一次失败不污染任何缓存**：只有通过公开核对的样品才登记进 `Lineage`；
  失败时旧模型、旧样品仍可继续使用。
* **同一进程可并行计算不同网络**：`Workspace`/`Lineage` 是显式对象，不使用
  任何全局状态；跨网络混合会被带指纹信息的 `InputError` 拒绝。

## 安装

```sh
pip install -e .      # 仅需 numpy>=1.21
```

或直接将 `src/` 放入 `PYTHONPATH`（Python ≥ 3.9）。

## 快速使用

```python
from chem_equilibrium_core import NetworkBuilder, Workspace

network = (
    NetworkBuilder(temperature=298.15, c_standard=1.0)   # 固定温度、标准浓度 mol/L
    .add_species("A", {"A": 1})
    .add_species("B", {"B": 1})
    .add_species("AB", {"A": 1, "B": 1})                 # 物种同时携带 A 与 B
    .add_species("AB2", {"A": 1, "B": 2})
    .add_reaction("AB_fwd", {"A": -1, "B": -1, "AB": 1}, K=10.0)
    .add_reaction("AB2_fwd", {"AB": -1, "B": -1, "AB2": 1}, K=5.0)
    .build()                                             # 模型接纳检查在此发生
)

ws = Workspace("batch")

# 直接由组分总浓度求平衡（返回带收支与核对报告的 EquilibriumState）
state = ws.equilibrate(network, {"A": 1.0, "B": 1.5})
state.concentrations          # (0.170094, 0.227997, 0.387809, 0.442097)
state.species_order           # ('A', 'B', 'AB', 'AB2')
state.component_residuals()   # 各组分公开收支残差（mol/L）
state.reaction_residuals()    # 各反应 ln(Q/K) 残差

# 配液：投料以“纯组分的 mol 数”描述，体积为 L
s1 = ws.prepare_sample(network, volume=1.0, feed={"A": 1.0, "B": 1.5})
s2 = ws.prepare_sample(network, volume=1.0, feed={"A": 0.5, "B": 3.0})

mix = ws.mix_samples([(s1, 2.0), (s2, 1.0)])   # 2:1 混合后重新平衡
spiked = ws.spike_component(mix, component="B", amount=0.5, final_volume=3.5)

mix.parent_ids                 # 新样品来自哪几份输入
ws.chain(spiked.sample_id)     # 沿操作链回溯全部祖先样品
```

### 反应网络随方案演进

```python
from chem_equilibrium_core import evolve_network

# 补充一条整体反应：K 必须等于 10*5=50，否则在 build() 阶段被拒绝
plus_overall = evolve_network(
    network, add_reactions=[("overall", {"A": -1, "B": -2, "AB2": 1}, 50.0)]
)
plus_overall.fingerprint == network.fingerprint   # True：同一化学体系
```

### 独立核对任意组成

```python
from chem_equilibrium_core import assess

report = assess(network, concentrations, total_concentrations)
report.ok                       # 物料收支与平衡同时满足
report.component_residuals      # {'A': ..., 'B': ...}
report.reaction_residuals       # {'AB_fwd': ..., 'AB2_fwd': ...}
report.violations               # 人类可读的不满足原因
```

## 目录结构

```
src/chem_equilibrium_core/
  errors.py      # ModelError / InputError / EquilibriumNotConfirmed（带反应与组分名）
  model.py       # 物种/组分/反应描述、网络接纳检查、体系指纹、网络演进
  engine.py      # 对偶变量阻尼牛顿（理想溶液最小自由能），失败不返回半成品
  verify.py      # 公开、独立于求解器的物料收支与平衡核对
  state.py       # EquilibriumState（浓度、顺序、温度、标准浓度、收支报告）
  samples.py     # Sample/Lineage：投料、混合、补加、不可变性与来源链
  workspace.py   # Workspace 外观：网络登记（指纹去重）、一致调用链
examples/
  verify_delivery.py   # 可运行的端到端交付验证（混合/补加/重表达/收支）
tests/                 # 标准库 unittest 套件（52 个测试，含极端投料尺度回归）
```

## 运行验证

```sh
sh run_checks.sh
# 或分步：
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 examples/verify_delivery.py
```

## 范围与非目标

当前范围：固定温度、单一液相、理想溶液（活度系数为 1）。

明确不包含：反应动力学、固体沉淀/多相平衡、非理想活度模型、在线热力学
数据库查询、网页与命令行界面。
