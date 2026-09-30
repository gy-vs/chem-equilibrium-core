#!/usr/bin/env python3
"""chem_equilibrium-core 端到端验证脚本（无需 pytest，直接 python 运行）。

覆盖用户提出的全部关键主张：

1. 复现理想溶液对照值（A、B、AB、AB2；K=10、5；b_A=1、b_B=1.5 mol/L）；
2. 两份样品 2:1 混合后重新平衡——朴素平均只守恒、不平衡，库结果两者都满足；
3. 补加组分并改变体积后重新平衡，旧样品不被改写且溯源完整；
4. 把整体反应拆成两步 / 补一条等价反应：组成不变，system_id 相同；
5. 冗余反应常数互相矛盾时，建模阶段即以结构化原因拒绝（不进入数值迭代）；
6. 结果公开携带 T、c°、物种顺序与逐组分收支，可供外部程序核对；
7. 一次模拟的求解失败不污染台账，旧样品继续可查。
"""

from __future__ import annotations

import sys

import numpy as np

from chem_equilibrium_core import (EquilibriumSolveError, ModelRejectionError,
                                   Reaction, Workspace, build_model,
                                   equilibrate_totals, make_sample, mix_samples,
                                   add_component, verify_composition)
from chem_equilibrium_core import sample as sample_module

COMPOSITION = {
    "A": {"A": 1},
    "B": {"B": 1},
    "AB": {"A": 1, "B": 1},
    "AB2": {"A": 1, "B": 2},
}
R1 = Reaction("A+B=AB", {"AB": 1, "A": -1, "B": -1}, k=10.0)
R2 = Reaction("AB+B=AB2", {"AB2": 1, "AB": -1, "B": -1}, k=5.0)
ROV = Reaction("A+2B=AB2", {"AB2": 1, "A": -1, "B": -2}, k=50.0)
REFERENCE = np.array([0.170094, 0.227997, 0.387809, 0.442097])

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def independent_check(c, bA, bB, k1, k2, c0=1.0) -> bool:
    A_, B_, AB, AB2 = c
    return all([
        abs(A_ + AB + AB2 - bA) < 1e-8,
        abs(B_ + AB + 2 * AB2 - bB) < 1e-8,
        abs(AB / (A_ * B_) / (k1 / c0) - 1) < 1e-6,
        abs(AB2 / (AB * B_) / (k2 / c0) - 1) < 1e-6,
    ])


def main() -> int:
    ws = Workspace()

    section("1. 对照计算")
    model = build_model(COMPOSITION, [R1, R2], temperature=298.15)
    ws.add_model(model)
    res = equilibrate_totals(model, {"A": 1.0, "B": 1.5}, volume_l=1.0)
    print("  物种顺序:", res.species)
    print("  平衡浓度:", np.round(res.concentrations, 6).tolist())
    print("  对照值:  ", REFERENCE.tolist())
    check("四个平衡浓度与对照值一致 (|Δ|<5e-7)",
          np.allclose(res.concentrations, REFERENCE, atol=5e-7))
    check("公开复核通过", res.verification.passed)
    for rec in res.verification.balance_records:
        check(f"组分 {rec.component} 收支闭合 "
              f"(投入 {rec.supplied:g} = 汇总 {rec.accounted:.9g})",
              rec.relative_residual < 1e-10)
    check("结果携带 T=298.15 K 与 c°=1 mol/L",
          res.temperature_k == 298.15 and res.standard_concentration == 1.0)

    section("2. 2:1 混合后的重新平衡")
    s1 = make_sample(equilibrate_totals(model, [1.0, 1.5], volume_l=2.0), "s1")
    s2 = make_sample(equilibrate_totals(model, [0.5, 3.0], volume_l=1.0), "s2")
    mixed = mix_samples([s1, s2], volumes_l=[2.0, 1.0])
    bA, bB = (2 * 1.0 + 0.5) / 3, (2 * 1.5 + 3.0) / 3
    naive = (2 * s1.concentrations + s2.concentrations) / 3
    naive_report = verify_composition(model, naive, model.formula_matrix @ naive)
    print("  混合后总浓度 b_A, b_B =", round(bA, 6), round(bB, 6))
    print("  朴素平均:", np.round(naive, 6).tolist())
    print("  重新平衡:", np.round(mixed.concentrations, 6).tolist())
    check("朴素平均守恒但不满足平衡",
          np.allclose(model.formula_matrix @ naive, [bA, bB])
          and not naive_report.passed
          and naive_report.max_equilibrium_residual > 1e-3,
          f"ln 平衡残差 {naive_report.max_equilibrium_residual:.3f}")
    check("库结果同时满足收支与平衡", mixed.verification.passed)
    check("库结果与朴素平均不同",
          np.max(np.abs(mixed.concentrations - naive)) > 1e-4)
    check("独立复核（不用库的 passed）",
          independent_check(mixed.concentrations, bA, bB, 10, 5))
    check("混合体积 = 3 L 且来源为 s1,s2",
          mixed.volume_l == 3.0 and mixed.origin.parents == ("s1", "s2"))

    section("3. 补加组分 + 改变最终体积")
    fed = add_component(s1, {"B": 0.5}, final_volume_l=2.0)
    fA, fB = 1.0, (2 * 1.5 + 0.5) / 2  # s1 是 2 L 样品
    print("  补加后总浓度:", np.round(fed.total_concentrations, 6).tolist())
    check("补加后的物料账正确",
          np.allclose(fed.total_concentrations, [fA, fB]))
    check("补加后重新平衡通过", fed.verification.passed)
    check("旧样品 s1 未被改写（浓度与体积）",
          np.allclose(s1.concentrations, res.concentrations)
          and s1.volume_l == 2.0)
    check("新样品记录来源 s1", fed.origin.parents == ("s1",))

    section("4. 反应重新表达不改变真实平衡")
    m_step = build_model(COMPOSITION, [R1, R2])
    m_plus = build_model(COMPOSITION, [R1, R2, ROV])
    m_alt = build_model(COMPOSITION, [R1, ROV])
    c_step = equilibrate_totals(m_step, [1.0, 1.5]).concentrations
    c_plus = equilibrate_totals(m_plus, [1.0, 1.5]).concentrations
    c_alt = equilibrate_totals(m_alt, [1.0, 1.5]).concentrations
    check("三种写法共享同一 system_id",
          m_step.system_id == m_plus.system_id == m_alt.system_id)
    check("补等价反应 / 换等价基后组成不变",
          np.allclose(c_step, c_plus, atol=1e-10)
          and np.allclose(c_step, c_alt, atol=1e-10))

    section("5. 不一致的化学条件在建模阶段被拒绝")
    bad = Reaction("A+2B=AB2(矛盾)", {"AB2": 1, "A": -1, "B": -2}, k=30.0)
    try:
        build_model(COMPOSITION, [R1, R2, bad])
        check("矛盾冗余反应被拒绝", False, "竟然被接纳了")
    except ModelRejectionError as e:
        reason = e.reasons[0]
        ok = (reason.code == "INCONSISTENT_DUPLICATE_REACTION"
              and "A+2B=AB2(矛盾)" in reason.reactions
              and abs(reason.details["factor"] - 50 / 30) < 1e-9)
        check("结构化拒绝（原因码 + 反应标识 + 倍数）", ok,
              f"K 相差 {reason.details['factor']:.4g} 倍")
    try:
        build_model(COMPOSITION, [ROV])  # 只写整体反应，欠描述
        check("欠描述网络被拒绝", False, "竟然被接纳了")
    except ModelRejectionError as e:
        check("欠描述网络被拒绝并点名涉及物种",
              any(r.code == "UNDERDESCRIBED_NETWORK" for r in e.reasons)
              and any("AB" in r.species for r in e.reasons))

    section("6. 失败计算不污染台账")
    base = ws.equilibrate(model, [1.0, 1.5], sample_id="base")
    n_before = len(ws.samples())
    original_solver = sample_module.solve_candidate

    def broken_solver(*a, **k):
        raise EquilibriumSolveError("模拟内核失败")

    sample_module.solve_candidate = broken_solver
    polluted = False
    try:
        ws.equilibrate(model, [0.9, 2.0], sample_id="ghost")
        polluted = True
    except EquilibriumSolveError:
        pass
    finally:
        sample_module.solve_candidate = original_solver
    check("失败后台账条目数不变且无 ghost 条目",
          not polluted and len(ws.samples()) == n_before
          and not ws.has_sample("ghost"))
    check("旧样品 base 仍可查询且通过复核",
          ws.get_sample("base") is base and base.verification.passed)

    print("\n" + "=" * 48)
    if _failures:
        print(f"验证失败 {len(_failures)} 项: {_failures}")
        return 1
    print("全部验证通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
