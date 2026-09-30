"""平衡结果的公开复核。

这里不依赖求解器，只根据调用方同样可见的信息核对两件事：

1. 物料收支：``A @ c == b``，每个组分逐条核对——物种必须为它所属的每一个
   组分的总量负责，不能“看起来合理”却悄悄吞掉某个组分；
2. 化学平衡：对 *每一条* 调用方给出的反应（包括冗余/等价反应），
   ``ln Q_r == ln K_r``，其中 ``Q_r = prod_i (c_i/c°)^nu_ir``。

另一个程序拿到 :class:`VerificationReport` 后可以自己复查，不必信任本库的
``passed`` 标志。任何候选解只要复核不过，都不会被包装成已确认样品。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import VerificationError
from .network import ChemicalModel

__all__ = [
    "BalanceRecord",
    "EquilibriumRecord",
    "VerificationReport",
    "verify_composition",
]


@dataclass(frozen=True)
class BalanceRecord:
    """一个组分的物料收支核对项。"""

    component: str
    supplied: float      # 配液投入换算出的总浓度 b_j
    accounted: float     # 由物种浓度汇总的 (A @ c)_j
    residual: float      # accounted - supplied

    @property
    def relative_residual(self) -> float:
        scale = max(1.0, abs(self.supplied))
        return abs(self.residual) / scale


@dataclass(frozen=True)
class EquilibriumRecord:
    """一条反应的化学平衡核对项（含冗余反应）。"""

    reaction_id: str
    ln_k: float
    ln_quotient: float
    residual: float                      # ln Q - ln K

    @property
    def quotient(self) -> float:
        return float(np.exp(self.ln_quotient))

    @property
    def factor_residual(self) -> float:
        """Q 与 K 相差的倍数（>=1）。"""

        return float(np.exp(abs(self.residual)))


@dataclass(frozen=True)
class VerificationReport:
    """一次公开复核的全部数据，可序列化为 dict 交给外部程序。"""

    passed: bool
    concentration_residual: float   # 若存在负浓度：最负者；否则 0
    balance_records: tuple[BalanceRecord, ...]
    equilibrium_records: tuple[EquilibriumRecord, ...]
    balance_tol: float
    equilibrium_tol: float
    min_concentration: float

    @property
    def max_balance_residual(self) -> float:
        return max((r.relative_residual for r in self.balance_records),
                   default=0.0)

    @property
    def max_equilibrium_residual(self) -> float:
        return max((abs(r.residual) for r in self.equilibrium_records),
                   default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "min_concentration": self.min_concentration,
            "concentration_residual": self.concentration_residual,
            "max_balance_relative_residual": self.max_balance_residual,
            "max_equilibrium_ln_residual": self.max_equilibrium_residual,
            "balance": [
                {"component": r.component, "supplied": r.supplied,
                 "accounted": r.accounted, "residual": r.residual,
                 "relative_residual": r.relative_residual}
                for r in self.balance_records],
            "equilibrium": [
                {"reaction": r.reaction_id, "ln_k": r.ln_k,
                 "ln_quotient": r.ln_quotient, "residual": r.residual,
                 "factor_residual": r.factor_residual}
                for r in self.equilibrium_records],
            "tolerances": {"balance": self.balance_tol,
                           "equilibrium": self.equilibrium_tol},
        }

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        problems: list[str] = []
        if self.concentration_residual < 0:
            problems.append(
                f"存在负浓度，最小值 {self.min_concentration:.3g}（平衡组成"
                "必须严格为正，禁止把负值裁剪为 0 冒充解）")
        bad_bal = [r for r in self.balance_records
                   if r.relative_residual > self.balance_tol]
        for r in bad_bal:
            problems.append(
                f"组分 {r.component!r} 收支不闭合：投入 {r.supplied:.6g}，"
                f"物种汇总 {r.accounted:.6g}，残差 {r.residual:+.3g}")
        bad_eq = [r for r in self.equilibrium_records
                  if abs(r.residual) > self.equilibrium_tol]
        for r in bad_eq:
            problems.append(
                f"反应 {r.reaction_id!r} 不满足平衡："
                f"ln K={r.ln_k:.6g}，ln Q={r.ln_quotient:.6g}，"
                f"残差 {r.residual:+.3g}（Q/K 相差 {r.factor_residual:.4g} 倍）")
        raise VerificationError(
            "候选组成未通过公开复核：\n  - " + "\n  - ".join(problems),
            report=self)


def verify_composition(
    model: ChemicalModel,
    concentrations: np.ndarray,
    total_concentrations: np.ndarray,
    *,
    balance_tol: float = 1e-8,
    equilibrium_tol: float = 1e-8,
) -> VerificationReport:
    """对给定候选浓度执行物料与平衡双重复核。

    ``balance_tol`` 是相对物料总量的无量纲容差；
    ``equilibrium_tol`` 是 ln Q − ln K 的绝对容差。
    这里不做任何裁剪或修正，只如实记录残差。
    """

    c = np.asarray(concentrations, dtype=float)
    b = np.asarray(total_concentrations, dtype=float)
    if c.shape != (model.n_species,):
        raise ValueError(
            f"浓度向量长度应为 {model.n_species}，收到 {c.shape}")
    if b.shape != (model.n_components,):
        raise ValueError(
            f"总浓度向量长度应为 {model.n_components}，收到 {b.shape}")

    min_c = float(np.min(c)) if c.size else 0.0
    neg = min(0.0, min_c)

    accounted = model.formula_matrix @ c
    balance_records = tuple(
        BalanceRecord(component=model.components[j],
                      supplied=float(b[j]),
                      accounted=float(accounted[j]),
                      residual=float(accounted[j] - b[j]))
        for j in range(model.n_components))

    c0 = model.standard_concentration
    S = model.stoichiometry_matrix
    ln_a = np.full_like(c, -np.inf)
    positive = c > 0
    ln_a[positive] = np.log(c[positive] / c0)
    eq_records: list[EquilibriumRecord] = []
    for rr, rxn in enumerate(model.reactions):
        col = S[:, rr]
        if np.all(positive):
            ln_q = float(np.dot(col, ln_a))
        else:
            # 非正浓度本来就不合法；仍如实给出 Q 的 0/inf 倾向，
            # 让失败报告包含完整信息。
            active = col != 0.0
            if np.any(~positive & active):
                if np.any(col[~positive] < 0):
                    ln_q = np.inf    # 反应物缺失/为负 -> Q 为无穷大
                else:
                    ln_q = -np.inf   # 产物缺失/为负 -> Q 为 0
            else:
                ln_q = float(np.dot(col[active], ln_a[active]))
        eq_records.append(EquilibriumRecord(
            reaction_id=rxn.id, ln_k=rxn.ln_k,
            ln_quotient=ln_q, residual=float(ln_q - rxn.ln_k)))

    passed = (
        neg == 0.0
        and all(r.relative_residual <= balance_tol for r in balance_records)
        and all(abs(r.residual) <= equilibrium_tol for r in eq_records)
    )
    return VerificationReport(
        passed=passed,
        concentration_residual=float(neg),
        balance_records=balance_records,
        equilibrium_records=tuple(eq_records),
        balance_tol=balance_tol,
        equilibrium_tol=equilibrium_tol,
        min_concentration=min_c,
    )
