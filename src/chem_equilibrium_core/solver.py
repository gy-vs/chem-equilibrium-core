"""数值内核：在固定温度理想溶液假设下求平衡组成。

关键约定：

* 未知量使用 ``x = ln(c/c°)``，因此任何候选浓度都由指数函数给出，天然严格
  为正——内核里没有、也不允许有“解出来再裁成非负”的步骤；
* 残差方程就是公开的那两组：物料 ``A c = b`` 与独立反应的质量作用定律
  ``S_ind^T ln(c/c°) = ln K``。最终是否算数由
  :mod:`chem_equilibrium_core.verification` 对 *全部* 反应重新判定；
* 两阶段：先在“基物种/非基物种”的 C 维空间稳健求初值，再在完整 N 维空间
  做最小二乘精修，并尝试多个独立初值，取残差最小的候选。

内核失败只抛 :class:`EquilibriumSolveError`，不返回任何“已确认”结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .errors import EquilibriumSolveError
from .network import ChemicalModel

__all__ = ["solve_candidate", "SolveOutcome"]

_BAL_SCALE_FLOOR = 1e-30  # 总量尺度的下限，避免除零


@dataclass(frozen=True)
class SolveOutcome:
    concentrations: np.ndarray
    balance_residual: float
    equilibrium_residual: float
    attempts: tuple[dict, ...] = field(default_factory=tuple)


def _formation_data(model: ChemicalModel):
    """返回约化求解所需的矩阵：

    * ``U``：非基物种关于基物种的形成计量（C×D），形成反应列
      ``[-U[:,k]; e_k]`` 张成配方矩阵 A 的零空间；
    * ``ln_beta``：每个非基物种形成反应的 ln K。
    """

    A = model.formula_matrix
    bidx = np.array(model.basis_species_idx)
    didx = np.array(model.nonbasis_species_idx)
    A_B = A[:, bidx]
    A_D = A[:, didx] if didx.size else np.zeros((model.n_components, 0))
    U = np.linalg.solve(A_B, A_D)  # C x D

    S = model.stoichiometry_matrix
    ind = list(model.independent_reaction_idx)
    S_ind = S[:, ind]
    S_D = S_ind[didx, :] if didx.size else np.zeros((0, len(ind)))
    ln_k = np.array([model.reactions[j].ln_k for j in ind])
    # S_D.T @ ln_beta = ln_k
    if didx.size:
        ln_beta = np.linalg.solve(S_D.T, ln_k)
    else:
        ln_beta = np.zeros(0)
    return bidx, didx, U, ln_beta


def _full_residuals(x, model: ChemicalModel, b: np.ndarray,
                    S_ind: np.ndarray, ln_k_ind: np.ndarray,
                    b_scale: np.ndarray) -> np.ndarray:
    """完整 N 维残差：C 个收支 + r 个独立反应平衡。"""

    c = model.standard_concentration * np.exp(x)
    bal = (model.formula_matrix @ c - b) / b_scale
    eq = S_ind.T @ x - ln_k_ind
    return np.concatenate([bal, eq])


def solve_candidate(
    model: ChemicalModel,
    total_concentrations: np.ndarray,
    *,
    xtol: float = 1e-13,
    max_nfev: int = 20000,
) -> SolveOutcome:
    """求一个高数值精度的候选平衡组成。是否被接纳由复核层决定。"""

    b = np.array(total_concentrations, dtype=float)
    c0 = model.standard_concentration
    n = model.n_species
    ind = list(model.independent_reaction_idx)
    S_ind = model.stoichiometry_matrix[:, ind]
    ln_k_ind = np.array([model.reactions[j].ln_k for j in ind])
    b_scale = np.maximum(np.abs(b), _BAL_SCALE_FLOOR)

    attempts: list[dict] = []

    def assess(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        with np.errstate(over="ignore"):
            c = c0 * np.exp(np.clip(x, np.log(1e-300 / c0),
                                    np.log(1e300 / c0)))
        bal_rel = float(np.max(np.abs(model.formula_matrix @ c - b) / b_scale))
        if S_ind.size:
            eq = float(np.max(np.abs(S_ind.T @ x - ln_k_ind)))
        else:
            eq = 0.0
        return c, bal_rel, eq

    def run_full(x0: np.ndarray, label: str) -> tuple[np.ndarray, float, float] | None:
        # x = ln(c/c°) 边界只用于防止指数上溢（±~690 ln 单位），
        # 远离任何物理浓度；它不是“把负浓度裁成非负”的步骤。
        lo = np.full(n, np.log(1e-300 / c0))
        hi = np.full(n, np.log(1e300 / c0))
        try:
            sol = least_squares(
                _full_residuals, x0,
                args=(model, b, S_ind, ln_k_ind, b_scale),
                method="trf", bounds=(lo, hi), x_scale="jac",
                xtol=xtol, ftol=1e-12, gtol=1e-12,
                max_nfev=max_nfev)
        except Exception as exc:  # 数值异常：换初值
            attempts.append({"start": label, "success": False,
                             "error": f"{type(exc).__name__}: {exc}"})
            return None
        c, bal, eq = assess(sol.x)
        attempts.append({"start": label, "success": bool(sol.success),
                         "balance_residual": bal,
                         "equilibrium_residual": eq,
                         "nfev": int(sol.nfev)})
        return c, bal, eq

    # ---------- 阶段一：约化（C 维）初值 --------------------------------
    bidx, didx, U, ln_beta = _formation_data(model)
    x_reduced: np.ndarray | None = None
    if didx.size == 0:
        # 没有反应自由度，直接线性求解 A c = b
        c = np.linalg.solve(model.formula_matrix, b)
        if np.all(c > 0):
            x_reduced = np.log(c / c0)
    else:
        def reduced_residuals(z):
            w = ln_beta - U.T @ z
            c_b = c0 * np.exp(z)
            c_d = c0 * np.exp(w)
            pred = model.formula_matrix[:, bidx] @ c_b \
                + model.formula_matrix[:, didx] @ c_d
            return (pred - b) / b_scale

        z0 = np.log(np.maximum(b / max(n, 1), 1e-30) / c0)
        lo = np.full(model.n_components, np.log(1e-300 / c0))
        hi = np.full(model.n_components, np.log(1e300 / c0))
        try:
            rsol = least_squares(reduced_residuals, z0, method="trf",
                                 bounds=(lo, hi), x_scale="jac",
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                 max_nfev=max_nfev)
            z = rsol.x
            x = np.empty(n)
            x[bidx] = z
            x[didx] = ln_beta - U.T @ z
            c, bal, eq = assess(x)
            attempts.append({"start": "reduced",
                             "success": bool(rsol.success),
                             "balance_residual": bal,
                             "equilibrium_residual": eq,
                             "nfev": int(rsol.nfev)})
            x_reduced = x
        except Exception as exc:
            attempts.append({"start": "reduced", "success": False,
                             "error": f"{type(exc).__name__}: {exc}"})

    # ---------- 阶段二：完整空间多初值精修 ------------------------------
    starts: list[tuple[str, np.ndarray]] = []
    if x_reduced is not None:
        starts.append(("reduced", x_reduced))

    # 收支可行的启发式：最小范数正解（初值允许启发式取值，不是结果裁剪）
    pinv = np.linalg.pinv(model.formula_matrix)
    c_heur = pinv @ b
    floor = float(np.max(b) * 1e-12 + 1e-30)
    c_heur = np.where(c_heur > 0, c_heur, floor)
    starts.append(("pinv-feasible", np.log(c_heur / c0)))

    # 均匀分配 + 沿平衡方向偏移的几组缩放
    typical = float(np.mean(b[b > 0])) if np.any(b > 0) else 1.0
    for factor in (1.0, 1e-4, 1e4):
        starts.append((f"uniform-{factor:g}",
                       np.full(n, np.log(typical * factor / c0))))

    best: tuple[np.ndarray, float, float] | None = None
    for label, x0 in starts:
        if not np.all(np.isfinite(x0)):
            continue
        outcome = run_full(x0, label)
        if outcome is None:
            continue
        c, bal, eq = outcome
        score = max(bal, eq)
        if best is None or score < max(best[1], best[2]):
            best = (c, bal, eq)

    if best is None:
        raise EquilibriumSolveError(
            "所有数值初值都未能给出候选解", residual=None)

    c, bal, eq = best
    # 这是“候选解精度”门槛；是否通过由 verification 独立复核
    if max(bal, eq) > 1e-7 or not np.all(np.isfinite(c)) \
            or not np.all(c > 0):
        raise EquilibriumSolveError(
            "数值内核未达到候选解精度 "
            f"（收支相对残差 {bal:.3g}，ln 平衡残差 {eq:.3g}）",
            residual=max(bal, eq))
    return SolveOutcome(concentrations=c, balance_residual=bal,
                        equilibrium_residual=eq,
                        attempts=tuple(attempts))
