"""反应网络描述与模型接纳检查。

化学记号（全文一致）：

* ``物种``（species）：液相中实际出现的分子形式，例如 A、B、AB、AB2；
* ``组分``（component / element）：守恒单元。一个物种的 ``composition``
  给出它携带每个组分的数目，例如 AB2 携带 1 份 A 和 2 份 B；
* 公式矩阵 ``A``（C×N）：``A[j,i]`` 是物种 i 携带组分 j 的数目；
* 反应计量矩阵 ``S``（N×R）：第 r 列是反应 r 的计量向量，产物取正、反应物取负。

理想溶液的平衡常数定义在活度上：``K_r = prod_i (c_i/c°)^nu_ir``，
``log_k`` 指自然对数 ln K。本模块只负责“这个网络作为化学描述是否被接纳”，
不做任何数值迭代。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .errors import ModelRejectionError, RejectionReason
from .units import concentration_to_mol_per_l, temperature_to_kelvin

__all__ = ["Reaction", "ChemicalModel", "build_model"]

_RANK_TOL = 1e-9  # 相对最大奇异值的矩阵秩容差


@dataclass(frozen=True)
class Reaction:
    """一条反应的调用方描述。

    Parameters
    ----------
    id:
        调用方提供的反应标识，会原样出现在拒绝原因和平衡残差里。
    stoichiometry:
        物种 -> 计量数，产物为正、反应物为负，例如
        ``{"AB": 1, "A": -1, "B": -1}``。
    k / log_k:
        无量纲平衡常数 K 或其自然对数 ln K，二者恰好给一个。
    """

    id: str
    stoichiometry: Mapping[str, float]
    k: float | None = None
    log_k: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("反应 id 必须是非空字符串")
        if not self.stoichiometry:
            raise ValueError(f"反应 {self.id!r} 没有任何计量项")
        if (self.k is None) == (self.log_k is None):
            raise ValueError(
                f"反应 {self.id!r} 必须且只能提供 k 或 log_k 其中之一"
            )
        if self.k is not None:
            if not np.isfinite(self.k) or self.k <= 0.0:
                raise ValueError(f"反应 {self.id!r} 的平衡常数 k 必须为正数")
        if self.log_k is not None and not np.isfinite(self.log_k):
            raise ValueError(f"反应 {self.id!r} 的 log_k 必须是有限数")

    @property
    def ln_k(self) -> float:
        if self.log_k is not None:
            return float(self.log_k)
        return float(np.log(self.k))  # type: ignore[arg-type]

    @classmethod
    def from_parts(
        cls,
        id: str,
        products: Mapping[str, float],
        reactants: Mapping[str, float],
        *,
        k: float | None = None,
        log_k: float | None = None,
    ) -> "Reaction":
        """用产物和反应物两个映射构造（产物正、反应物内部取负）。"""

        stoich: dict[str, float] = {}
        for name, nu in products.items():
            stoich[name] = stoich.get(name, 0.0) + float(nu)
        for name, nu in reactants.items():
            stoich[name] = stoich.get(name, 0.0) - float(nu)
        return cls(id, {k_: v for k_, v in stoich.items() if v != 0.0},
                   k=k, log_k=log_k)


@dataclass(frozen=True)
class ReactionCheck:
    """一条冗余（可由其他反应线性表出）反应的一致性核对记录。"""

    reaction_id: str
    given_ln_k: float
    implied_ln_k: float

    @property
    def ln_k_residual(self) -> float:
        return abs(self.given_ln_k - self.implied_ln_k)

    @property
    def factor_residual(self) -> float:
        """给定 K 与推导 K 相差的倍数（>=1）。"""

        return float(np.exp(self.ln_k_residual))


@dataclass(frozen=True)
class ChemicalModel:
    """被接纳的化学体系描述。不可变，可在同一进程中并存任意多个。

    矩阵的列顺序与 :attr:`species` 一致，行顺序与 :attr:`components` 一致。
    """

    model_id: str
    species: tuple[str, ...]
    components: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    temperature_k: float
    standard_concentration: float
    composition: Mapping[str, Mapping[str, float]]
    reactions: tuple[Reaction, ...]
    formula_matrix: np.ndarray          # A，形状 (C, N)
    stoichiometry_matrix: np.ndarray    # S，形状 (N, R)，全部反应
    independent_reaction_idx: tuple[int, ...]
    redundant_checks: tuple[ReactionCheck, ...]
    basis_species_idx: tuple[int, ...]      # 长度 C，A[:, basis] 可逆
    nonbasis_species_idx: tuple[int, ...]   # 长度 N-C
    system_id: str       # 物种/组分/配方矩阵/T/c°：同一化学体系
    network_id: str      # 在 system_id 之上再绑定反应描述

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def n_independent_reactions(self) -> int:
        return len(self.independent_reaction_idx)

    def component_amounts(self, concentrations: np.ndarray) -> np.ndarray:
        """由物种浓度向量计算各组分的总浓度 A @ c。"""

        c = np.asarray(concentrations, dtype=float)
        return self.formula_matrix @ c

    def is_same_system(self, other: "ChemicalModel") -> bool:
        """是否描述同一个化学体系（允许反应写法不同）。"""

        return self.system_id == other.system_id


def _rank(a: np.ndarray) -> int:
    """数值秩：以最大奇异值的相对阈值判定。"""

    if a.size == 0:
        return 0
    sv = np.linalg.svd(a, compute_uv=False)
    if sv[0] <= 0.0:
        return 0
    return int(np.sum(sv > _RANK_TOL * max(a.shape) * sv[0]))


def _independent_columns(mat: np.ndarray, rank: int) -> list[int]:
    """按输入顺序贪心选出 rank 个线性无关列。

    刻意保持调用方给列的先后顺序：先写的反应更可能是“主干”，后补的等价/
    拆分反应自然成为被核对的冗余项，使不一致归因到调用方后加的那条反应。
    """

    if rank == 0:
        return []
    scale = float(np.max(np.linalg.norm(mat, axis=0)))
    chosen: list[int] = []
    ortho: list[np.ndarray] = []
    for j in range(mat.shape[1]):
        v = mat[:, j].astype(float).copy()
        for e in ortho:
            v -= np.dot(v, e) * e
        if np.linalg.norm(v) > _RANK_TOL * max(1.0, scale) * 1e2:
            chosen.append(j)
            ortho.append(v / np.linalg.norm(v))
        if len(chosen) == rank:
            break
    return chosen


def _fingerprint(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_model(
    species_composition: Mapping[str, Mapping[str, float]],
    reactions: Sequence[Reaction],
    *,
    temperature: float = 298.15,
    temperature_unit: str = "K",
    standard_concentration: float = 1.0,
    standard_concentration_unit: str = "mol/L",
    components: Sequence[str] | None = None,
    model_id: str | None = None,
    consistency_tol_ln: float = 1e-7,
) -> ChemicalModel:
    """根据物种配方、反应和常数构建被接纳的模型；不合法时抛出
    :class:`ModelRejectionError`。

    Parameters
    ----------
    species_composition:
        物种 -> {组分: 携带数目}。物种的给定顺序即结果中的物种顺序。
    reactions:
        :class:`Reaction` 序列。允许包含冗余/等价反应，但其 K 必须与
        独立反应推导值在 ``consistency_tol_ln``（ln K 容差）内一致。
    temperature / temperature_unit:
        固定反应温度（K 或 °C）。
    standard_concentration:
        标准浓度 c°，默认 1 mol/L。
    components:
        显式组分顺序；默认取所有物种配方中出现过的组分。
    model_id:
        调用方可读的模型标识，默认生成短 uuid。
    """

    reasons: list[RejectionReason] = []

    # ---- 1. 名称与配方矩阵 ---------------------------------------------
    species = tuple(species_composition.keys())
    if not species:
        raise ModelRejectionError(RejectionReason(
            "EMPTY_SPECIES", "至少需要一个物种"))
    if len(set(species)) != len(species):
        dup = sorted({s for s in species if species.count(s) > 1})
        raise ModelRejectionError(RejectionReason(
            "DUPLICATE_SPECIES", f"物种标识重复：{dup}", species=tuple(dup)))

    if components is None:
        comps: tuple[str, ...] = tuple(dict.fromkeys(
            comp for vec in species_composition.values() for comp in vec))
    else:
        comps = tuple(components)
    if not comps:
        reasons.append(RejectionReason(
            "EMPTY_COMPONENTS", "没有任何守恒组分"))
    elif len(set(comps)) != len(comps):
        dup = sorted({c for c in comps if comps.count(c) > 1})
        reasons.append(RejectionReason(
            "DUPLICATE_COMPONENT", f"组分标识重复：{dup}",
            components=tuple(dup)))

    n_c, n_s = len(comps), len(species)
    c_index = {c: j for j, c in enumerate(comps)}
    s_index = {s: i for i, s in enumerate(species)}
    A = np.zeros((n_c, n_s))
    for i, sp in enumerate(species):
        vec = species_composition[sp]
        unknown = [c for c in vec if c not in c_index]
        if unknown:
            reasons.append(RejectionReason(
                "UNKNOWN_COMPONENT_IN_COMPOSITION",
                f"物种 {sp!r} 的配方引用了未知组分 {unknown}",
                components=tuple(unknown), species=(sp,)))
            continue
        for comp, num in vec.items():
            if not np.isfinite(num) or num < 0:
                reasons.append(RejectionReason(
                    "BAD_COMPOSITION_NUMBER",
                    f"物种 {sp!r} 携带组分 {comp!r} 的数目必须是非负有限数，"
                    f"收到 {num!r}",
                    components=(comp,), species=(sp,)))
                continue
            A[c_index[comp], i] = num
        if np.all(A[:, i] == 0.0):
            reasons.append(RejectionReason(
                "SPECIES_WITHOUT_COMPONENTS",
                f"物种 {sp!r} 不属于任何组分，无法被任何物料总量约束；"
                "当前模型不接纳无组成物种",
                species=(sp,)))

    t_k = temperature_to_kelvin(temperature, temperature_unit)
    if not np.isfinite(t_k) or t_k <= 0:
        reasons.append(RejectionReason(
            "BAD_TEMPERATURE", f"温度必须为正（开尔文），收到 {t_k}"))
    c0 = concentration_to_mol_per_l(
        standard_concentration, standard_concentration_unit)
    if not np.isfinite(c0) or c0 <= 0:
        reasons.append(RejectionReason(
            "BAD_STANDARD_CONCENTRATION",
            f"标准浓度必须为正数，收到 {c0}"))

    # ---- 2. 反应列与逐反应守恒检查 --------------------------------------
    reaction_ids = tuple(r.id for r in reactions)
    if not reactions:
        reasons.append(RejectionReason(
            "NO_REACTIONS", "网络中没有任何反应"))
    if len(set(reaction_ids)) != len(reaction_ids):
        dup = sorted({r for r in reaction_ids if reaction_ids.count(r) > 1})
        reasons.append(RejectionReason(
            "DUPLICATE_REACTION_ID", f"反应 id 重复：{dup}",
            reactions=tuple(dup)))

    S_cols: list[np.ndarray] = []
    kept_reactions: list[Reaction] = []
    for rxn in reactions:
        col = np.zeros(n_s)
        bad_species = [sp for sp in rxn.stoichiometry if sp not in s_index]
        if bad_species:
            reasons.append(RejectionReason(
                "UNKNOWN_SPECIES_IN_REACTION",
                f"反应 {rxn.id!r} 引用了未定义物种 {bad_species}",
                reactions=(rxn.id,), species=tuple(bad_species)))
            continue
        for sp, nu in rxn.stoichiometry.items():
            if not np.isfinite(nu):
                reasons.append(RejectionReason(
                    "BAD_STOICHIOMETRY",
                    f"反应 {rxn.id!r} 中物种 {sp!r} 的计量数必须有限",
                    reactions=(rxn.id,), species=(sp,)))
            else:
                col[s_index[sp]] = nu
        if not np.any(col != 0.0):
            reasons.append(RejectionReason(
                "EMPTY_REACTION",
                f"反应 {rxn.id!r} 去掉零计量项后为空",
                reactions=(rxn.id,)))
            continue
        imbalance = A @ col
        # 相对尺度：以 A 的列大小与反应计量大小的乘积为参照，
        # 对整数或分数计量、c° 取值都稳健。
        a_scale = float(np.linalg.norm(A, ord=np.inf)) if A.size else 0.0
        nu_scale = float(np.max(np.abs(col)))
        tol = _RANK_TOL * 100.0 * max(1.0, a_scale) * max(1.0, nu_scale)
        bad = np.abs(imbalance) > tol
        if np.any(bad):
            offending = tuple(comps[j] for j in np.where(bad)[0])
            amount = {comps[j]: float(imbalance[j])
                      for j in np.where(bad)[0]}
            reasons.append(RejectionReason(
                "REACTION_NOT_CONSERVATIVE",
                f"反应 {rxn.id!r} 不满足组分守恒，"
                f"反应前后各组分净变化为 {amount}（应为 0）",
                reactions=(rxn.id,), components=offending,
                details={"net_change": amount}))
            continue
        S_cols.append(col)
        kept_reactions.append(rxn)

    if reasons:
        raise ModelRejectionError(reasons)

    S = np.column_stack(S_cols)  # N x R
    rank_a = _rank(A)

    # 组分行线性相关意味着两个组分其实无法分别做收支核算
    if rank_a != n_c:
        reasons.append(RejectionReason(
            "DEPENDENT_COMPONENTS",
            f"{n_c} 个组分的配方向量只有秩 {rank_a}，存在可互相线性表出的"
            "组分，无法分别核对收支，请合并或删除冗余组分"))

    # ---- 3. 反应空间必须覆盖全部允许的组成变化方向 ----------------------
    expected_dim = n_s - rank_a          # nullity(A)
    rank_s = _rank(S)
    if rank_s > expected_dim:
        # 理论上不会到这里（守恒的列都在 A 的零空间内），作为防御性检查
        reasons.append(RejectionReason(
            "REACTION_SPACE_TOO_LARGE",
            f"反应矩阵秩 {rank_s} 超过配方矩阵零空间维数 {expected_dim}"))
    elif rank_s < expected_dim:
        # 找出未被反应列覆盖的零空间方向，定位涉及哪些物种。
        # null(A) 的一组正交基来自 A 的完全 SVD：A=U Σ V^T，
        # A x=0 的方向对应零奇异值的右奇异向量（V 的最后若干列）。
        _u, sigma, vt = np.linalg.svd(A, full_matrices=True)
        z_basis = vt[rank_a:, :].T                       # N x expected_dim
        covered = z_basis.T @ S                          # d x R
        u_c, _, _ = np.linalg.svd(covered, full_matrices=True)
        # 未覆盖方向 z 满足 covered^T z = 0，即 covered 的左零空间
        miss_coord = u_c[:, rank_s:]                     # d x (d-rank_s)
        missing = z_basis @ miss_coord                   # N x ...
        for k in range(missing.shape[1]):
            direction = missing[:, k]
            hit = np.argsort(np.abs(direction))[::-1]
            involved = tuple(species[i] for i in hit
                             if abs(direction[i]) > 1e-8)
            reasons.append(RejectionReason(
                "UNDERDESCRIBED_NETWORK",
                "反应数不足：存在满足全部组分守恒、却不由任何反应生成的组成"
                f"变化方向（涉及物种 {involved}）。模型秩为 {rank_s}，"
                f"独立反应应为 {expected_dim} 个；请补充缺失的反应",
                species=involved,
                details={"rank_reactions": rank_s,
                         "expected_independent_reactions": expected_dim}))

    if reasons:
        raise ModelRejectionError(reasons)

    # ---- 4. 独立反应选择与冗余反应的 K 一致性 ---------------------------
    ind_idx = _independent_columns(S, rank_s)
    dep_idx = [j for j in range(S.shape[1]) if j not in ind_idx]
    S_ind = S[:, ind_idx]
    ln_k_ind = np.array([kept_reactions[j].ln_k for j in ind_idx])

    checks: list[ReactionCheck] = []
    for j in dep_idx:
        # S[:,j] = S_ind @ x （列在张成空间内，最小二乘给出唯一系数）
        x, *_ = np.linalg.lstsq(S_ind, S[:, j], rcond=None)
        implied_ln = float(x @ ln_k_ind)
        given_ln = kept_reactions[j].ln_k
        check = ReactionCheck(kept_reactions[j].id, given_ln, implied_ln)
        checks.append(check)
        if check.ln_k_residual > consistency_tol_ln:
            reasons.append(RejectionReason(
                "INCONSISTENT_DUPLICATE_REACTION",
                f"反应 {check.reaction_id!r} 可由反应 "
                f"{[kept_reactions[k].id for k in ind_idx]} 线性表出，"
                f"给定 ln K={given_ln:.6g} 与由这些反应推导的 "
                f"ln K={implied_ln:.6g} 不一致（相差 {check.ln_k_residual:.3g}"
                f"，即 {check.factor_residual:.4g} 倍）。这表示不同的化学条件，"
                "不能放进同一个平衡网络",
                reactions=(check.reaction_id,),
                details={"given_ln_k": given_ln,
                         "implied_ln_k": implied_ln,
                         "factor": check.factor_residual}))
    if reasons:
        raise ModelRejectionError(reasons)

    # ---- 5. 求解器用的基物种选择（A[:, basis] 可逆） --------------------
    basis = _independent_columns(A, rank_a)
    assert len(basis) == n_c
    nonbasis = [i for i in range(n_s) if i not in basis]
    # 让两个序列按物种顺序排列，便于阅读
    basis = tuple(sorted(basis))
    nonbasis = tuple(nonbasis)

    # ---- 6. 体系指纹 ----------------------------------------------------
    # system_id 只绑定“有哪些物种、各携带哪些组分、T、c°”：
    # 对同一体系重新表达反应（拆步、补等价反应）不会改变 system_id。
    # network_id 额外绑定反应计量与 ln K；完全等价的重复反应不改变它。
    system_payload = {
        "species": species,
        "components": comps,
        "A": [[float(x) for x in row] for row in A],
        "T": float(t_k),
        "c0": float(c0),
    }
    system_key = _fingerprint(system_payload)
    rxn_payload = sorted(
        (
            sorted((species[i], float(S[i, j]))
                   for i in range(n_s) if S[i, j] != 0.0),
            round(float(kept_reactions[j].ln_k), 12),
        )
        for j in range(S.shape[1])
    )
    network_key = _fingerprint(
        {"system": system_key, "reactions": rxn_payload})

    model = ChemicalModel(
        model_id=model_id or f"model-{uuid.uuid4().hex[:8]}",
        species=species,
        components=comps,
        reaction_ids=tuple(r.id for r in kept_reactions),
        temperature_k=float(t_k),
        standard_concentration=float(c0),
        composition={sp: dict(species_composition[sp]) for sp in species},
        reactions=tuple(kept_reactions),
        formula_matrix=A,
        stoichiometry_matrix=S,
        independent_reaction_idx=tuple(ind_idx),
        redundant_checks=tuple(checks),
        basis_species_idx=basis,
        nonbasis_species_idx=nonbasis,
        system_id=system_key,
        network_id=network_key,
    )
    # 数值矩阵锁写，防止下游代码原地改写被共享的模型
    model.formula_matrix.setflags(write=False)
    model.stoichiometry_matrix.setflags(write=False)
    return model
