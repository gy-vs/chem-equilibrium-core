"""已确认样品与配液操作。

样品（:class:`Sample`）是“一份已经达到平衡的液相”：它携带模型身份、物种
平衡浓度、体积（因而携带各组分的*物质的量*）、完整的公开复核报告和溯源信息。
样品一经创建即不可变——混合和补加都返回新样品，输入样品不被改写。

操作分两层：

* 纯函数 :func:`equilibrate_totals` / :func:`mix_samples` /
  :func:`add_component`：不依赖任何全局状态，适合批处理管线；
* :class:`~chem_equilibrium_core.workspace.Workspace`：在上面加一层进程内
  台账，记录已确认模型与样品，且**只有通过复核的结果才会入册**，一次失败
  计算不会污染此前的台账。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .errors import SampleOperationError
from .network import ChemicalModel
from .solver import solve_candidate
from .units import volume_to_liters
from .verification import VerificationReport, verify_composition

__all__ = ["Origin", "Sample", "EquilibriumResult", "equilibrate_totals",
           "mix_samples", "add_component"]


@dataclass(frozen=True)
class Origin:
    """样品的来源记录，回答“这份样品从哪来”。"""

    operation: str
    """``direct`` / ``mix`` / ``add_component`` 之一。"""

    parents: tuple[str, ...] = ()
    """输入样品 id（direct 时为空）。"""

    detail: Mapping[str, object] = field(default_factory=dict)
    """操作参数，例如取出体积、补加的组分与数量、混合体积比。"""


@dataclass(frozen=True)
class EquilibriumResult:
    """一次再平衡计算返回的公开结果包。"""

    model: ChemicalModel
    temperature_k: float
    standard_concentration: float
    species: tuple[str, ...]
    components: tuple[str, ...]
    system_id: str
    network_id: str
    model_id: str
    concentrations: np.ndarray
    total_concentrations: np.ndarray
    volume_l: float
    verification: VerificationReport
    numerical_attempts: tuple[dict, ...]
    origin: Origin

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "system_id": self.system_id,
            "network_id": self.network_id,
            "temperature_k": self.temperature_k,
            "standard_concentration": self.standard_concentration,
            "species": list(self.species),
            "components": list(self.components),
            "concentrations": list(map(float, self.concentrations)),
            "total_concentrations": list(map(float, self.total_concentrations)),
            "volume_l": self.volume_l,
            "verification": self.verification.to_dict(),
            "origin": {"operation": self.origin.operation,
                       "parents": list(self.origin.parents),
                       "detail": dict(self.origin.detail)},
        }


@dataclass(frozen=True)
class Sample:
    """一份不可变的、已通过复核的平衡样品。"""

    sample_id: str
    model: ChemicalModel
    concentrations: np.ndarray   # mol/L，顺序与 model.species 一致
    volume_l: float
    verification: VerificationReport
    origin: Origin

    # ---- 基础读取 ------------------------------------------------------
    @property
    def concentration_map(self) -> dict[str, float]:
        return dict(zip(self.model.species,
                        (float(x) for x in self.concentrations)))

    @property
    def total_concentrations(self) -> np.ndarray:
        """各组分总浓度 b = A @ c，mol/L。"""

        return self.model.formula_matrix @ self.concentrations

    @property
    def component_totals(self) -> dict[str, float]:
        b = self.total_concentrations
        return {comp: float(b[j])
                for j, comp in enumerate(self.model.components)}

    @property
    def component_amounts(self) -> dict[str, float]:
        """各组分携带的物质的量（mol），由体积与总量决定。"""

        return {k: v * self.volume_l
                for k, v in self.component_totals.items()}

    @property
    def species_amounts(self) -> dict[str, float]:
        return {sp: float(c) * self.volume_l
                for sp, c in self.concentration_map.items()}

    @property
    def system_id(self) -> str:
        return self.model.system_id

    def concentration_of(self, species: str) -> float:
        if species not in self.model.species:
            raise SampleOperationError(
                f"物种 {species!r} 不属于模型 {self.model.model_id!r}")
        return float(self.concentrations[self.model.species.index(species)])

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (f"Sample(id={self.sample_id!r}, op={self.origin.operation}, "
                f"V={self.volume_l:g} L, system={self.system_id[:8]})")


# ----------------------------------------------------------------------
# 内部：求解 + 公开复核，只有复核通过才组装结果
# ----------------------------------------------------------------------

def _rebalance(
    model: ChemicalModel,
    b: np.ndarray,
    volume_l: float,
    origin: Origin,
    *,
    balance_tol: float,
    equilibrium_tol: float,
) -> EquilibriumResult:
    if not np.all(np.isfinite(b)) or np.any(b < 0.0):
        raise SampleOperationError(
            f"各组分总浓度必须是非负有限数，收到 {b.tolist()}")
    if not np.all(b > 0.0):
        zero = [model.components[j] for j in range(model.n_components)
                if b[j] <= 0.0]
        raise SampleOperationError(
            f"组分 {zero} 的总浓度为零：严格正浓度假设下不接纳完全不含某"
            "组分的配方（请从体系中去掉该组分，或给出正的投入量）")
    if not np.isfinite(volume_l) or volume_l <= 0:
        raise SampleOperationError(f"最终体积必须为正数，收到 {volume_l!r}")

    outcome = solve_candidate(model, b)
    report = verify_composition(
        model, outcome.concentrations, b,
        balance_tol=balance_tol, equilibrium_tol=equilibrium_tol)
    # 复核不过：直接抛出，调用方此前的模型与样品不受任何影响
    report.raise_if_failed()

    c = np.array(outcome.concentrations, dtype=float)
    c.setflags(write=False)
    bb = np.array(b, dtype=float)
    bb.setflags(write=False)
    return EquilibriumResult(
        model=model,
        temperature_k=model.temperature_k,
        standard_concentration=model.standard_concentration,
        species=model.species,
        components=model.components,
        system_id=model.system_id,
        network_id=model.network_id,
        model_id=model.model_id,
        concentrations=c,
        total_concentrations=bb,
        volume_l=float(volume_l),
        verification=report,
        numerical_attempts=outcome.attempts,
        origin=origin,
    )


def _new_sample(model: ChemicalModel, result: EquilibriumResult,
                sample_id: str | None) -> Sample:
    return Sample(
        sample_id=sample_id or f"sample-{uuid.uuid4().hex[:8]}",
        model=model,
        concentrations=result.concentrations,
        volume_l=result.volume_l,
        verification=result.verification,
        origin=result.origin,
    )


# ----------------------------------------------------------------------
# 公开操作
# ----------------------------------------------------------------------

def equilibrate_totals(
    model: ChemicalModel,
    total_concentrations: Mapping[str, float] | Sequence[float],
    *,
    volume_l: float = 1.0,
    balance_tol: float = 1e-8,
    equilibrium_tol: float = 1e-8,
) -> EquilibriumResult:
    """从“各组分总浓度 + 最终体积”直接计算一份新平衡（配液起点）。

    ``total_concentrations`` 可以是按 ``model.components`` 排序的序列，也
    可以是组分 -> mol/L 的映射。
    """

    b = _component_vector(model, total_concentrations, "total_concentrations")
    origin = Origin(
        operation="direct",
        detail={"total_concentrations_mol_per_l": dict(
            zip(model.components, (float(x) for x in b))),
                "volume_l": float(volume_l)})
    return _rebalance(model, b, volume_l, origin,
                      balance_tol=balance_tol,
                      equilibrium_tol=equilibrium_tol)


def mix_samples(
    samples: Sequence[Sample],
    *,
    volumes_l: Sequence[float] | None = None,
    model: ChemicalModel | None = None,
    balance_tol: float = 1e-8,
    equilibrium_tol: float = 1e-8,
) -> EquilibriumResult:
    """把若干份已平衡样品按体积混合，并对混合后的物料重新平衡。

    混合按体积可加处理：新总量为各取出体积携带的物质的量之和除以混合后
    体积。直接平均旧物种浓度（``sum V c / sum V``）一般不是新平衡——它只
    保证组分守恒，不满足混合体积下的质量作用定律。

    ``volumes_l`` 缺省表示各自整份混合；给出的值不得超过样品保存体积。
    所有样品必须属于同一体系（``system_id`` 相同，反应写法可以不同）。
    ``model`` 缺省取第一份样品的模型。
    """

    samples = list(samples)
    if len(samples) < 2:
        raise SampleOperationError("mix_samples 至少需要两份样品")
    sys_ids = {s.system_id for s in samples}
    if len(sys_ids) != 1:
        raise SampleOperationError(
            "不能混合不同化学体系的样品，发现 system_id="
            + ", ".join(sorted(sys_ids)))

    if volumes_l is None:
        volumes = [s.volume_l for s in samples]
    else:
        volumes = [float(v) for v in volumes_l]
        if len(volumes) != len(samples):
            raise SampleOperationError(
                f"volumes_l 长度 {len(volumes)} 与样品数 {len(samples)} 不一致")
    for s, v in zip(samples, volumes):
        if v <= 0 or v > s.volume_l + 1e-12:
            raise SampleOperationError(
                f"样品 {s.sample_id!r} 保存体积为 {s.volume_l:g} L，"
                f"不能取出 {v:g} L")
    total_v = float(sum(volumes))

    n_c = samples[0].model.n_components
    n_mol = np.zeros(n_c)
    for s, v in zip(samples, volumes):
        n_mol += s.total_concentrations * v
    b = n_mol / total_v

    solver_model = model or samples[0].model
    if solver_model.system_id not in sys_ids:
        raise SampleOperationError(
            "显式给出的 model 与输入样品不属于同一体系")

    origin = Origin(
        operation="mix",
        parents=tuple(s.sample_id for s in samples),
        detail={"volumes_l": volumes, "total_volume_l": total_v,
                "parent_model_ids": [s.model.model_id for s in samples],
                "solver_model_id": solver_model.model_id})
    return _rebalance(solver_model, b, total_v, origin,
                      balance_tol=balance_tol,
                      equilibrium_tol=equilibrium_tol)


def add_component(
    sample: Sample,
    additions: Mapping[str, float],
    *,
    final_volume_l: float,
    addition_unit: str = "mol",
    model: ChemicalModel | None = None,
    balance_tol: float = 1e-8,
    equilibrium_tol: float = 1e-8,
) -> EquilibriumResult:
    """向一份样品补加组分（纯物质投料）并改变最终体积，然后重新平衡。

    Parameters
    ----------
    additions:
        组分 -> 补加数量。默认单位 mol；``addition_unit="mmol"`` 时按毫摩。
        补加的是守恒组分而非具体物种——具体由平衡决定生成哪些物种。
    final_volume_l:
        补加后的最终液相体积（必须为正）。
    """

    if not additions:
        raise SampleOperationError("add_component 至少需要补加一种组分")
    factor = {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6}.get(addition_unit)
    if factor is None:
        raise SampleOperationError(
            f"不支持的投料单位 {addition_unit!r}，仅接受 mol/mmol/umol")
    m = sample.model
    unknown = [c for c in additions if c not in m.components]
    if unknown:
        raise SampleOperationError(
            f"组分 {unknown} 不属于体系 {m.system_id[:8]}（组分列表："
            f"{list(m.components)}）")
    if not np.isfinite(final_volume_l) or final_volume_l <= 0:
        raise SampleOperationError(
            f"最终体积必须为正数，收到 {final_volume_l!r}")
    for comp, amount in additions.items():
        if not np.isfinite(amount) or amount < 0:
            raise SampleOperationError(
                f"组分 {comp!r} 的补加量必须是非负有限数，收到 {amount!r}")

    n_mol = sample.total_concentrations * sample.volume_l
    for comp, amount in additions.items():
        n_mol[m.components.index(comp)] += float(amount) * factor
    b = n_mol / final_volume_l

    solver_model = model or m
    if solver_model.system_id != sample.system_id:
        raise SampleOperationError(
            "显式给出的 model 与输入样品不属于同一体系")

    origin = Origin(
        operation="add_component",
        parents=(sample.sample_id,),
        detail={"additions": {c: float(a) * factor for c, a in additions.items()},
                "addition_unit": "mol",
                "previous_volume_l": sample.volume_l,
                "final_volume_l": float(final_volume_l),
                "solver_model_id": solver_model.model_id})
    return _rebalance(solver_model, b, final_volume_l, origin,
                      balance_tol=balance_tol,
                      equilibrium_tol=equilibrium_tol)


def make_sample(result: EquilibriumResult,
                sample_id: str | None = None) -> Sample:
    """把已确认结果包成不可变样品（id 可由台账或调用方指定）。"""

    return _new_sample(result.model, result, sample_id)


def _component_vector(model: ChemicalModel,
                      data: Mapping[str, float] | Sequence[float],
                      where: str) -> np.ndarray:
    if isinstance(data, Mapping):
        missing = [c for c in data if c not in model.components]
        if missing:
            raise SampleOperationError(
                f"{where} 中出现了体系外组分 {missing}；"
                f"已知组分：{list(model.components)}")
        vec = np.zeros(model.n_components)
        for comp, val in data.items():
            vec[model.components.index(comp)] = float(val)
        return vec
    vec = np.asarray(data, dtype=float)
    if vec.shape != (model.n_components,):
        raise SampleOperationError(
            f"{where} 需要长度 {model.n_components}（组分 "
            f"{list(model.components)}）的序列，收到形状 {vec.shape}")
    return vec
