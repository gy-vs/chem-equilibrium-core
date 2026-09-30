"""chem_equilibrium_core 公开的异常类型。

调用链上的失败分成四个层次，避免把所有问题都变成“数值不收敛”：

* :class:`ModelRejectionError`：反应网络本身不被接纳（缺反应、冗余反应常数
  不一致、计量不守恒……），在任何配液计算之前抛出，且携带调用方给出的
  反应/组分标识。
* :class:`SampleOperationError`：混合、补加等样品操作本身不合法（体系不同、
  体积超额、组分未知等）。
* :class:`EquilibriumSolveError`：模型合法，但数值内核没有找到残差足够小的
  候选解。它绝不携带“已确认”的结果。
* :class:`VerificationError`：数值内核给出了候选解，但公开的物料收支与化学
  平衡复核没有通过。结果只作为“未经确认候选”附加，不进入样品台账。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class RejectionReason:
    """模型被拒绝的一条结构化原因，关联调用方提供的反应/组分标识。"""

    code: str
    message: str
    reactions: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    species: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        tags: list[str] = []
        if self.reactions:
            tags.append("反应=" + ",".join(self.reactions))
        if self.components:
            tags.append("组分=" + ",".join(self.components))
        if self.species:
            tags.append("物种=" + ",".join(self.species))
        suffix = f" [{'; '.join(tags)}]" if tags else ""
        return f"[{self.code}] {self.message}{suffix}"


class ChemEquilibriumError(Exception):
    """本库所有异常的基类。"""


class ModelRejectionError(ChemEquilibriumError):
    """反应网络在建模阶段被拒绝，附带一条或多条 :class:`RejectionReason`。"""

    def __init__(self, reasons: RejectionReason | Sequence[RejectionReason]):
        if isinstance(reasons, RejectionReason):
            reasons = (reasons,)
        self.reasons: tuple[RejectionReason, ...] = tuple(reasons)
        body = "\n  - ".join(r.format() for r in self.reasons)
        super().__init__("反应网络未被接纳：\n  - " + body)


class SampleOperationError(ChemEquilibriumError):
    """混合或补加操作无法执行（与数值求解无关）。"""


class EquilibriumSolveError(ChemEquilibriumError):
    """模型合法，但数值内核未能交出通过复核所需精度的候选组成。"""

    def __init__(self, message: str, *, residual: float | None = None) -> None:
        super().__init__(message)
        self.residual = residual


class VerificationError(ChemEquilibriumError):
    """候选解在公开复核中不满足物料守恒与化学平衡，不能成为已确认样品。"""

    def __init__(self, message: str, report: Any = None) -> None:
        super().__init__(message)
        self.report = report
