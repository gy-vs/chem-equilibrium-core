"""chem_equilibrium_core — 固定温度单液相理想溶液的反应平衡与批量配液内核。

调用链（职责分明，数值求解器只负责其中一环）：

  反应网络描述 (Reaction)
      │  build_model：计量守恒、秩/零空间覆盖、冗余反应 ln K 一致性
      ▼
  ChemicalModel（不可变；system_id 标识同一化学体系）
      │  equilibrate_totals / mix_samples / add_component
      ▼
  solver.solve_candidate（ln(c/c°) 空间，多初值，不裁剪）
      │  候选浓度
      ▼
  verification.verify_composition（物料收支 + 全部反应的质量作用定律）
      ▼
  EquilibriumResult / Sample（不可变、带溯源；Workspace 仅注册复核通过者）
"""

from __future__ import annotations

from .errors import (ChemEquilibriumError, EquilibriumSolveError,
                     ModelRejectionError, RejectionReason, SampleOperationError,
                     VerificationError)
from .network import ChemicalModel, Reaction, build_model
from .sample import (EquilibriumResult, Origin, Sample, add_component,
                     equilibrate_totals, make_sample, mix_samples)
from .verification import (BalanceRecord, EquilibriumRecord, VerificationReport,
                           verify_composition)
from .workspace import Workspace

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 网络与模型
    "Reaction",
    "ChemicalModel",
    "build_model",
    # 配液操作
    "equilibrate_totals",
    "mix_samples",
    "add_component",
    "make_sample",
    # 样品与结果
    "Sample",
    "Origin",
    "EquilibriumResult",
    # 复核
    "verify_composition",
    "VerificationReport",
    "BalanceRecord",
    "EquilibriumRecord",
    # 台账
    "Workspace",
    # 异常
    "ChemEquilibriumError",
    "ModelRejectionError",
    "RejectionReason",
    "SampleOperationError",
    "EquilibriumSolveError",
    "VerificationError",
]
