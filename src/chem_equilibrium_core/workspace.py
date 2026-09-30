"""进程内模型/样品台账。

台账是可选的便利层：纯计算函数在 :mod:`chem_equilibrium_core.sample` 中
即可使用。台账保证两点：

1. 一次失败的计算绝不留下条目——新样品只在公开复核通过后才注册，因此调用方
   总能继续使用此前确认过的模型与样品；
2. 内部加锁，适合在同一进程里并发计算不同网络，样品以 id 检索、来源可追溯。
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping, Sequence

from .network import ChemicalModel
from .sample import (EquilibriumResult, Sample, add_component,
                     equilibrate_totals, make_sample, mix_samples)

__all__ = ["Workspace"]


class Workspace:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, ChemicalModel] = {}
        self._samples: dict[str, Sample] = {}

    # ---- 模型 ----------------------------------------------------------
    def add_model(self, model: ChemicalModel) -> ChemicalModel:
        with self._lock:
            self._models[model.model_id] = model
        return model

    def get_model(self, model_id: str) -> ChemicalModel:
        with self._lock:
            return self._models[model_id]

    def models(self) -> tuple[ChemicalModel, ...]:
        with self._lock:
            return tuple(self._models.values())

    def find_models_by_system(self, system_id: str) -> tuple[ChemicalModel, ...]:
        with self._lock:
            return tuple(m for m in self._models.values()
                         if m.system_id == system_id)

    # ---- 样品 ----------------------------------------------------------
    def get_sample(self, sample_id: str) -> Sample:
        with self._lock:
            return self._samples[sample_id]

    def samples(self) -> tuple[Sample, ...]:
        with self._lock:
            return tuple(self._samples.values())

    def has_sample(self, sample_id: str) -> bool:
        with self._lock:
            return sample_id in self._samples

    def _register(self, result: EquilibriumResult,
                  sample_id: str | None) -> Sample:
        """复核已在计算函数中完成；此处只做注册，失败计算不会到达这里。"""

        with self._lock:
            sid = sample_id or f"sample-{uuid.uuid4().hex[:8]}"
            if sid in self._samples:
                raise ValueError(f"样品 id {sid!r} 已存在")
            sample = make_sample(result, sid)
            self._samples[sid] = sample
            if result.model.model_id not in self._models:
                self._models[result.model.model_id] = result.model
            return sample

    def equilibrate(
        self,
        model: ChemicalModel,
        total_concentrations: Mapping[str, float] | Sequence[float],
        *,
        volume_l: float = 1.0,
        sample_id: str | None = None,
        balance_tol: float = 1e-8,
        equilibrium_tol: float = 1e-8,
    ) -> Sample:
        result = equilibrate_totals(
            model, total_concentrations, volume_l=volume_l,
            balance_tol=balance_tol, equilibrium_tol=equilibrium_tol)
        return self._register(result, sample_id)

    def mix(
        self,
        sample_ids: Sequence[str],
        *,
        volumes_l: Sequence[float] | None = None,
        model: ChemicalModel | None = None,
        sample_id: str | None = None,
        balance_tol: float = 1e-8,
        equilibrium_tol: float = 1e-8,
    ) -> Sample:
        with self._lock:
            inputs = tuple(self._samples[sid] for sid in sample_ids)
        result = mix_samples(
            inputs, volumes_l=volumes_l, model=model,
            balance_tol=balance_tol, equilibrium_tol=equilibrium_tol)
        return self._register(result, sample_id)

    def add_component(
        self,
        sample_id: str,
        additions: Mapping[str, float],
        *,
        final_volume_l: float,
        addition_unit: str = "mol",
        model: ChemicalModel | None = None,
        sample_id_out: str | None = None,
        balance_tol: float = 1e-8,
        equilibrium_tol: float = 1e-8,
    ) -> Sample:
        with self._lock:
            source = self._samples[sample_id]
        result = add_component(
            source, additions, final_volume_l=final_volume_l,
            addition_unit=addition_unit, model=model,
            balance_tol=balance_tol, equilibrium_tol=equilibrium_tol)
        return self._register(result, sample_id_out)

    def lineage(self, sample_id: str) -> dict:
        """返回样品的递归来源树（id -> 操作 -> 父样品 ...）。"""

        with self._lock:
            def node(sid: str) -> dict:
                s = self._samples[sid]
                return {"sample_id": sid,
                        "operation": s.origin.operation,
                        "detail": dict(s.origin.detail),
                        "parents": [node(p) for p in s.origin.parents]}
            return node(sample_id)
