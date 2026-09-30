"""Samples: equilibrated material parcels and the solution-preparation chain.

A :class:`Sample` is an immutable record of *what is actually in the
flask* -- its species amounts in mol and its volume in L -- together
with the confirmed equilibrium state and full provenance.

Preparation operations never modify an existing sample:

* :func:`prepare_sample` -- equilibrate component amounts in a volume;
* :func:`mix_samples` -- pool aliquots of equilibrated samples and
  re-equilibrate.  Pooling averages *amounts*, so the component totals
  of the pooled liquor are conserved, but the pooled species
  concentrations are generally not at equilibrium; the returned sample
  carries the re-equilibrated composition;
* :func:`spike_component` -- add moles of a single (pure) component to
  a sample and equilibrate at a new final volume.

Only a successful, publicly verified calculation is registered in a
:class:`Lineage`; failures leave every prior sample untouched.
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .engine import equilibrium_concentrations
from .errors import InputError
from .model import ReactionNetwork
from .state import EquilibriumState


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationRecord:
    """One preparation step that produced a sample."""

    kind: str  # "prepare" | "mix" | "spike"
    detail: Mapping[str, object] = field(default_factory=dict)

    def parent_ids(self) -> Tuple[str, ...]:
        if self.kind == "mix":
            return tuple(entry["sample_id"] for entry in self.detail["inputs"])  # type: ignore[index]
        if self.kind == "spike":
            return (str(self.detail["parent"]),)
        return ()


@dataclass(frozen=True)
class Sample:
    """An immutable equilibrated parcel: species *amounts* and a volume."""

    sample_id: str
    network: ReactionNetwork
    volume: float
    amounts: Tuple[float, ...]           # mol in network.species order
    equilibrium: EquilibriumState
    provenance: Tuple[OperationRecord, ...] = ()

    # ----- material views ------------------------------------------------------
    @property
    def concentrations(self) -> Tuple[float, ...]:
        return tuple(n / self.volume for n in self.amounts)

    def concentration_map(self) -> Dict[str, float]:
        return dict(zip(self.network.species, self.concentrations))

    def amount_map(self) -> Dict[str, float]:
        return dict(zip(self.network.species, self.amounts))

    def component_amounts(self) -> Tuple[float, ...]:
        """Moles of each component actually carried by this sample."""
        n = np.asarray(self.amounts)
        return tuple(self.network.C().T @ n)

    def component_amount_map(self) -> Dict[str, float]:
        return dict(zip(self.network.components, self.component_amounts()))

    def component_total_concentrations(self) -> Tuple[float, ...]:
        return tuple(m / self.volume for m in self.component_amounts())

    @property
    def temperature(self) -> float:
        return self.network.temperature

    @property
    def c_standard(self) -> float:
        return self.network.c_standard

    @property
    def species_order(self) -> Tuple[str, ...]:
        return self.network.species

    @property
    def component_order(self) -> Tuple[str, ...]:
        return self.network.components

    @property
    def network_fingerprint(self) -> str:
        return self.network.fingerprint

    @property
    def parent_ids(self) -> Tuple[str, ...]:
        if not self.provenance:
            return ()
        return self.provenance[-1].parent_ids()

    def describe_origin(self) -> str:
        if not self.provenance:
            return f"sample {self.sample_id[:8]} (no recorded operation)"
        op = self.provenance[-1]
        if op.kind == "prepare":
            feed = op.detail["feed"]
            return (f"sample {self.sample_id[:8]} prepared from feed {feed} "
                    f"in {op.detail['volume']:g} L")
        if op.kind == "mix":
            parts = ", ".join(
                f"{entry['sample_id'][:8]}@{entry['volume']:g} L"
                for entry in op.detail["inputs"]  # type: ignore[union-attr]
            )
            return f"sample {self.sample_id[:8]} mixed from [{parts}]"
        if op.kind == "spike":
            return (f"sample {self.sample_id[:8]} from {str(op.detail['parent'])[:8]} "
                    f"+ {op.detail['amount']:g} mol {op.detail['component']!r}, "
                    f"final volume {op.detail['volume']:g} L")
        return f"sample {self.sample_id[:8]} via {op.kind}"


# ---------------------------------------------------------------------------
# Lineage registry
# ---------------------------------------------------------------------------


class Lineage:
    """Process-local, thread-safe record of produced samples.

    A lineage is deliberately *not* global: create separate instances
    (or use separate workspaces) to compute unrelated networks in the
    same process without interference.  Failed calculations never
    register anything.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._samples: Dict[str, Sample] = {}

    def register(self, sample: Sample) -> Sample:
        with self._lock:
            self._samples[sample.sample_id] = sample
        return sample

    def get(self, sample_id: str) -> Sample:
        with self._lock:
            try:
                return self._samples[sample_id]
            except KeyError:
                raise InputError(
                    f"no sample with id {sample_id!r} in this lineage",
                    code="unknown_sample",
                )

    def __contains__(self, sample_id: object) -> bool:
        with self._lock:
            return sample_id in self._samples

    def sample_ids(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._samples)

    def ancestors(self, sample_id: str) -> Tuple[Sample, ...]:
        """All samples the given sample descends from, nearest first."""
        start = self.get(sample_id)
        seen = {start.sample_id}
        ordered: List[Sample] = []
        frontier = list(start.parent_ids)
        while frontier:
            nxt_id = frontier.pop(0)
            if nxt_id in seen:
                continue
            seen.add(nxt_id)
            nxt = self.get(nxt_id)
            ordered.append(nxt)
            frontier.extend(p for p in nxt.parent_ids if p not in seen)
        return tuple(ordered)

    def chain(self, sample_id: str) -> Tuple[Sample, ...]:
        """Sample plus all ancestors, child first."""
        return (self.get(sample_id),) + self.ancestors(sample_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return uuid.uuid4().hex


def _tol_kwargs(
    balance_atol: Optional[float],
    balance_rtol: Optional[float],
    logk_atol: Optional[float],
) -> Dict[str, float]:
    kwargs: Dict[str, float] = {}
    if balance_atol is not None:
        kwargs["balance_atol"] = balance_atol
    if balance_rtol is not None:
        kwargs["balance_rtol"] = balance_rtol
    if logk_atol is not None:
        kwargs["logk_atol"] = logk_atol
    return kwargs


def _build_sample(
    network: ReactionNetwork,
    volume: float,
    component_amounts: np.ndarray,
    operation: OperationRecord,
    parents: Sequence[Sample],
    lineage: Optional[Lineage],
    tolerance_kwargs: Mapping[str, float],
) -> Sample:
    totals = np.asarray(component_amounts, dtype=float) / volume
    # The heavy call: raises EquilibriumNotConfirmed before anything is
    # registered or returned, so failed calculations cannot pollute state.
    concentrations, _, _, report = equilibrium_concentrations(
        network,
        totals,
        **tolerance_kwargs,
    )
    state = EquilibriumState(
        network=network,
        concentrations=tuple(float(x) for x in concentrations),
        total_concentrations=tuple(float(x) for x in totals),
        report=report,
    )
    history = tuple(sample.provenance for sample in parents)
    provenance: Tuple[OperationRecord, ...] = ()
    for chain in history:
        provenance += chain
    provenance += (operation,)

    sample = Sample(
        sample_id=_new_id(),
        network=network,
        volume=float(volume),
        amounts=tuple(float(x) for x in concentrations * volume),
        equilibrium=state,
        provenance=provenance,
    )
    if lineage is not None:
        lineage.register(sample)
    return sample


def _check_volume(volume: float, label: str = "volume") -> float:
    v = float(volume)
    if not math.isfinite(v) or v <= 0:
        raise InputError(
            f"{label} must be a positive finite number, got {volume!r}",
            code="bad_volume",
        )
    return v


def _check_feed(network: ReactionNetwork, feed: Mapping[str, float]) -> np.ndarray:
    amounts = np.zeros(network.n_components)
    for name, moles in feed.items():
        idx = network.component_index(name)
        v = float(moles)
        if not math.isfinite(v) or v < 0:
            raise InputError(
                f"feed for component {name!r} must be a finite non-negative "
                f"amount in mol, got {moles!r}",
                code="bad_feed",
            )
        amounts[idx] = v
    return amounts


# ---------------------------------------------------------------------------
# Public preparation operations
# ---------------------------------------------------------------------------


def prepare_sample(
    network: ReactionNetwork,
    volume: float,
    feed: Mapping[str, float],
    *,
    lineage: Optional[Lineage] = None,
    balance_atol: Optional[float] = None,
    balance_rtol: Optional[float] = None,
    logk_atol: Optional[float] = None,
) -> Sample:
    """Equilibrate a flask prepared from pure component amounts.

    Parameters
    ----------
    network:
        Accepted reaction network.
    volume:
        Final liquid volume in litres.
    feed:
        Moles of each component introduced, keyed by component name
        (omitted components default to zero).  A component fed via a
        complex species is still just its elemental ingredient amount:
        the network decides which species form.
    """
    v = _check_volume(volume)
    component_amounts = _check_feed(network, feed)
    operation = OperationRecord(
        kind="prepare",
        detail={"feed": {k: float(v_) for k, v_ in feed.items()}, "volume": v},
    )
    return _build_sample(
        network, v, component_amounts, operation, (), lineage,
        _tol_kwargs(balance_atol, balance_rtol, logk_atol),
    )


def mix_samples(
    aliquots: Sequence[Tuple[Sample, float]],
    *,
    lineage: Optional[Lineage] = None,
    balance_atol: Optional[float] = None,
    balance_rtol: Optional[float] = None,
    logk_atol: Optional[float] = None,
) -> Sample:
    """Pool aliquots of already-equilibrated samples and re-equilibrate.

    ``aliquots`` is a sequence of ``(sample, volume_in_litres)`` pairs;
    the volumes are the relative amounts pooled (so a 2:1 combination is
    ``[(s1, 2.0), (s2, 1.0)]`` even when each physical sample is smaller
    -- amounts are taken proportionally).  Every input sample must
    describe the same chemical system (fingerprint check).  The pooled
    species *amounts* are added, which preserves every component total
    exactly, and the mixture is equilibrated at the summed volume.
    """
    if not aliquots:
        raise InputError("mix_samples needs at least one aliquot", code="empty_mix")

    network: Optional[ReactionNetwork] = None
    total_volume = 0.0
    total_amounts = np.zeros(0)
    inputs_detail = []

    for sample, aliquot_v in aliquots:
        v = _check_volume(aliquot_v, label="aliquot volume")
        if network is None:
            network = sample.network
            total_amounts = np.zeros(network.n_species)
        elif sample.network.fingerprint != network.fingerprint:
            raise InputError(
                "all mixed samples must describe the same chemical system; "
                f"sample {sample.sample_id[:8]} has fingerprint "
                f"{sample.network.fingerprint[:16]} but the first aliquot has "
                f"{network.fingerprint[:16]} (check temperature, standard "
                "concentration, species/component order and the physical model)",
                code="network_mismatch",
            )
        fraction = v / sample.volume
        total_amounts = total_amounts + fraction * np.asarray(sample.amounts)
        total_volume += v
        inputs_detail.append({"sample_id": sample.sample_id, "volume": v})

    assert network is not None
    parents = tuple(sample for sample, _ in aliquots)
    operation = OperationRecord(
        kind="mix",
        detail={"inputs": tuple(inputs_detail), "volume": total_volume},
    )
    return _build_sample(
        network,
        total_volume,
        network.C().T @ total_amounts,  # species amounts -> component amounts
        operation,
        parents,
        lineage,
        _tol_kwargs(balance_atol, balance_rtol, logk_atol),
    )


def spike_component(
    sample: Sample,
    component: str,
    amount: float,
    final_volume: float,
    *,
    lineage: Optional[Lineage] = None,
    balance_atol: Optional[float] = None,
    balance_rtol: Optional[float] = None,
    logk_atol: Optional[float] = None,
) -> Sample:
    """Add moles of one pure component to a sample and re-equilibrate.

    The existing sample's full material content is carried over at its
    old volume (amounts are volume-independent), the component is added,
    and the mixture is equilibrated at ``final_volume`` litres (use this
    to model topping up with solvent after the addition).
    """
    comp_idx = sample.network.component_index(component)
    added = float(amount)
    if not math.isfinite(added) or added < 0:
        raise InputError(
            f"spike amount must be a finite non-negative number of mol, got {amount!r}",
            code="bad_spike",
        )
    v = _check_volume(final_volume, label="final volume")

    component_amounts = np.asarray(sample.component_amounts(), dtype=float)
    component_amounts[comp_idx] += added
    operation = OperationRecord(
        kind="spike",
        detail={
            "parent": sample.sample_id,
            "component": component,
            "amount": added,
            "volume_before": sample.volume,
            "volume": v,
        },
    )
    return _build_sample(
        sample.network,
        v,
        component_amounts,
        operation,
        (sample,),
        lineage,
        _tol_kwargs(balance_atol, balance_rtol, logk_atol),
    )
