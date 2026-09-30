"""Workspace facade: one consistent call chain for a batch-prep process.

A workspace binds together:

* accepted reaction networks (registered by fingerprint, so re-adding an
  equivalent description reuses one canonical object);
* a :class:`~chem_equilibrium_core.samples.Lineage` of produced samples;
* the preparation/equilibration entry points.

Create as many workspaces as needed in one process; they share no state.
This is the supported way to calculate unrelated reaction networks
concurrently.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .engine import equilibrium_concentrations
from .errors import InputError
from .model import ReactionNetwork
from .samples import (
    Lineage,
    Sample,
    mix_samples,
    prepare_sample,
    spike_component,
)
from .state import EquilibriumState
from .verify import VerificationReport, assess


class Workspace:
    """Registry of networks and samples for one calling program."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or "workspace"
        self._lock = threading.RLock()
        self._networks: Dict[str, ReactionNetwork] = {}
        self.lineage = Lineage()

    # ----- networks ------------------------------------------------------------
    def register_network(self, network: ReactionNetwork) -> ReactionNetwork:
        """Accept a validated network.

        Equivalent descriptions (same fingerprint, e.g. re-split
        reactions) collapse to the first canonical object registered,
        so identity comparisons stay stable.
        """
        with self._lock:
            existing = self._networks.get(network.fingerprint)
            if existing is None:
                self._networks[network.fingerprint] = network
                return network
            return existing

    def networks(self) -> Tuple[ReactionNetwork, ...]:
        with self._lock:
            return tuple(self._networks.values())

    def get_network(self, fingerprint_or_id: str) -> ReactionNetwork:
        with self._lock:
            if fingerprint_or_id in self._networks:
                return self._networks[fingerprint_or_id]
            for network in self._networks.values():
                if network.network_id == fingerprint_or_id:
                    return network
        raise InputError(
            f"no network with fingerprint/id {fingerprint_or_id!r} in "
            f"workspace {self.name!r}",
            code="unknown_network",
        )

    # ----- direct equilibration ------------------------------------------------
    def equilibrate(
        self,
        network: ReactionNetwork,
        total_concentrations,
        *,
        balance_atol: Optional[float] = None,
        balance_rtol: Optional[float] = None,
        logk_atol: Optional[float] = None,
    ) -> EquilibriumState:
        """Equilibrate from component total concentrations (mol/L).

        Convenience wrapper that does not create a sample (no volume /
        provenance).  For batch preparation use :meth:`prepare_sample`,
        :meth:`mix_samples` and :meth:`spike_component`.
        """
        network = self.register_network(network)
        kwargs = {}
        if balance_atol is not None:
            kwargs["balance_atol"] = balance_atol
        if balance_rtol is not None:
            kwargs["balance_rtol"] = balance_rtol
        if logk_atol is not None:
            kwargs["logk_atol"] = logk_atol
        concentrations, _, _, report = equilibrium_concentrations(
            network, total_concentrations, **kwargs
        )
        totals = _resolve_totals(network, total_concentrations)
        return EquilibriumState(
            network=network,
            concentrations=tuple(float(x) for x in concentrations),
            total_concentrations=tuple(float(x) for x in totals),
            report=report,
        )

    def verify(self, network: ReactionNetwork, state_or_concentrations,
               total_concentrations=None, **kwargs) -> VerificationReport:
        """Run the public, independent verification on any composition."""
        network = self.register_network(network)
        if isinstance(state_or_concentrations, EquilibriumState):
            concentrations = state_or_concentrations.concentrations
            if total_concentrations is None:
                total_concentrations = state_or_concentrations.total_concentrations
        else:
            concentrations = state_or_concentrations
        return assess(network, concentrations, total_concentrations, **kwargs)

    # ----- sample chain --------------------------------------------------------
    def prepare_sample(
        self,
        network: ReactionNetwork,
        volume: float,
        feed: Mapping[str, float],
        **kwargs,
    ) -> Sample:
        network = self.register_network(network)
        return prepare_sample(network, volume, feed, lineage=self.lineage, **kwargs)

    def mix_samples(self, aliquots: Sequence[Tuple[Sample, float]], **kwargs) -> Sample:
        return mix_samples(aliquots, lineage=self.lineage, **kwargs)

    def spike_component(
        self,
        sample: Sample,
        component: str,
        amount: float,
        final_volume: float,
        **kwargs,
    ) -> Sample:
        return spike_component(
            sample, component, amount, final_volume,
            lineage=self.lineage, **kwargs,
        )

    def get_sample(self, sample_id: str) -> Sample:
        return self.lineage.get(sample_id)

    def ancestors(self, sample_id: str) -> Tuple[Sample, ...]:
        return self.lineage.ancestors(sample_id)

    def chain(self, sample_id: str) -> Tuple[Sample, ...]:
        return self.lineage.chain(sample_id)


def _resolve_totals(network: ReactionNetwork, total_concentrations) -> np.ndarray:
    if isinstance(total_concentrations, Mapping):
        arr = np.zeros(network.n_components)
        for name, value in total_concentrations.items():
            arr[network.component_index(name)] = float(value)
        return arr
    return np.asarray(total_concentrations, dtype=float)
