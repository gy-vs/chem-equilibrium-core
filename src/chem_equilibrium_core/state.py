"""Confirmed equilibrium state returned for every successful calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .model import ReactionNetwork
from .verify import VerificationReport


@dataclass(frozen=True)
class EquilibriumState:
    """A composition that has passed material-balance and equilibrium checks.

    Attributes
    ----------
    network:
        The exact accepted network used (temperature, standard
        concentration, species/component order included).
    concentrations:
        Equilibrium concentrations in ``network.species`` order (mol/L).
    total_concentrations:
        Component totals actually satisfied, in
        ``network.components`` order (mol/L).
    temperature, c_standard:
        Echoed physical settings, so another program can confirm it is
        looking at the same chemical system without digging through the
        network object.
    report:
        The public :class:`~chem_equilibrium_core.verify.VerificationReport`.
    """

    network: ReactionNetwork
    concentrations: Tuple[float, ...]
    total_concentrations: Tuple[float, ...]
    report: VerificationReport

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

    def concentration_map(self) -> Dict[str, float]:
        return dict(zip(self.network.species, self.concentrations))

    def component_total_map(self) -> Dict[str, float]:
        return dict(zip(self.network.components, self.total_concentrations))

    def concentrations_array(self) -> np.ndarray:
        a = np.asarray(self.concentrations, dtype=float)
        a.setflags(write=False)
        return a

    def component_residuals(self) -> Dict[str, float]:
        return dict(self.report.component_residuals)

    def reaction_residuals(self) -> Dict[str, float]:
        return dict(self.report.reaction_residuals)

    def confirmed(self) -> bool:
        return self.report.ok
