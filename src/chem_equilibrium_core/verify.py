"""Public, solver-independent confirmation of an equilibrium composition.

A result is accepted only when *both* hold:

* **Material balance** -- for every component,
  ``sum_i C[i,k] * c_i == b_k`` within tolerance.  A species carrying
  several components is counted in every corresponding balance.
* **Chemical equilibrium** -- for every reaction whose participating
  species are all present, ``sum_i nu_ji * ln(c_i / c_standard) == ln K``.
  A reaction whose reactant or product is exactly zero is flagged, since
  no finite equilibrium constant can describe that state.

No non-negativity is enforced here by clipping.  Negative or non-finite
concentrations fail verification outright, so an unfinished iteration
can never be hidden behind a final ``max(c, 0)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .errors import InputError
from .model import ReactionNetwork

DEFAULT_BALANCE_ATOL = 1e-9   # mol/L absolute on component totals
DEFAULT_BALANCE_RTOL = 1e-9   # relative to the magnitude of the total
DEFAULT_LOGK_ATOL = 1e-7      # dimensionless on sum(nu ln activity) - ln K
MIN_POSITIVE_CONC = 0.0       # exactly zero is reported, never repaired


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of checking one composition against the public balances."""

    ok: bool
    concentration_valid: bool
    component_residuals: Dict[str, float] = field(default_factory=dict)
    reaction_residuals: Dict[str, float] = field(default_factory=dict)
    boundary_reactions: Dict[str, str] = field(default_factory=dict)
    violations: Sequence[str] = field(default_factory=tuple)
    balance_atol: float = DEFAULT_BALANCE_ATOL
    balance_rtol: float = DEFAULT_BALANCE_RTOL
    logk_atol: float = DEFAULT_LOGK_ATOL

    @property
    def worst_component_residual(self) -> Optional[float]:
        return max((abs(v) for v in self.component_residuals.values()), default=None)

    @property
    def worst_reaction_residual(self) -> Optional[float]:
        vals = [v for v in self.reaction_residuals.values() if math.isfinite(v)]
        return max((abs(v) for v in vals), default=None)

    def assert_ok(self) -> None:
        if not self.ok:
            raise AssertionError(
                "equilibrium verification failed: " + "; ".join(self.violations)
            )


def _as_concentrations(network: ReactionNetwork, concentrations) -> np.ndarray:
    if isinstance(concentrations, Mapping):
        missing = [name for name in concentrations if name not in network.species]
        if missing:
            raise InputError(
                f"concentrations reference unknown species {missing}",
                code="unknown_species",
            )
        arr = np.zeros(network.n_species)
        for i, name in enumerate(network.species):
            if name in concentrations:
                arr[i] = float(concentrations[name])
        return arr
    arr = np.asarray(concentrations, dtype=float)
    if arr.shape != (network.n_species,):
        raise InputError(
            f"expected {network.n_species} concentrations in species order "
            f"{network.species}, got shape {arr.shape}",
            code="bad_concentration_shape",
        )
    return arr


def _as_totals(network: ReactionNetwork, total_concentrations) -> np.ndarray:
    if total_concentrations is None:
        return np.full(network.n_components, np.nan)
    if isinstance(total_concentrations, Mapping):
        arr = np.full(network.n_components, np.nan)
        for name, value in total_concentrations.items():
            arr[network.component_index(name)] = float(value)
        return arr
    arr = np.asarray(total_concentrations, dtype=float)
    if arr.shape != (network.n_components,):
        raise InputError(
            f"expected {network.n_components} component totals in component order "
            f"{network.components}, got shape {arr.shape}",
            code="bad_totals_shape",
        )
    return arr


def assess(
    network: ReactionNetwork,
    concentrations,
    total_concentrations=None,
    *,
    balance_atol: float = DEFAULT_BALANCE_ATOL,
    balance_rtol: float = DEFAULT_BALANCE_RTOL,
    logk_atol: float = DEFAULT_LOGK_ATOL,
) -> VerificationReport:
    """Independently verify a composition.

    Parameters
    ----------
    network:
        An accepted :class:`ReactionNetwork`.
    concentrations:
        Sequence in ``network.species`` order, or mapping of species
        name to concentration (mol/L).
    total_concentrations:
        Component totals to balance against (sequence in
        ``network.components`` order or mapping).  Components left
        ``None``/``NaN`` are skipped (equilibrium is still checked).
    balance_atol, balance_rtol:
        Component k passes when ``|residual| <= atol + rtol * |b_k|``.
    logk_atol:
        Tolerance on the (dimensionless) equilibrium residual.

    Returns a :class:`VerificationReport`; nothing is raised for a mere
    failed check (invalid *input shape* does raise
    :class:`~chem_equilibrium_core.errors.InputError`).
    """
    c = _as_concentrations(network, concentrations)
    b = _as_totals(network, total_concentrations)
    violations = []

    valid_shape = True
    if not np.all(np.isfinite(c)):
        valid_shape = False
        bad = [network.species[i] for i in range(network.n_species)
               if not math.isfinite(float(c[i]))]
        violations.append(f"non-finite concentration for {bad}")
    if np.any(c < MIN_POSITIVE_CONC):
        valid_shape = False
        bad = [f"{network.species[i]}={c[i]:g}"
               for i in range(network.n_species) if c[i] < MIN_POSITIVE_CONC]
        violations.append(f"negative concentration for {bad}")

    comp_res: Dict[str, float] = {}
    C = network.C()
    for k, comp in enumerate(network.components):
        if b.size and math.isnan(float(b[k])):
            continue
        resid = float(C[:, k] @ c - b[k])
        comp_res[comp] = resid
        allowed = balance_atol + balance_rtol * abs(float(b[k]))
        if abs(resid) > allowed:
            violations.append(
                f"component {comp!r} balance residual {resid:+.3e} mol/L "
                f"exceeds atol {balance_atol:g} + rtol {balance_rtol:g}*|total|"
            )

    rxn_res: Dict[str, float] = {}
    boundary: Dict[str, str] = {}
    if valid_shape:
        N = network.N()
        c0 = network.c_standard
        for j, rname in enumerate(network.reaction_names):
            nu = N[j]
            reactants = (nu < 0) & (c == 0.0)
            products = (nu > 0) & (c == 0.0)
            if np.any(reactants):
                # Boundary equilibrium (KKT inequality): a missing reactant
                # blocks the forward reaction, so ln(Q/K) = +infinity is the
                # *satisfied* condition.  (If a product is also zero the
                # reaction simply cannot proceed either way -- still fine.)
                rxn_res[rname] = float("inf")
                missing = [network.species[i] for i in range(network.n_species)
                           if reactants[i]]
                boundary[rname] = "blocked by zero reactant(s) " + ", ".join(missing)
                continue
            if np.any(products):
                zeros = [network.species[i] for i in range(network.n_species)
                         if products[i]]
                violations.append(
                    f"reaction {rname!r} has all reactants present but product(s) "
                    f"{zeros} exactly zero; equilibrium with finite K would form it"
                )
                rxn_res[rname] = float("-inf")
                continue
            resid = float(np.sum(nu[c > 0.0] * np.log(c[c > 0.0] / c0))
                          - network.log_k[j])
            rxn_res[rname] = resid
            if abs(resid) > logk_atol:
                violations.append(
                    f"reaction {rname!r} equilibrium residual ln(Q/K)={resid:+.3e} "
                    f"exceeds atol {logk_atol:g}"
                )
    else:
        # Cannot evaluate log activities on invalid numbers; mark each.
        for rname in network.reaction_names:
            rxn_res[rname] = float("nan")

    return VerificationReport(
        ok=not violations,
        concentration_valid=valid_shape,
        component_residuals=comp_res,
        reaction_residuals=rxn_res,
        boundary_reactions=boundary,
        violations=tuple(violations),
        balance_atol=balance_atol,
        balance_rtol=balance_rtol,
        logk_atol=logk_atol,
    )
