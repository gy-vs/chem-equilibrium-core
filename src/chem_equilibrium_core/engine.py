"""Numerical engine: equilibrium concentrations for an accepted network.

Mathematics
-----------
For an ideal solution the equilibrium conditions are

    sum_i nu_ji ln(c_i / c0) = ln K_j        for every reaction j

together with the component balances ``C^T c = b``.  Introducing dual
variables ``lambda`` for the balances, the unique (strictly convex)
solution can be parameterised as

    ln(c_i / c0) = (C lambda)_i - g_i

where ``g`` is the minimum-norm per-species standard Gibbs energy
supplied by the accepted network (``N g / c0 = -ln K``).  Substitution
gives the dual balance equations

    F(lambda) = C^T exp(C lambda - g) - b = 0,

whose Jacobian ``H = C^T diag(c) C`` is symmetric positive (semi-)
definite.  The damped Newton iteration starts from a *carrier* guess at
the feed's order of magnitude, caps each log-concentration move, and
backtracks on the balance residual, so it converges for feasible feeds
even when the component totals span many orders of magnitude.

The engine never clips concentrations to be non-negative: positivity is
structural (``c = c0 * exp(...)``).  After solving, the composition is
handed to the public
:func:`~chem_equilibrium_core.verify.assess` check and is returned only
if that report passes.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import EquilibriumNotConfirmed, InputError
from .model import ReactionNetwork
from .verify import (
    DEFAULT_BALANCE_ATOL,
    DEFAULT_BALANCE_RTOL,
    DEFAULT_LOGK_ATOL,
    VerificationReport,
    assess,
)

MAX_NEWTON_ITERATIONS = 100
LOG_CONC_CLIP = 700.0  # exp(+/-700) bounds any chemically meaningful value
MAX_LOG_STEP = 3.0     # cap |delta ln c| per Newton step before backtracking
CONVERGENCE_BALANCE_RTOL = 1e-10  # per fed component (public check is 1e-9)
CONVERGENCE_BALANCE_ATOL = 1e-12  # mol/L floor on tiny component totals

TotalsInput = Union[Mapping[str, float], Sequence[float], None]


def _totals_vector(network: ReactionNetwork, totals: TotalsInput) -> np.ndarray:
    if totals is None:
        raise InputError(
            "component total concentrations are required (mapping or sequence "
            "in component order)",
            code="missing_totals",
        )
    if isinstance(totals, Mapping):
        arr = np.zeros(network.n_components)
        for name, value in totals.items():
            idx = network.component_index(name)
            v = float(value)
            if not math.isfinite(v):
                raise InputError(
                    f"total for component {name!r} is not finite: {value!r}",
                    code="bad_totals",
                )
            if v < 0:
                raise InputError(
                    f"total for component {name!r} is negative ({v:g}); feeds "
                    "cannot contain negative amounts",
                    code="negative_totals",
                )
            arr[idx] = v
        return arr
    arr = np.asarray(totals, dtype=float)
    if arr.shape != (network.n_components,):
        raise InputError(
            f"expected {network.n_components} component totals in component order "
            f"{network.components}, got shape {arr.shape}",
            code="bad_totals_shape",
        )
    if not np.all(np.isfinite(arr)):
        raise InputError("component totals must all be finite", code="bad_totals")
    if np.any(arr < 0):
        raise InputError(
            f"component totals must be non-negative, got {arr.tolist()}",
            code="negative_totals",
        )
    return arr


def equilibrium_concentrations(
    network: ReactionNetwork,
    total_concentrations: TotalsInput,
    *,
    balance_atol: float = DEFAULT_BALANCE_ATOL,
    balance_rtol: float = DEFAULT_BALANCE_RTOL,
    logk_atol: float = DEFAULT_LOGK_ATOL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, VerificationReport]:
    """Solve one equilibrium problem.

    Returns ``(concentrations, active_components, active_species, report)``
    where the two boolean masks mark the positive-total components and
    the species reachable by them.  Raises
    :class:`~chem_equilibrium_core.errors.EquilibriumNotConfirmed` if no
    publicly verifiable composition is found; no partial result escapes.
    """
    b = _totals_vector(network, total_concentrations)

    C = network.C()
    g = network.standard_gibbs_energies()
    c0 = network.c_standard

    active_comp = b > 0.0
    # A species can be present only if *every* component it carries is
    # fed: a species carrying an un-fed component would create mass out
    # of nothing.  Species that fail this are exactly zero in the
    # solution (boundary reactions involving them are checked as
    # inequalities by the public verifier).
    inactive_comp = ~active_comp
    if np.any(inactive_comp):
        active_sp = np.all(C[:, inactive_comp] == 0.0, axis=1)
    else:
        active_sp = np.ones(network.n_species, dtype=bool)

    # Positive demand for a component no species carries: infeasible feed.
    for k in range(network.n_components):
        if b[k] > 0.0 and not np.any(C[:, k] > 0.0):
            raise InputError(
                f"feed demands component {network.components[k]!r} (total "
                f"{b[k]:g} mol/L) but no declared species carries it; the model "
                "cannot balance this feed",
                code="unsatisfiable_totals",
            )

    concentrations = np.zeros(network.n_species)
    if np.any(active_comp):
        Ca = C[np.ix_(active_sp, active_comp)]
        ga = g[active_sp]
        ba = b[active_comp]

        # Rank deficiency on the active substructure means the fed
        # component balances are not independent among reachable species.
        from .linalg_utils import rank
        from .model import RANK_TOL
        if rank(Ca, RANK_TOL) < int(active_comp.sum()):
            fed = [network.components[k] for k in range(network.n_components)
                   if active_comp[k]]
            raise EquilibriumNotConfirmed(
                "the reachable species cannot balance every fed component "
                f"independently (fed components: {fed})",
            )

        concentrations[active_sp] = _solve_dual(Ca, ga, ba, c0)

    report = assess(
        network,
        concentrations,
        b,
        balance_atol=balance_atol,
        balance_rtol=balance_rtol,
        logk_atol=logk_atol,
    )
    if not report.ok:
        raise EquilibriumNotConfirmed(
            "solver output failed public verification", report=report
        )
    return concentrations, active_comp, active_sp, report


def _solve_dual(Ca: np.ndarray, ga: np.ndarray, ba: np.ndarray, c0: float) -> np.ndarray:
    """Damped Newton on the dual balance equations ``Ca^T c(lam) = ba``.

    Concentrations stay strictly positive through
    ``c = c0 exp((Ca lam - g)/c0)``.  Robustness for feeds spanning many
    orders of magnitude comes from:

    * a *carrier initial guess* -- each component total starts in a
      species that carries only that component, so the initial imbalance
      is O(K), never O(total);
    * a cap on the largest per-species log-concentration change in one
      iteration (avoids being thrown against the exp() clipping wall);
    * an Armijo backtracking line search on the (unweighted) balance
      residual, consistent with the Newton direction.

    Convergence is judged **per component on a relative scale**, so a
    trace component fed at 1e-5 mol/L must close as tightly as a bulk
    component fed at 1e5 mol/L.
    """
    m = Ca.shape[1]
    n_sp = Ca.shape[0]
    scale = np.maximum(np.abs(ba), CONVERGENCE_BALANCE_ATOL / CONVERGENCE_BALANCE_RTOL)

    def conc(lam_vec: np.ndarray) -> np.ndarray:
        exponent = (Ca @ lam_vec - ga) / c0
        np.clip(exponent, -LOG_CONC_CLIP, LOG_CONC_CLIP, out=exponent)
        return c0 * np.exp(exponent)

    # Carrier initial guess: for every component k find a species
    # carrying only k (it exists whenever the composition matrix has full
    # column rank), and place that component's total there.
    c_guess = np.full(n_sp, 1e-30)
    missing = False
    for k in range(m):
        carrier = -1
        for i in range(n_sp):
            if Ca[i, k] > 0.0 and np.count_nonzero(Ca[i]) == 1:
                carrier = i
                break
        if carrier < 0:
            missing = True
            break
        c_guess[carrier] = ba[k] / Ca[carrier, k]
    if missing:
        # No pure carriers (unusual but rank-valid): spread O(total) over
        # the carrying species.
        per_carrier = ba / np.maximum((Ca > 0.0).sum(axis=0), 1)
        c_guess = np.maximum((Ca > 0.0) @ per_carrier, 1e-30)

    lam, *_ = np.linalg.lstsq(Ca, ga + c0 * np.log(c_guess), rcond=None)
    c = conc(lam)
    F = Ca.T @ c - ba

    for iteration in range(MAX_NEWTON_ITERATIONS):
        rel = np.max(np.abs(F) / scale)
        if rel <= CONVERGENCE_BALANCE_RTOL:
            return c

        H = (Ca.T * c) @ Ca
        try:
            step = np.linalg.solve(H, F)
        except np.linalg.LinAlgError as exc:
            raise EquilibriumNotConfirmed(
                f"dual Hessian is singular at iteration {iteration}: {exc}"
            )

        # Cap the largest induced log-concentration move.
        dx = Ca @ step
        max_dx = float(np.max(np.abs(dx))) if dx.size else 0.0
        t = min(1.0, MAX_LOG_STEP / max_dx) if max_dx > MAX_LOG_STEP else 1.0
        norm = max(float(np.max(np.abs(F))), 1e-300)
        slack = 1e-12 * max(1.0, norm)
        accepted = False
        for _ in range(80):
            c_try = conc(lam - t * step)
            F_try = Ca.T @ c_try - ba
            if np.max(np.abs(F_try)) <= (1.0 - 1e-4 * t) * norm + slack:
                lam = lam - t * step
                c = c_try
                F = F_try
                accepted = True
                break
            t *= 0.5
        if not accepted:
            raise EquilibriumNotConfirmed(
                "line search could not reduce the balance residual at iteration "
                f"{iteration} (worst relative residual {rel:.3e})"
            )

    raise EquilibriumNotConfirmed(
        f"dual Newton did not reach relative tolerance "
        f"{CONVERGENCE_BALANCE_RTOL:.0e} in {MAX_NEWTON_ITERATIONS} iterations "
        f"(worst relative residual {np.max(np.abs(F) / scale):.3e})"
    )
