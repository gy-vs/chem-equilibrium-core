"""Domain exceptions for chem-equilibrium-core.

The library never lets a raw :class:`numpy.linalg.LinAlgError` escape.
Every failure points back at the call site's reactions, components,
samples, or at a result that could not be independently confirmed.
"""

from __future__ import annotations

from typing import Any, Sequence


class ChemCoreError(Exception):
    """Base class for every exception raised by the package."""


class ModelError(ChemCoreError):
    """A reaction network cannot be accepted as a chemical model.

    The error always carries the names of the offending reaction(s) or
    component(s)/species so a caller can locate the problem in the data
    it supplied, instead of receiving a linear-algebra message.
    """

    def __init__(
        self,
        reason: str,
        *,
        code: str = "model_invalid",
        reactions: Sequence[str] = (),
        components: Sequence[str] = (),
        species: Sequence[str] = (),
        detail: Any = None,
    ) -> None:
        self.reason = reason
        self.code = code
        self.reactions = tuple(reactions)
        self.components = tuple(components)
        self.species = tuple(species)
        self.detail = detail
        parts = [f"[{code}] {reason}"]
        if self.reactions:
            parts.append(f"reaction(s): {', '.join(self.reactions)}")
        if self.components:
            parts.append(f"component(s): {', '.join(self.components)}")
        if self.species:
            parts.append(f"species: {', '.join(self.species)}")
        super().__init__("; ".join(parts))


class InputError(ChemCoreError):
    """A sample operation or equilibrium call was given invalid input."""

    def __init__(self, reason: str, *, code: str = "input_invalid", detail: Any = None) -> None:
        self.reason = reason
        self.code = code
        self.detail = detail
        super().__init__(f"[{code}] {reason}")


class EquilibriumNotConfirmed(ChemCoreError):
    """The numeric calculation failed verification (or the solver failed).

    No sample or cached result is created when this is raised.  The
    attached :class:`~chem_equilibrium_core.verify.VerificationReport`
    is ``None`` only when the solver itself could not produce a vector;
    otherwise the report explains which public check failed.
    """

    def __init__(self, reason: str, *, report: Any = None, detail: Any = None) -> None:
        self.reason = reason
        self.report = report
        self.detail = detail
        msg = f"equilibrium not confirmed: {reason}"
        if report is not None and not getattr(report, "ok", True):
            if report.component_residuals:
                worst_comp = max(
                    report.component_residuals.items(), key=lambda kv: abs(kv[1])
                )
                msg += f" | worst component residual {worst_comp[0]}={worst_comp[1]:.3e}"
            if report.reaction_residuals:
                worst_rxn = max(
                    report.reaction_residuals.items(), key=lambda kv: abs(kv[1])
                )
                msg += f" | worst ln(Q/K) {worst_rxn[0]}={worst_rxn[1]:.3e}"
        super().__init__(msg)
