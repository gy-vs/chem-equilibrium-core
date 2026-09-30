"""Reaction-network description and model acceptance.

Conventions
-----------
* A *component* is an independently conserved material ingredient.
* A *species* is a dissolved species carrying a non-negative amount of
  each component, recorded as a row of the composition matrix ``C``
  (species x components).  Therefore a species belonging to several
  components contributes to all of their totals simultaneously.
* A :class:`Reaction` is one row of the stoichiometric matrix ``N``
  (reactions x species): products positive, reactants negative.
* ``K`` is the (dimensionless) thermodynamic equilibrium constant at the
  network's fixed temperature, defined with standard concentration
  ``c_standard`` in mol/L; the numerical activity of species i is
  ``c_i / c_standard``.

A network is accepted only if:

1. names are unique and every stoichiometric coefficient references a
   declared species;
2. components are linearly independent;
3. every reaction conserves every component: ``N C = 0``;
4. every ``K`` is a positive finite number;
5. the equilibrium constants are thermodynamically consistent -- a
   reaction implied by others must carry the implied ``K``;
6. the reactions span all composition-conserving directions, so every
   species' equilibrium abundance is actually determined.

Nothing here depends on a numerical iteration; rejected models raise
:class:`~chem_equilibrium_core.errors.ModelError` at build time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import ModelError
from .linalg_utils import nullspace

# Tolerances used at model-acceptance time.
RANK_TOL = 1e-10          # structural: rank / nullspace decisions
CONSERVATION_TOL = 1e-9   # |N C| relative to largest stoichiometric entry
LOGK_CONSISTENCY_TOL = 1e-8  # |sum nu ln K - implied| per implied reaction

Number = Union[int, float]
StoichInput = Mapping[str, Number]


@dataclass(frozen=True)
class Species:
    """A dissolved species and the components it carries.

    ``composition`` maps component name to the (non-negative) amount of
    that component per mole of species.  Integer stoichiometric amounts
    are conventional but any finite non-negative value is allowed.
    """

    name: str
    composition: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def of(cls, name: str, composition: Optional[Mapping[str, Number]] = None) -> "Species":
        return cls(name=name, composition=dict(composition or {}))


@dataclass(frozen=True)
class Reaction:
    """One equilibrium reaction: products positive, reactants negative.

    Construct directly from a signed mapping, e.g.
    ``Reaction("AB_fwd", {"A": -1, "B": -1, "AB": 1}, K=10.0)``,
    or from two-sided form with :meth:`from_formula`.
    """

    name: str
    stoichiometry: Mapping[str, float]
    log_k: float

    @property
    def K(self) -> float:  # noqa: N802 - familiar chemistry symbol
        return math.exp(self.log_k)

    @classmethod
    def from_formula(
        cls,
        name: str,
        reactants: Mapping[str, Number],
        products: Mapping[str, Number],
        K: Number,  # noqa: N803
    ) -> "Reaction":
        stoich: Dict[str, float] = {}
        for sp, nu in reactants.items():
            stoich[sp] = stoich.get(sp, 0.0) - float(nu)
        for sp, nu in products.items():
            stoich[sp] = stoich.get(sp, 0.0) + float(nu)
        return cls(name=name, stoichiometry={k: v for k, v in stoich.items() if v != 0.0},
                   log_k=math.log(float(K)))


@dataclass(frozen=True)
class ReactionNetwork:
    """An accepted, immutable ideal-solution reaction network.

    Construct networks with :class:`NetworkBuilder`.  All arrays are
    exposed read-only via ``np.ndarray`` views; tuple fields hold the
    authoritative (ordered) names.
    """

    temperature: float
    c_standard: float
    species: Tuple[str, ...]
    components: Tuple[str, ...]
    reaction_names: Tuple[str, ...]
    composition: Tuple[Tuple[float, ...], ...]
    stoichiometry: Tuple[Tuple[float, ...], ...]
    log_k: Tuple[float, ...]
    logk_consistency_tol: float = LOGK_CONSISTENCY_TOL
    network_id: str = ""
    fingerprint: str = ""

    # ----- dimensions / indices ------------------------------------------------
    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_names)

    def species_index(self, name: str) -> int:
        try:
            return self.species.index(name)
        except ValueError:
            raise ModelError(
                f"species {name!r} is not part of this network",
                code="unknown_species",
                species=(name,),
            )

    def component_index(self, name: str) -> int:
        try:
            return self.components.index(name)
        except ValueError:
            raise ModelError(
                f"component {name!r} is not part of this network",
                code="unknown_component",
                components=(name,),
            )

    # ----- matrices (read-only) ------------------------------------------------
    def C(self) -> np.ndarray:  # noqa: N802
        a = np.array(self.composition, dtype=float)
        a.setflags(write=False)
        return a

    def N(self) -> np.ndarray:  # noqa: N802
        a = np.array(self.stoichiometry, dtype=float)
        a.setflags(write=False)
        return a

    def lnK(self) -> np.ndarray:
        a = np.array(self.log_k, dtype=float)
        a.setflags(write=False)
        return a

    def standard_gibbs_energies(self) -> np.ndarray:
        """Minimum-norm per-species ``g`` satisfying ``N g / c0 = -ln K``.

        ``g_i`` is the standard Gibbs energy of species i divided by
        ``R T`` and shifted so that the component basis species have
        zero reference.  Adding an equivalent reaction cannot change the
        physical system; the minimum-norm solution is identical for any
        thermodynamically equivalent reaction set spanning the same
        reaction space, which is why it feeds the network fingerprint.
        """
        n = self.N()
        if self.n_reactions == 0:
            a = np.zeros(self.n_species)
        else:
            a, *_ = np.linalg.lstsq(n / self.c_standard, -self.lnK(), rcond=None)
        a.setflags(write=False)
        return a

    def describes_same_system(self, other: "ReactionNetwork") -> bool:
        """True iff the two networks are the same chemical system.

        Comparison uses fingerprints, so networks that differ only by an
        added/re-split but thermodynamically equivalent reaction compare
        equal as long as species and component order agree.
        """
        return isinstance(other, ReactionNetwork) and self.fingerprint == other.fingerprint


# ---------------------------------------------------------------------------
# Builder / acceptance
# ---------------------------------------------------------------------------


class NetworkBuilder:
    """Accumulate a reaction description and validate it as one unit.

    Example
    -------
    >>> network = (NetworkBuilder(temperature=298.15, c_standard=1.0)
    ...     .add_species("A", {"A": 1})
    ...     .add_species("B", {"B": 1})
    ...     .add_species("AB", {"A": 1, "B": 1})
    ...     .add_reaction("AB_fwd", {"A": -1, "B": -1, "AB": 1}, K=10.0)
    ...     .build())
    """

    def __init__(
        self,
        *,
        temperature: Number = 298.15,
        c_standard: Number = 1.0,
        logk_consistency_tol: float = LOGK_CONSISTENCY_TOL,
    ) -> None:
        t = float(temperature)
        c0 = float(c_standard)
        if not (math.isfinite(t) and t > 0):
            raise ModelError(
                f"temperature must be a positive finite number, got {temperature!r}",
                code="bad_temperature",
            )
        if not (math.isfinite(c0) and c0 > 0):
            raise ModelError(
                f"c_standard must be a positive finite number, got {c_standard!r}",
                code="bad_standard_concentration",
            )
        if not (math.isfinite(logk_consistency_tol) and logk_consistency_tol > 0):
            raise ModelError(
                "logk_consistency_tol must be a positive finite number",
                code="bad_tolerance",
            )
        self.temperature = t
        self.c_standard = c0
        self.logk_consistency_tol = float(logk_consistency_tol)
        self._components: List[str] = []
        self._species: List[Tuple[str, Dict[str, float]]] = []
        self._reactions: List[Reaction] = []

    # ----- incremental input ---------------------------------------------------
    def add_components(self, names: Iterable[str]) -> "NetworkBuilder":
        for name in names:
            self._require_unique(str(name), "component")
            self._components.append(str(name))
        return self

    def add_species(
        self, name: str, composition: Optional[Mapping[str, Number]] = None
    ) -> "NetworkBuilder":
        name = str(name)
        self._require_unique(name, "species")
        comp: Dict[str, float] = {}
        for cname, amount in (composition or {}).items():
            cname = str(cname)
            value = float(amount)
            if not math.isfinite(value):
                raise ModelError(
                    f"amount of component {cname!r} in species {name!r} must be finite",
                    code="bad_composition",
                    components=(cname,),
                    species=(name,),
                )
            if value < 0:
                raise ModelError(
                    f"species {name!r} cannot carry a negative amount of component "
                    f"{cname!r} ({value:g}); decomposition is expressed with reactions",
                    code="negative_composition",
                    components=(cname,),
                    species=(name,),
                )
            if value == 0.0:
                continue
            comp[cname] = value
        if not comp:
            raise ModelError(
                f"species {name!r} carries no component; it cannot be conserved or "
                "balanced against any feed",
                code="empty_species",
                species=(name,),
            )
        for cname in comp:
            if cname not in self._components:
                self._components.append(cname)
        self._species.append((name, comp))
        return self

    def add_reaction(
        self,
        name: str,
        stoichiometry: Mapping[str, Number],
        K: Number,  # noqa: N803
    ) -> "NetworkBuilder":
        name = str(name)
        self._require_unique(name, "reaction")
        k = float(K)
        if not (math.isfinite(k) and k > 0):
            raise ModelError(
                f"reaction {name!r} needs a positive finite equilibrium constant, got {K!r}",
                code="bad_equilibrium_constant",
                reactions=(name,),
            )
        coeffs: Dict[str, float] = {}
        for sp, nu in stoichiometry.items():
            value = float(nu)
            if not math.isfinite(value) or value == 0.0:
                continue
            coeffs[str(sp)] = coeffs.get(str(sp), 0.0) + value
        coeffs = {sp: nu for sp, nu in coeffs.items() if nu != 0.0}
        if not coeffs:
            raise ModelError(
                f"reaction {name!r} has no non-zero stoichiometric coefficients",
                code="empty_reaction",
                reactions=(name,),
            )
        self._reactions.append(Reaction(name=name, stoichiometry=coeffs, log_k=math.log(k)))
        return self

    def add_reaction_formula(
        self,
        name: str,
        reactants: Mapping[str, Number],
        products: Mapping[str, Number],
        K: Number,  # noqa: N803
    ) -> "NetworkBuilder":
        return self.add_reaction(
            name,
            {**{s: -float(v) for s, v in reactants.items()},
             **{s: float(v) for s, v in products.items()}},
            K,
        )

    def extend_reactions(
        self,
        reactions: Iterable[Tuple[str, Mapping[str, Number], Number]],
    ) -> "NetworkBuilder":
        for name, stoich, K in reactions:  # noqa: N806
            self.add_reaction(name, stoich, K)
        return self

    # ----- acceptance ----------------------------------------------------------
    def build(self) -> ReactionNetwork:
        species_names = tuple(name for name, _ in self._species)
        components = tuple(self._components)

        if not components:
            raise ModelError(
                "network declares no components; nothing can be conserved",
                code="no_components",
            )
        if not species_names:
            raise ModelError(
                "network declares no species", code="no_species",
                components=components,
            )

        # Assemble C.
        c_rows = np.zeros((len(species_names), len(components)))
        for i, (_, comp) in enumerate(self._species):
            for cname, value in comp.items():
                c_rows[i, components.index(cname)] = value

        self._check_component_independence(c_rows, components)

        # Assemble N, checking referenced species.
        n_rows = np.zeros((len(self._reactions), len(species_names)))
        max_nu = 1.0
        for j, rxn in enumerate(self._reactions):
            for sp, nu in rxn.stoichiometry.items():
                if sp not in species_names:
                    raise ModelError(
                        f"reaction {rxn.name!r} references species {sp!r}, which is "
                        "not declared in the network",
                        code="reaction_unknown_species",
                        reactions=(rxn.name,),
                        species=(sp,),
                    )
                n_rows[j, species_names.index(sp)] = nu
                max_nu = max(max_nu, abs(nu))

        self._check_reaction_conservation(n_rows, c_rows, components)
        self._check_equilibrium_determinacy(n_rows, c_rows, species_names)
        g, bad = self._check_thermodynamic_consistency(n_rows)

        network = ReactionNetwork(
            temperature=self.temperature,
            c_standard=self.c_standard,
            species=species_names,
            components=components,
            reaction_names=tuple(r.name for r in self._reactions),
            stoichiometry=tuple(tuple(row) for row in n_rows),
            composition=tuple(tuple(row) for row in c_rows),
            log_k=tuple(r.log_k for r in self._reactions),
            logk_consistency_tol=self.logk_consistency_tol,
        )
        fingerprint = _compute_fingerprint(network, g)
        # Frozen dataclass: construct a copy carrying identity fields.
        return ReactionNetwork(
            temperature=network.temperature,
            c_standard=network.c_standard,
            species=network.species,
            components=network.components,
            reaction_names=network.reaction_names,
            composition=network.composition,
            stoichiometry=network.stoichiometry,
            log_k=network.log_k,
            logk_consistency_tol=network.logk_consistency_tol,
            network_id=fingerprint[:12],
            fingerprint=fingerprint,
        )

    # ----- individual checks ---------------------------------------------------
    def _require_unique(self, name: str, kind: str) -> None:
        if kind == "component":
            taken = set(self._components)
        elif kind == "species":
            taken = {n for n, _ in self._species}
        else:
            taken = {r.name for r in self._reactions}
        if name in taken:
            raise ModelError(
                f"a {kind} named {name!r} is already declared in this network",
                code=f"duplicate_{kind}",
                species=(name,) if kind == "species" else (),
                reactions=(name,) if kind == "reaction" else (),
                components=(name,) if kind == "component" else (),
            )

    def _check_component_independence(self, c_rows: np.ndarray, components: Tuple[str, ...]) -> None:
        # Components are columns of C; dependence is detected through the
        # right nullspace of C: vectors w with C w = 0 (w in R^n_components).
        basis = nullspace(c_rows, RANK_TOL)  # rows w
        if basis.shape[0] == 0:
            return
        for w in basis:
            idxs = [i for i, wi in enumerate(w) if abs(wi) > RANK_TOL]
            involved = [components[i] for i in idxs]
            # Express the relation readably: name one component as a linear
            # combination of the others.
            lead = max(idxs, key=lambda i: abs(w[i]))
            others = [components[i] for i in idxs if i != lead]
            detail = {"coefficients": [float(x) for x in w]}
            if others:
                reason = (
                    f"component {components[lead]!r} is a linear combination of "
                    f"{', '.join(others)!r} under the declared species compositions; "
                    "components must be linearly independent ingredients"
                )
            else:
                reason = (
                    f"component {components[lead]!r} appears in no species and cannot "
                    "be part of any conserved balance"
                )
            raise ModelError(
                reason,
                code="dependent_components",
                components=tuple(involved),
                detail=detail,
            )

    def _check_reaction_conservation(
        self,
        n_rows: np.ndarray,
        c_rows: np.ndarray,
        components: Tuple[str, ...],
    ) -> None:
        if n_rows.shape[0] == 0:
            return
        nc = n_rows @ c_rows  # reactions x components
        scale = max(1.0, float(np.max(np.abs(n_rows))))
        for j in range(nc.shape[0]):
            bad_k = [k for k in range(nc.shape[1])
                     if abs(nc[j, k]) > CONSERVATION_TOL * scale]
            if bad_k:
                bad = [components[k] for k in bad_k]
                imbalance = {components[k]: float(nc[j, k]) for k in bad_k}
                raise ModelError(
                    f"reaction {self._reactions[j].name!r} does not conserve "
                    f"component(s) {', '.join(bad)}: net production "
                    f"{imbalance} (sum of stoichiometric coefficients weighted by "
                    "species composition must be zero). Fix the stoichiometry or the "
                    "species compositions before the model can be used.",
                    code="reaction_not_conserving",
                    reactions=(self._reactions[j].name,),
                    components=tuple(bad),
                    detail=imbalance,
                )

    def _check_equilibrium_determinacy(
        self,
        n_rows: np.ndarray,
        c_rows: np.ndarray,
        species_names: Tuple[str, ...],
    ) -> None:
        """Reactions must span the whole nullspace of C^T."""
        missing = nullspace(c_rows.T, RANK_TOL)  # rows w: w C = 0
        if missing.shape[0] == 0:
            return
        if n_rows.shape[0] == 0:
            span = np.zeros((0, missing.shape[1]))
        else:
            span = n_rows
        # Project each required conservation-null direction onto the
        # orthogonal complement of the reaction space.
        if span.shape[0]:
            q, _ = np.linalg.qr(span.T, mode="reduced")
            proj = missing - (missing @ q) @ q.T
        else:
            proj = missing
        for i in range(missing.shape[0]):
            if np.linalg.norm(proj[i]) > 1e-8:
                w = missing[i]
                involved = [species_names[k] for k, x in enumerate(w) if abs(x) > RANK_TOL]
                raise ModelError(
                    f"the reactions do not determine the equilibrium of species "
                    f"{', '.join(involved)}: there is a composition-conserving "
                    "interconversion among these species that no reaction with an "
                    "equilibrium constant covers. Add the missing reaction (or remove "
                    "the undetermined species) before using the model.",
                    code="underdetermined_species",
                    reactions=tuple(r.name for r in self._reactions),
                    species=tuple(involved),
                )

    def _check_thermodynamic_consistency(self, n_rows: np.ndarray):
        """Return minimum-norm g or name the reaction cycle that is inconsistent."""
        if n_rows.shape[0] == 0:
            return np.zeros(n_rows.shape[1]), None
        g, *_ = np.linalg.lstsq(
            n_rows / self.c_standard,
            -np.array([r.log_k for r in self._reactions]),
            rcond=None,
        )
        # Rows of the left nullspace z (z^T N = 0) are the reaction
        # cycles; consistency demands z^T ln K = 0 for every cycle.
        kernel = nullspace((n_rows / self.c_standard).T, RANK_TOL)
        lnK = np.array([r.log_k for r in self._reactions])
        for z in kernel:
            cycle_resid = float(-z @ lnK)
            if abs(cycle_resid) > self.logk_consistency_tol:
                self._raise_cycle_inconsistency(z, cycle_resid)
        return g, None

    def _raise_cycle_inconsistency(self, z: np.ndarray, resid: float) -> None:
        idx = [(j, float(zj)) for j, zj in enumerate(z) if abs(zj) > RANK_TOL]
        names = [self._reactions[j].name for j, _ in idx]

        # Name the reaction implied by the others as explicitly as
        # possible.  A candidate j is implied when every other cycle
        # coefficient is present.  We only quote an explicit implied K
        # when the implied relation has near-integer stoichiometric
        # coefficients (e.g. 1, -1 or 1, 1); otherwise the generic cycle
        # message is less misleading.  Ties prefer the latest reaction,
        # i.e. the one just being added.
        target = None
        best_score = None
        for j0, z0 in idx:
            others = [(j, zj) for j, zj in idx if j != j0]
            if not others or abs(z0) <= RANK_TOL:
                continue
            factors = tuple(-zj / z0 for j, zj in others)
            implied = sum(
                factor * self._reactions[j].log_k for (j, _), factor in zip(others, factors)
            )
            declared = self._reactions[j0].log_k
            if abs(implied - declared) <= self.logk_consistency_tol:
                continue
            coeff_frac = max(abs(f - round(f)) for f in factors)
            if coeff_frac > 1e-6:
                continue
            score = (coeff_frac, -j0)  # cleaner coefficients, then latest
            if best_score is None or score < best_score:
                best_score = score
                target = (j0, math.exp(implied), math.exp(declared),
                          tuple(round(f) for f in factors))

        def term(coef: float, label: str) -> str:
            if coef == 1.0:
                return f"+{label}"
            if coef == -1.0:
                return f"-{label}"
            return f"{coef:+g}·{label}"

        terms = " ".join(term(coef, self._reactions[j].name) for j, coef in idx).lstrip("+")
        if target is not None:
            j0, implied_k, declared_k, factors = target
            # Re-express the cycle in integer-combination form against
            # the named target reaction for a readable implication.
            combo_terms = []
            for (j, _), factor in zip(
                [(j, zj) for j, zj in idx if j != j0], factors
            ):
                combo_terms.append(term(factor, self._reactions[j].name))
            combo = " ".join(combo_terms).lstrip("+")
            reason = (
                f"reaction {self._reactions[j0].name!r} is implied by the other "
                f"reactions (it equals {combo}), but its constant is inconsistent "
                f"with them: implied K = {implied_k:.6g}, declared K = "
                f"{declared_k:.6g}. These constants describe different chemical "
                "conditions; resolve the discrepancy before the model can be accepted."
            )
        else:
            reason = (
                "the equilibrium constants are thermodynamically inconsistent around "
                f"the reaction cycle {terms}: the weighted log-constants must sum to "
                f"zero but give {resid:+.3e} (tolerance "
                f"{self.logk_consistency_tol:g}). The reactions cannot describe one "
                "chemical system simultaneously."
            )
        raise ModelError(
            reason,
            code="inconsistent_equilibrium_constants",
            reactions=tuple(names),
            detail={"log_constant_cycle_residual": resid},
        )


def _compute_fingerprint(network: ReactionNetwork, g: np.ndarray) -> str:
    """Identity hash of the physical system.

    Includes temperature, standard concentration, ordered species and
    components, the composition matrix, and the gauge-invariant
    (minimum-norm) per-species standard Gibbs energies.  Reaction
    *descriptions* are intentionally excluded: re-splitting a reaction
    or adding an equivalent one leaves the fingerprint unchanged.
    """

    def rounded(values) -> list:
        # 9 decimal digits on dimensionless logs is far inside float noise
        # while stable across equivalent reaction descriptions.
        return [round(float(v), 9) for v in np.asarray(values).ravel()]

    payload = {
        "schema": "chem-equilibrium-core/fingerprint/v1",
        "temperature": round(float(network.temperature), 9),
        "c_standard": round(float(network.c_standard), 12),
        "species": list(network.species),
        "components": list(network.components),
        "C": rounded(network.C()),
        "g_over_RT": rounded(g),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def evolve_network(
    network: ReactionNetwork,
    *,
    add_species: Sequence[Tuple[str, Mapping[str, Number]]] = (),
    add_reactions: Sequence[Tuple[str, Mapping[str, Number], Number]] = (),
    logk_consistency_tol: Optional[float] = None,
) -> ReactionNetwork:
    """Create a new validated network that extends an existing one.

    The input network is never modified.  Added reactions are checked
    against the whole existing description, so an extra equivalent
    reaction with the wrong constant is rejected here.
    """
    builder = NetworkBuilder(
        temperature=network.temperature,
        c_standard=network.c_standard,
        logk_consistency_tol=(network.logk_consistency_tol
                              if logk_consistency_tol is None
                              else logk_consistency_tol),
    )
    # Components are discovered from species on demand, but to keep the
    # original component ordering, predeclare any that species do not
    # themselves introduce first.
    builder.add_components(network.components)
    for i, name in enumerate(network.species):
        comp = {
            network.components[k]: network.composition[i][k]
            for k in range(network.n_components)
            if network.composition[i][k] != 0.0
        }
        builder.add_species(name, comp)
    for j, rname in enumerate(network.reaction_names):
        stoich = {
            network.species[k]: network.stoichiometry[j][k]
            for k in range(network.n_species)
            if network.stoichiometry[j][k] != 0.0
        }
        builder.add_reaction(rname, stoich, math.exp(network.log_k[j]))
    for name, comp in add_species:
        builder.add_species(name, comp)
    for rname, stoich, K in add_reactions:  # noqa: N806
        builder.add_reaction(rname, stoich, K)
    return builder.build()
