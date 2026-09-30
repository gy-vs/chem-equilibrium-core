"""chem-equilibrium-core: batch solution-preparation equilibrium kernel.

Pure-Python kernel (NumPy only) for fixed-temperature, single-liquid-
phase ideal solutions.  The public call chain is::

    NetworkBuilder(...).build()          # network description, accepted or ModelError
      -> Workspace().equilibrate(...)    # direct equilibrium from component totals
      -> prepare_sample / mix_samples / spike_component  # solution preparation
      -> assess(...) / state.report      # public material + equilibrium confirmation

See ``examples/verify_delivery.py`` and ``tests/`` for runnable usage.
"""

from .errors import (
    ChemCoreError,
    EquilibriumNotConfirmed,
    InputError,
    ModelError,
)
from .model import (
    LOGK_CONSISTENCY_TOL,
    NetworkBuilder,
    Reaction,
    ReactionNetwork,
    Species,
    evolve_network,
)
from .engine import equilibrium_concentrations
from .state import EquilibriumState
from .samples import (
    Lineage,
    Sample,
    mix_samples,
    prepare_sample,
    spike_component,
)
from .verify import (
    DEFAULT_BALANCE_ATOL,
    DEFAULT_BALANCE_RTOL,
    DEFAULT_LOGK_ATOL,
    VerificationReport,
    assess,
)
from .workspace import Workspace

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # description / acceptance
    "NetworkBuilder",
    "Reaction",
    "Species",
    "ReactionNetwork",
    "evolve_network",
    # equilibrium
    "equilibrium_concentrations",
    "EquilibriumState",
    # preparation
    "Workspace",
    "Lineage",
    "Sample",
    "prepare_sample",
    "mix_samples",
    "spike_component",
    # verification
    "assess",
    "VerificationReport",
    "DEFAULT_BALANCE_ATOL",
    "DEFAULT_BALANCE_RTOL",
    "DEFAULT_LOGK_ATOL",
    "LOGK_CONSISTENCY_TOL",
    # errors
    "ChemCoreError",
    "ModelError",
    "InputError",
    "EquilibriumNotConfirmed",
]
