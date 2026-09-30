"""pytest 共享夹具：用户给出的 A / B / AB / AB2 体系。"""

import numpy as np
import pytest

from chem_equilibrium_core import Reaction, build_model

# 用户给定的理想溶液对照
REFERENCE_CONCENTRATIONS = np.array([0.170094, 0.227997, 0.387809, 0.442097])
REFERENCE_TOTALS = {"A": 1.0, "B": 1.5}

SPECIES_COMPOSITION = {
    "A": {"A": 1},
    "B": {"B": 1},
    "AB": {"A": 1, "B": 1},
    "AB2": {"A": 1, "B": 2},
}

R1_AB = lambda: Reaction("A+B=AB", {"AB": 1, "A": -1, "B": -1}, k=10.0)
R2_AB2 = lambda: Reaction("AB+B=AB2", {"AB2": 1, "AB": -1, "B": -1}, k=5.0)
R_OVERALL = lambda: Reaction("A+2B=AB2", {"AB2": 1, "A": -1, "B": -2}, k=50.0)
R_OVERALL_BAD = lambda: Reaction("A+2B=AB2", {"AB2": 1, "A": -1, "B": -2}, k=30.0)


@pytest.fixture
def composition():
    return {sp: dict(vec) for sp, vec in SPECIES_COMPOSITION.items()}


@pytest.fixture
def model(composition):
    return build_model(composition, [R1_AB(), R2_AB2()],
                       temperature=298.15, standard_concentration=1.0)
