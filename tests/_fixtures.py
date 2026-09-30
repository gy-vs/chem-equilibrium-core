"""Shared fixtures for the reference A/B/AB/AB2 chemistry."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chem_equilibrium_core import NetworkBuilder  # noqa: E402

# Reference numerical values from the independent ideal-solution check:
# A + B <-> AB, K=10; AB + B <-> AB2, K=5; c0=1 mol/L;
# total A=1, total B=1.5 mol/L.
REFERENCE_CONCENTRATIONS = (0.170094, 0.227997, 0.387809, 0.442097)
REFERENCE_TOTALS = (1.0, 1.5)


def build_reference_network(temperature=298.15, c_standard=1.0):
    return (
        NetworkBuilder(temperature=temperature, c_standard=c_standard)
        .add_species("A", {"A": 1})
        .add_species("B", {"B": 1})
        .add_species("AB", {"A": 1, "B": 1})
        .add_species("AB2", {"A": 1, "B": 2})
        .add_reaction("AB_fwd", {"A": -1, "B": -1, "AB": 1}, K=10.0)
        .add_reaction("AB2_fwd", {"AB": -1, "B": -1, "AB2": 1}, K=5.0)
        .build()
    )
