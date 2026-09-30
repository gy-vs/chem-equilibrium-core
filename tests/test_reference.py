"""Reference numerical case and public verification."""

import math
import unittest

import numpy as np

from _fixtures import (
    REFERENCE_CONCENTRATIONS,
    REFERENCE_TOTALS,
    build_reference_network,
)
from chem_equilibrium_core import (
    DEFAULT_BALANCE_ATOL,
    EquilibriumState,
    NetworkBuilder,
    Workspace,
    assess,
)
from chem_equilibrium_core.errors import InputError


class TestReferenceEquilibrium(unittest.TestCase):
    def setUp(self):
        self.network = build_reference_network()
        self.workspace = Workspace("reference")

    def test_matches_independent_numerical_check(self):
        state = self.workspace.equilibrate(self.network, list(REFERENCE_TOTALS))
        np.testing.assert_allclose(
            state.concentrations, REFERENCE_CONCENTRATIONS, atol=1e-6
        )

    def test_mapping_totals_and_species_order(self):
        state = self.workspace.equilibrate(
            self.network, {"A": 1.0, "B": 1.5}
        )
        self.assertEqual(state.species_order, ("A", "B", "AB", "AB2"))
        self.assertEqual(state.component_order, ("A", "B"))
        self.assertEqual(state.temperature, 298.15)
        self.assertEqual(state.c_standard, 1.0)
        np.testing.assert_allclose(
            list(state.concentration_map().values()),
            REFERENCE_CONCENTRATIONS,
            atol=1e-6,
        )

    def test_equilibrium_constants_are_satisfied(self):
        state = self.workspace.equilibrate(self.network, list(REFERENCE_TOTALS))
        for name, resid in state.reaction_residuals().items():
            self.assertLess(abs(resid), 1e-10, name)

    def test_every_component_balance_closes_to_solver_precision(self):
        state = self.workspace.equilibrate(self.network, list(REFERENCE_TOTALS))
        for name, resid in state.component_residuals().items():
            self.assertLess(abs(resid), 1e-10, name)

    def test_state_is_frozen_and_arrays_read_only(self):
        state = self.workspace.equilibrate(self.network, list(REFERENCE_TOTALS))
        with self.assertRaises(Exception):
            state.concentrations = (0.25,) * 4  # type: ignore[misc]
        arr = state.concentrations_array()
        with self.assertRaises(ValueError):
            arr[0] = 0.0

    def test_assess_rejects_wrong_totals_but_keeps_balance_info(self):
        state = self.workspace.equilibrate(self.network, list(REFERENCE_TOTALS))
        report = assess(self.network, state.concentrations, [1.0, 1.4])
        self.assertFalse(report.ok)
        self.assertTrue(any("B" in v for v in report.violations))

    def test_assess_accepts_mapping_input(self):
        state = self.workspace.equilibrate(
            self.network, list(REFERENCE_TOTALS)
        )
        report = assess(
            self.network,
            state.concentration_map(),
            state.component_total_map(),
        )
        self.assertTrue(report.ok, report.violations)

    def test_negative_concentrations_fail_and_are_not_repaired(self):
        report = assess(
            self.network, [-0.1, 0.4, 0.4, 0.3], [1.0, 1.5]
        )
        self.assertFalse(report.ok)
        self.assertFalse(report.concentration_valid)
        self.assertTrue(any("negative" in v for v in report.violations))

    def test_wrong_shape_is_input_error_not_linalg_error(self):
        with self.assertRaises(InputError):
            self.workspace.equilibrate(self.network, [1.0])
        with self.assertRaises(InputError):
            assess(self.network, [0.1, 0.2], [1.0, 1.5])

    def test_boundary_feed_zero_component(self):
        state = self.workspace.equilibrate(self.network, {"A": 1.0, "B": 0.0})
        self.assertTrue(state.report.ok)
        np.testing.assert_allclose(
            state.concentrations, [1.0, 0.0, 0.0, 0.0], atol=1e-12
        )
        self.assertEqual(
            set(state.report.boundary_reactions), {"AB_fwd", "AB2_fwd"}
        )

    def test_all_zero_feed(self):
        state = self.workspace.equilibrate(self.network, [0.0, 0.0])
        self.assertTrue(state.report.ok)
        np.testing.assert_array_equal(state.concentrations, np.zeros(4))

    def test_direct_constants_equivalent(self):
        # The two-reaction description versus a one-reaction isomer-style
        # check: K1*K2 must equal the overall constant.
        overall = (
            NetworkBuilder()
            .add_species("A", {"A": 1})
            .add_species("B", {"B": 1})
            .add_species("AB", {"A": 1, "B": 1})
            .add_species("AB2", {"A": 1, "B": 2})
            .add_reaction("AB_fwd", {"A": -1, "B": -1, "AB": 1}, K=10.0)
            .add_reaction("AB2_fwd", {"AB": -1, "B": -1, "AB2": 1}, K=5.0)
            .add_reaction("overall", {"A": -1, "B": -2, "AB2": 1}, K=50.0)
            .build()
        )
        self.assertEqual(overall.fingerprint, self.network.fingerprint)
        ws = Workspace()
        c_a = ws.equilibrate(self.network, [1.0, 1.5]).concentrations
        c_b = ws.equilibrate(overall, [1.0, 1.5]).concentrations
        np.testing.assert_allclose(c_a, c_b, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
