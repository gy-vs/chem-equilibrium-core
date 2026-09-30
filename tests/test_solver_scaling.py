"""Solver robustness across extreme feed concentration ratios.

Regression test for the case where component totals span ~10 orders of
magnitude (e.g. 1e5 mol/L bulk component next to 1e-5 mol/L trace
component).  The trace component's balance must close just as tightly
(relative to its own total) as the bulk component's, and the result must
still pass the independent public verification.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from chem_equilibrium_core import NetworkBuilder, Workspace, assess
from chem_equilibrium_core.linalg_utils import nullspace
from chem_equilibrium_core.model import RANK_TOL


class TestExtremeScaling(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace("scaling")

    def _build(self, rng, nc, extra):
        ns = nc + extra
        while True:
            comp = rng.integers(0, 3, size=(ns, nc)).astype(float)
            comp[:nc] = np.eye(nc)
            if (
                np.linalg.matrix_rank(comp, tol=1e-10) == nc
                and np.all(comp.sum(axis=1) > 0)
            ):
                break
        builder = NetworkBuilder()
        for i in range(ns):
            builder.add_species(
                f"S{i}",
                {f"E{k}": comp[i, k] for k in range(nc) if comp[i, k] != 0},
            )
        for j, vec in enumerate(nullspace(comp.T, RANK_TOL)):
            stoich = {
                f"S{i}": float(vec[i])
                for i in range(ns)
                if abs(vec[i]) > 1e-9
            }
            builder.add_reaction(
                f"r{j}", stoich, K=float(10.0 ** rng.uniform(-4, 4))
            )
        return builder.build(), comp

    def test_random_networks_with_extreme_feed_ratios(self):
        rng = np.random.default_rng(20260929)
        worst_relative = 0.0
        count = 0
        for _ in range(40):
            nc = int(rng.integers(1, 5))
            network, _ = self._build(rng, nc, int(rng.integers(1, 6)))
            for _ in range(5):
                totals = np.where(
                    rng.random(nc) < 0.15,
                    0.0,
                    10.0 ** rng.uniform(-6, 6, size=nc),
                )
                state = self.ws.equilibrate(network, totals)
                self.assertTrue(state.report.ok, state.report.violations)
                c = np.asarray(state.concentrations)
                self.assertTrue(np.all(np.isfinite(c)))
                self.assertTrue(np.all(c >= 0.0))
                # Independently verify as well.
                report = assess(network, c, totals)
                self.assertTrue(report.ok, report.violations)
                for k, comp_name in enumerate(network.components):
                    if totals[k] > 0:
                        # Match the public verifier's scale convention
                        # (atol + rtol*|total|): normalize by max(|total|, 1).
                        scale = max(abs(totals[k]), 1.0)
                        rel = abs(state.component_residuals()[comp_name]) / scale
                        worst_relative = max(worst_relative, rel)
                count += 1
        self.assertGreater(count, 0)
        self.assertLess(worst_relative, 1e-8, worst_relative)

    def test_explicit_bulk_and_trace_components(self):
        # One bulk component at 1e5 and a trace component at 1e-5.
        network = (
            NetworkBuilder()
            .add_species("X", {"A": 1})
            .add_species("Y", {"B": 1})
            .add_species("Z", {"A": 1, "B": 1})
            .add_reaction("bind", {"X": -1, "Y": -1, "Z": 1}, K=100.0)
            .build()
        )
        totals = np.array([1.0e5, 1.0e-5])
        state = self.ws.equilibrate(network, totals)
        self.assertTrue(state.report.ok, state.report.violations)
        c = state.concentrations
        # B is the limiting trace ingredient and ends up almost fully bound.
        self.assertAlmostEqual(c[1] + c[2], 1e-5, delta=1e-12)
        self.assertGreater(c[2], 0.9e-5)
        # Bulk balance closes absolutely within the public band.
        self.assertLess(abs(state.component_residuals()["A"]), 1e-9)
        self.assertLess(abs(state.component_residuals()["B"]), 1e-12)

    def test_very_large_totals_remain_finite(self):
        network = (
            NetworkBuilder()
            .add_species("A", {"A": 1})
            .add_species("B", {"B": 1})
            .add_species("AB", {"A": 1, "B": 1})
            .add_reaction("r", {"A": -1, "B": -1, "AB": 1}, K=2.0)
            .build()
        )
        state = self.ws.equilibrate(network, [1.0e8, 1.0e8])
        self.assertTrue(state.report.ok, state.report.violations)
        self.assertTrue(np.all(np.isfinite(state.concentrations_array())))


if __name__ == "__main__":
    unittest.main()
