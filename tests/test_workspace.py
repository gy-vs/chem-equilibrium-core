"""Multiple networks in one process and reaction re-expression scenarios."""

import threading
import unittest

import numpy as np

from _fixtures import build_reference_network
from chem_equilibrium_core import NetworkBuilder, Workspace, evolve_network
from chem_equilibrium_core.errors import InputError


def isomer_network(temperature=298.15, K=3.0):
    return (
        NetworkBuilder(temperature=temperature)
        .add_species("X", {"A": 1})
        .add_species("Y", {"A": 1})
        .add_reaction("iso", {"X": -1, "Y": 1}, K=K)
        .build()
    )


class TestWorkspaceIsolationAndReExpression(unittest.TestCase):
    def test_equivalent_networks_share_canonical_object(self):
        ws = Workspace()
        base = build_reference_network()
        equivalent = evolve_network(
            base,
            add_reactions=[("overall", {"A": -1, "B": -2, "AB2": 1}, 50.0)],
        )
        self.assertIs(ws.register_network(base), base)
        self.assertIs(ws.register_network(equivalent), base)
        self.assertEqual(len(ws.networks()), 1)

    def test_different_networks_coexist_and_compute(self):
        ws = Workspace("combined")
        ref = build_reference_network()
        iso = isomer_network()
        ref_state = ws.equilibrate(ref, [1.0, 1.5])
        iso_state = ws.equilibrate(iso, [2.0])
        np.testing.assert_allclose(
            ref_state.concentrations,
            (0.170094, 0.227997, 0.387809, 0.442097),
            atol=1e-6,
        )
        np.testing.assert_allclose(iso_state.concentrations, (0.5, 1.5), atol=1e-10)
        self.assertEqual(len(ws.networks()), 2)

    def test_separate_workspaces_do_not_share_state(self):
        ws_a = Workspace("a")
        ws_b = Workspace("b")
        net = build_reference_network()
        ws_a.prepare_sample(net, 1.0, {"A": 1.0, "B": 1.5})
        ws_b.prepare_sample(net, 2.0, {"A": 2.0, "B": 3.0})
        self.assertEqual(len(ws_a.lineage.sample_ids()), 1)
        self.assertEqual(len(ws_b.lineage.sample_ids()), 1)
        with self.assertRaises(InputError):
            ws_b.get_sample(ws_a.lineage.sample_ids()[0])

    def test_concurrent_workspace_use(self):
        ws = Workspace("parallel")
        errors = []

        def worker(totals, temperature):
            try:
                net = build_reference_network(temperature=temperature)
                for _ in range(20):
                    state = ws.equilibrate(net, totals)
                    self.assertTrue(state.report.ok)
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=([1.0, 1.5], 298.15)),
            threading.Thread(target=worker, args=([0.5, 3.0], 298.15)),
            threading.Thread(target=worker, args=([1.0, 0.0], 298.15)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_reaction_split_with_intermediate(self):
        # A world containing only the overall reaction is a genuinely
        # different physical system from one that also contains the
        # intermediate AB: the fingerprints differ, and each world must
        # simply be self-consistent at its own equilibrium.
        direct = (
            NetworkBuilder()
            .add_species("A", {"A": 1})
            .add_species("B", {"B": 1})
            .add_species("AB2", {"A": 1, "B": 2})
            .add_reaction("overall", {"A": -1, "B": -2, "AB2": 1}, K=50.0)
            .build()
        )
        split = (
            NetworkBuilder()
            .add_species("A", {"A": 1})
            .add_species("B", {"B": 1})
            .add_species("AB", {"A": 1, "B": 1})
            .add_species("AB2", {"A": 1, "B": 2})
            .add_reaction("step1", {"A": -1, "B": -1, "AB": 1}, K=10.0)
            .add_reaction("step2", {"AB": -1, "B": -1, "AB2": 1}, K=5.0)
            .build()
        )
        self.assertNotEqual(direct.fingerprint, split.fingerprint)
        ws = Workspace()
        c_direct = ws.equilibrate(direct, [1.0, 1.5]).concentrations
        c_split = ws.equilibrate(split, [1.0, 1.5]).concentrations
        # Both states satisfy their own balances and constants publicly.
        self.assertTrue(ws.verify(direct, c_direct, [1.0, 1.5]).ok)
        self.assertTrue(ws.verify(split, c_split, [1.0, 1.5]).ok)
        # But the mono-complex available only in the split world changes
        # the distribution: the two systems must not be claimed equal.
        self.assertFalse(
            np.isclose(c_direct[2], c_split[3], atol=1e-6),
            (c_direct, c_split),
        )

    def test_adding_inconsistent_step_rejected_before_use(self):
        # Split whose step constants do not multiply to the overall value.
        def build():
            return (
                NetworkBuilder()
                .add_species("A", {"A": 1})
                .add_species("B", {"B": 1})
                .add_species("AB", {"A": 1, "B": 1})
                .add_species("AB2", {"A": 1, "B": 2})
                .add_reaction("step1", {"A": -1, "B": -1, "AB": 1}, K=10.0)
                .add_reaction("step2", {"AB": -1, "B": -1, "AB2": 1}, K=5.0)
                .add_reaction("overall", {"A": -1, "B": -2, "AB2": 1}, K=51.0)
                .build()
            )
        from chem_equilibrium_core.errors import ModelError

        with self.assertRaises(ModelError) as ctx:
            build()
        self.assertEqual(
            ctx.exception.code, "inconsistent_equilibrium_constants"
        )

    def test_standard_concentration_part_of_identity(self):
        net1 = build_reference_network(c_standard=1.0)
        net2 = build_reference_network(c_standard=0.989)
        self.assertNotEqual(net1.fingerprint, net2.fingerprint)

    def test_get_network_by_short_id(self):
        ws = Workspace()
        net = ws.register_network(build_reference_network())
        fetched = ws.get_network(net.network_id)
        self.assertIs(fetched, net)
        with self.assertRaises(InputError):
            ws.get_network("does-not-exist")


if __name__ == "__main__":
    unittest.main()
