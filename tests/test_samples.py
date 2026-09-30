"""Solution-preparation chain: prepare, mix, spike, provenance, immutability."""

import math
import unittest

import numpy as np

from _fixtures import REFERENCE_TOTALS, build_reference_network
from chem_equilibrium_core import Workspace, assess
from chem_equilibrium_core.errors import InputError


class TestSamples(unittest.TestCase):
    def setUp(self):
        self.network = build_reference_network()
        self.ws = Workspace("prep")
        # 1 L samples: the reference condition and a second condition.
        self.s1 = self.ws.prepare_sample(
            self.network, 1.0, {"A": 1.0, "B": 1.5}
        )
        self.s2 = self.ws.prepare_sample(
            self.network, 1.0, {"A": 0.5, "B": 3.0}
        )

    def test_prepare_matches_reference(self):
        np.testing.assert_allclose(
            self.s1.concentrations,
            (0.170094, 0.227997, 0.387809, 0.442097),
            atol=1e-6,
        )

    def test_sample_carries_amounts_volume_and_balances(self):
        self.assertEqual(self.s1.volume, 1.0)
        amounts = np.array(self.s1.amounts)
        np.testing.assert_allclose(
            amounts / self.s1.volume, self.s1.concentrations, atol=1e-15
        )
        comp = np.array(self.s1.component_amounts())
        np.testing.assert_allclose(comp, [1.0, 1.5], atol=1e-9)

    def test_two_to_one_mix_re_equilibrates(self):
        mixture = self.ws.mix_samples([(self.s1, 2.0), (self.s2, 1.0)])
        self.assertEqual(mixture.volume, 3.0)

        # Independently verified state.
        self.assertTrue(mixture.equilibrium.report.ok)
        totals = np.array(mixture.component_total_concentrations())
        report = assess(self.network, mixture.concentrations, totals)
        self.assertTrue(report.ok, report.violations)

        # Material: component moles are the pool sum exactly.
        pooled = 2.0 * np.array(self.s1.component_amounts()) + np.array(
            self.s2.component_amounts()
        )
        np.testing.assert_allclose(mixture.component_amounts(), pooled, atol=1e-9)

    def test_volume_weighted_average_conserves_but_is_not_equilibrium(self):
        # The naive pool-averaged species concentrations ...
        naive = (
            2.0 * np.array(self.s1.concentrations)
            + np.array(self.s2.concentrations)
        ) / 3.0
        totals = (
            2.0 * np.array(self.s1.component_total_concentrations())
            + np.array(self.s2.component_total_concentrations())
        ) / 3.0
        report = assess(self.network, naive, totals)
        # ... keep every component balance ...
        self.assertTrue(
            all(abs(v) < 1e-9 for v in report.component_residuals.values())
        )
        # ... but fail chemical equilibrium.
        self.assertFalse(
            all(abs(v) < 1e-7 for v in report.reaction_residuals.values())
        )
        # The library's mixture differs from that naive average.
        mixture = self.ws.mix_samples([(self.s1, 2.0), (self.s2, 1.0)])
        self.assertFalse(np.allclose(mixture.concentrations, naive, atol=1e-6))

    def test_old_samples_are_not_mutated_by_mix_or_spike(self):
        before_s1 = np.array(self.s1.concentrations)
        before_s2 = np.array(self.s2.concentrations)
        mixture = self.ws.mix_samples([(self.s1, 2.0), (self.s2, 1.0)])
        spiked = self.ws.spike_component(
            mixture, "B", 0.5, final_volume=3.5
        )
        np.testing.assert_array_equal(self.s1.concentrations, before_s1)
        np.testing.assert_array_equal(self.s2.concentrations, before_s2)
        mix_conc = np.array(mixture.concentrations)
        # Re-querying the mixture after a later spike gives the same values.
        fetched = self.ws.get_sample(mixture.sample_id)
        np.testing.assert_array_equal(fetched.concentrations, mix_conc)
        self.assertEqual(spiked.volume, 3.5)

    def test_spike_carries_old_material_and_added_component(self):
        mixture = self.ws.mix_samples([(self.s1, 2.0), (self.s2, 1.0)])
        base_moles = np.array(mixture.component_amounts())
        spiked = self.ws.spike_component(
            mixture, "B", 0.5, final_volume=3.5
        )
        new_moles = np.array(spiked.component_amounts())
        np.testing.assert_allclose(new_moles[0], base_moles[0], atol=1e-9)
        self.assertAlmostEqual(new_moles[1], base_moles[1] + 0.5, places=9)
        self.assertTrue(spiked.equilibrium.report.ok)

    def test_spike_without_volume_change_default_check(self):
        spiked = self.ws.spike_component(
            self.s1, "A", 0.2, final_volume=1.0
        )
        moles = np.array(spiked.component_amounts())
        self.assertAlmostEqual(moles[0], 1.2, places=9)
        self.assertAlmostEqual(moles[1], 1.5, places=9)

    def test_provenance_records_inputs(self):
        mixture = self.ws.mix_samples([(self.s1, 2.0), (self.s2, 1.0)])
        self.assertEqual(
            set(mixture.parent_ids),
            {self.s1.sample_id, self.s2.sample_id},
        )
        self.assertIn("mixed from", mixture.describe_origin())

        chain = self.ws.chain(mixture.sample_id)
        ids = [sample.sample_id for sample in chain]
        self.assertEqual(ids[0], mixture.sample_id)
        self.assertIn(self.s1.sample_id, ids)
        self.assertIn(self.s2.sample_id, ids)

        spiked = self.ws.spike_component(mixture, "B", 0.5, final_volume=3.5)
        self.assertEqual(spiked.parent_ids, (mixture.sample_id,))
        ancestors = self.ws.ancestors(spiked.sample_id)
        ancestor_ids = {sample.sample_id for sample in ancestors}
        self.assertEqual(
            ancestor_ids,
            {mixture.sample_id, self.s1.sample_id, self.s2.sample_id},
        )

    def test_prepare_origin_recorded(self):
        self.assertIn("prepared from feed", self.s1.describe_origin())

    def test_failed_calculation_does_not_pollute_lineage(self):
        before = set(self.ws.lineage.sample_ids())
        with self.assertRaises(InputError):
            self.ws.prepare_sample(self.network, 0.0, {"A": 1.0})
        with self.assertRaises(InputError):
            self.ws.spike_component(self.s1, "B", -1.0, final_volume=2.0)
        with self.assertRaises(InputError):
            self.ws.mix_samples([])
        after = set(self.ws.lineage.sample_ids())
        self.assertEqual(before, after)
        # Old samples still usable.
        self.assertTrue(
            assess(
                self.network,
                self.s1.concentrations,
                self.s1.component_total_concentrations(),
            ).ok
        )

    def test_mixing_different_networks_rejected(self):
        other_ws = Workspace("other")
        other = (
            NetworkBuilderHelper.isomer_network(other_ws)
        )
        with self.assertRaises(InputError) as ctx:
            self.ws.mix_samples([(self.s1, 1.0), (other, 1.0)])
        self.assertEqual(ctx.exception.code, "network_mismatch")

    def test_samples_are_frozen(self):
        with self.assertRaises(Exception):
            self.s1.volume = 5.0  # type: ignore[misc]


class NetworkBuilderHelper:
    @staticmethod
    def isomer_network(workspace):
        from chem_equilibrium_core import NetworkBuilder

        net = (
            NetworkBuilder(temperature=310.0)
            .add_species("X", {"A": 1})
            .add_species("Y", {"A": 1})
            .add_reaction("iso", {"X": -1, "Y": 1}, K=4.0)
            .build()
        )
        return workspace.prepare_sample(net, 1.0, {"A": 1.0})


if __name__ == "__main__":
    unittest.main()
