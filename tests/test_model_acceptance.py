"""Model acceptance: descriptions rejected before any numeric iteration."""

import unittest

from _fixtures import build_reference_network
from chem_equilibrium_core import NetworkBuilder, evolve_network
from chem_equilibrium_core.errors import ModelError


def build_simple():
    return (
        NetworkBuilder()
        .add_species("A", {"A": 1})
        .add_species("B", {"B": 1})
    )


class TestModelAcceptance(unittest.TestCase):
    def assertRejects(self, fn, code):  # noqa: N802
        try:
            fn()
        except ModelError as exc:
            self.assertEqual(exc.code, code, exc)
            return exc
        self.fail(f"expected ModelError({code})")

    def test_equivalent_added_reaction_accepted_and_same_fingerprint(self):
        base = build_reference_network()
        extended = evolve_network(
            base,
            add_reactions=[("overall", {"A": -1, "B": -2, "AB2": 1}, 50.0)],
        )
        self.assertEqual(extended.fingerprint, base.fingerprint)

    def test_reverse_duplicate_reaction_accepted(self):
        base = build_reference_network()
        extended = evolve_network(
            base,
            add_reactions=[("AB_rev", {"A": 1, "B": 1, "AB": -1}, 0.1)],
        )
        self.assertEqual(extended.fingerprint, base.fingerprint)

    def test_inconsistent_overall_constant_rejected_with_names(self):
        base = build_reference_network()
        err = self.assertRejects(
            lambda: evolve_network(
                base,
                add_reactions=[("overall_bad",
                                {"A": -1, "B": -2, "AB2": 1}, 40.0)],
            ),
            "inconsistent_equilibrium_constants",
        )
        self.assertIn("overall_bad", err.reactions)
        self.assertIn("implied K = 50", str(err))
        # Original network object remains usable.
        self.assertEqual(len(base.reaction_names), 2)

    def test_inconsistent_reverse_constant_rejected(self):
        base = build_reference_network()
        self.assertRejects(
            lambda: evolve_network(
                base,
                add_reactions=[("rev_bad", {"A": 1, "B": 1, "AB": -1}, 0.2)],
            ),
            "inconsistent_equilibrium_constants",
        )

    def test_reaction_must_conserve_every_component(self):
        err = self.assertRejects(
            lambda: build_simple()
            .add_reaction("bad", {"A": -1, "B": 1}, K=1.0)
            .build(),
            "reaction_not_conserving",
        )
        self.assertEqual(err.reactions, ("bad",))
        self.assertEqual(set(err.components), {"A", "B"})

    def test_reaction_referencing_unknown_species_rejected(self):
        err = self.assertRejects(
            lambda: build_simple()
            .add_reaction("r", {"A": -1, "B": -1, "X": 1}, K=1.0)
            .build(),
            "reaction_unknown_species",
        )
        self.assertEqual(err.species, ("X",))

    def test_underdetermined_species_named(self):
        err = self.assertRejects(
            lambda: (
                NetworkBuilder()
                .add_species("A", {"A": 1})
                .add_species("B", {"B": 1})
                .add_species("AB", {"A": 1, "B": 1})
                .build()
            ),
            "underdetermined_species",
        )
        self.assertEqual(set(err.species), {"A", "B", "AB"})

    def test_dependent_components_named(self):
        err = self.assertRejects(
            lambda: (
                NetworkBuilder()
                .add_components(["A", "B", "AplusB"])
                .add_species("A", {"A": 1})
                .add_species("B", {"B": 1})
                .add_reaction("r", {"A": -1, "B": 1}, K=1.0)
                .build()
            ),
            "dependent_components",
        )
        self.assertIn("AplusB", err.components)

    def test_unused_component_rejected(self):
        self.assertRejects(
            lambda: (
                NetworkBuilder()
                .add_components(["A", "S"])
                .add_species("A", {"A": 1})
                .build()
            ),
            "dependent_components",
        )

    def test_non_positive_constant_rejected(self):
        for bad_k in (0.0, -1.0, float("nan"), float("inf")):
            self.assertRejects(
                lambda bad_k=bad_k: build_simple()
                .add_reaction("r", {"A": -1, "B": 1}, K=bad_k)
                .build(),
                "bad_equilibrium_constant",
            )

    def test_negative_composition_rejected(self):
        self.assertRejects(
            lambda: NetworkBuilder().add_species("A", {"A": -1}),
            "negative_composition",
        )

    def test_species_without_components_rejected(self):
        self.assertRejects(
            lambda: NetworkBuilder().add_species("A"),
            "empty_species",
        )

    def test_duplicate_names_within_a_kind_rejected(self):
        self.assertRejects(
            lambda: build_simple().add_species("A", {"A": 1}),
            "duplicate_species",
        )

    def test_species_and_component_may_share_a_name(self):
        network = build_reference_network()
        self.assertEqual(network.species[0], "A")
        self.assertEqual(network.components[0], "A")

    def test_bad_temperature_and_standard_concentration(self):
        self.assertRejects(lambda: NetworkBuilder(temperature=0), "bad_temperature")
        self.assertRejects(
            lambda: NetworkBuilder(c_standard=-1), "bad_standard_concentration"
        )

    def test_determinacy_for_zero_reaction_network(self):
        # One species, one component: no reaction needed.
        network = (
            NetworkBuilder().add_species("A", {"A": 1}).build()
        )
        self.assertEqual(network.n_reactions, 0)

    def test_no_linalg_exception_type_escapes(self):
        # Malformed builds surface as ModelError, never as a numpy
        # linear-algebra error.
        import numpy

        bad_builds = [
            lambda: build_simple().add_reaction("r", {"A": -1, "B": 1}, 1.0).build(),
            lambda: NetworkBuilder().build(),
        ]
        for fn in bad_builds:
            try:
                fn()
            except ModelError:
                continue
            except numpy.linalg.LinAlgError:
                self.fail("raw LinAlgError escaped model acceptance")
            self.fail("expected a model rejection")


if __name__ == "__main__":
    unittest.main()
