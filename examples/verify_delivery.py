#!/usr/bin/env python3
"""Runnable end-to-end verification of the chem-equilibrium-core kernel.

Run with::

    PYTHONPATH=src python3 examples/verify_delivery.py

The script prints and asserts, in order:

1. reference ideal-solution equilibrium (A/B/AB/AB2, K=10, K=5);
2. public component balance AND equilibrium confirmation;
3. 2:1 mixing re-equilibrates; the naive volume-weighted average keeps
   the component balances but is provably not at equilibrium;
4. spike + dilution re-equilibrates while carrying all old material;
5. previously returned samples never change; provenance is recorded;
6. adding an equivalent / reversed / re-split reaction does not change
   the equilibrium, while inconsistent constants are refused at model
   acceptance (before any numerical iteration), with named reactions;
7. failed calculations do not pollute the lineage;
8. different networks compute side by side in one process.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from chem_equilibrium_core import (
    ModelError,
    NetworkBuilder,
    Workspace,
    assess,
    evolve_network,
)
from chem_equilibrium_core.errors import InputError

CHECK_COUNT = 0


def check(label: str, condition, detail="") -> None:
    global CHECK_COUNT
    CHECK_COUNT += 1
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{(' -- ' + detail) if detail else ''}")
    if not condition:
        raise SystemExit(f"verification failed at: {label} {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def reference_network(temperature=298.15, c_standard=1.0):
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


def main() -> None:
    section("1-2. Reference equilibrium and public confirmation")
    network = reference_network()
    ws = Workspace("batch")
    state = ws.equilibrate(network, [1.0, 1.5])

    expected = (0.170094, 0.227997, 0.387809, 0.442097)
    print("species order:    ", state.species_order)
    print("component order:  ", state.component_order)
    print("concentrations:   ", tuple(round(c, 6) for c in state.concentrations))
    print("independent check:", expected)
    print("temperature / c0: ", state.temperature, state.c_standard)
    check("matches independent numerical values",
          np.allclose(state.concentrations, expected, atol=1e-6))
    check("component balances close publicly",
          max(map(abs, state.component_residuals().values())) < 1e-10,
          str({k: f"{v:.2e}" for k, v in state.component_residuals().items()}))
    check("equilibrium constants satisfied publicly",
          max(abs(v) for v in state.reaction_residuals().values() if np.isfinite(v))
          < 1e-10,
          str({k: f"{v:.2e}" for k, v in state.reaction_residuals().items()}))
    check("multi-component species counts in every balance (AB2 feeds A and B)",
          abs(state.component_total_map()["A"] - 1.0) < 1e-10
          and abs(state.component_total_map()["B"] - 1.5) < 1e-10)

    section("3. Two-to-one mixing: average conserves balance but is not equilibrium")
    s1 = ws.prepare_sample(network, 1.0, {"A": 1.0, "B": 1.5})
    s2 = ws.prepare_sample(network, 1.0, {"A": 0.5, "B": 3.0})
    mixture = ws.mix_samples([(s1, 2.0), (s2, 1.0)])

    naive = (2.0 * np.array(s1.concentrations) + np.array(s2.concentrations)) / 3.0
    naive_totals = (
        2.0 * np.array(s1.component_total_concentrations())
        + np.array(s2.component_total_concentrations())
    ) / 3.0
    naive_report = assess(network, naive, naive_totals)
    print("naive average concentrations:", tuple(round(c, 6) for c in naive))
    print("mixture equilibrium:         ", tuple(round(c, 6) for c in mixture.concentrations))
    check("naive average still conserves every component",
          all(abs(v) < 1e-9 for v in naive_report.component_residuals.values()))
    check("naive average fails chemical equilibrium",
          not all(abs(v) < 1e-7 for v in naive_report.reaction_residuals.values()),
          f"residuals { {k: round(v, 4) for k, v in naive_report.reaction_residuals.items()} }")
    check("mixture result independently verified", mixture.equilibrium.report.ok)
    pooled = 2.0 * np.array(s1.component_amounts()) + np.array(s2.component_amounts())
    check("pooled component moles conserved through re-equilibration",
          np.allclose(pooled, mixture.component_amounts(), atol=1e-9))
    check("mixture differs from naive average",
          not np.allclose(mixture.concentrations, naive, atol=1e-8))

    section("4. Spike and dilution")
    spiked = ws.spike_component(mixture, "B", 0.5, final_volume=3.5)
    new_moles = np.array(spiked.component_amounts())
    print("after spike component moles:", spiked.component_amount_map())
    check("spiked sample is at verified equilibrium", spiked.equilibrium.report.ok)
    check("B amount increased exactly by spike",
          abs(new_moles[1] - (pooled[1] + 0.5)) < 1e-9)
    check("A amount carried over unchanged", abs(new_moles[0] - pooled[0]) < 1e-9)
    check("final volume recorded", spiked.volume == 3.5)

    section("5. Immutability and provenance")
    check("input sample s1 unchanged by mixing and later spike",
          np.allclose(s1.concentrations, expected, atol=1e-6))
    again = ws.get_sample(mixture.sample_id)
    check("old mixture unchanged after spike and re-queryable",
          np.array_equal(again.concentrations, mixture.concentrations))
    print("origin of mixture:", mixture.describe_origin())
    print("origin of spike:  ", spiked.describe_origin())
    chain_ids = {s.sample_id for s in ws.chain(spiked.sample_id)}
    check("spike provenance reaches both mixing inputs",
          {s1.sample_id, s2.sample_id, mixture.sample_id} <= chain_ids)

    section("6. Reaction re-expression vs inconsistent constants")
    overall = evolve_network(
        network,
        add_reactions=[("overall", {"A": -1, "B": -2, "AB2": 1}, 50.0)],
    )
    reversed_rxn = evolve_network(
        network,
        add_reactions=[("AB_rev", {"A": 1, "B": 1, "AB": -1}, 0.1)],
    )
    check("added overall reaction leaves fingerprint unchanged",
          overall.fingerprint == network.fingerprint)
    check("reversed duplicate leaves fingerprint unchanged",
          reversed_rxn.fingerprint == network.fingerprint)
    c0 = np.array(ws.equilibrate(network, [1.0, 1.5]).concentrations)
    c1 = np.array(ws.equilibrate(overall, [1.0, 1.5]).concentrations)
    c2 = np.array(ws.equilibrate(reversed_rxn, [1.0, 1.5]).concentrations)
    check("equilibrium composition identical across re-expressions",
          np.allclose(c0, c1, atol=1e-12) and np.allclose(c0, c2, atol=1e-12))
    check("equivalent network canonicalized to one object",
          ws.register_network(overall) is network)

    try:
        evolve_network(
            network,
            add_reactions=[("overall_bad", {"A": -1, "B": -2, "AB2": 1}, 40.0)],
        )
        raise SystemExit("inconsistent K=40 should have been refused")
    except ModelError as exc:
        print("refused:", exc)
        check("inconsistent constant refused before solving",
              exc.code == "inconsistent_equilibrium_constants"
              and "overall_bad" in exc.reactions and "implied K = 50" in str(exc))

    try:
        (NetworkBuilder()
         .add_species("A", {"A": 1}).add_species("B", {"B": 1})
         .add_reaction("creates_matter", {"A": -1, "B": 1}, K=2.0)
         .build())
        raise SystemExit("non-conserving reaction should have been refused")
    except ModelError as exc:
        check("non-conserving reaction named at acceptance",
              exc.code == "reaction_not_conserving"
              and "creates_matter" in exc.reactions
              and {"A", "B"} >= set(exc.components))

    section("7. Failures do not pollute prior results")
    before = set(ws.lineage.sample_ids())
    for attempt in (
        lambda: ws.prepare_sample(network, 0.0, {"A": 1.0}),
        lambda: ws.prepare_sample(network, 1.0, {"A": -1.0}),
        lambda: ws.spike_component(mixture, "B", -3.0, final_volume=2.0),
        lambda: ws.mix_samples([]),
    ):
        try:
            attempt()
            raise SystemExit("invalid operation should have raised")
        except InputError:
            pass
    check("lineage sample set unchanged after failed operations",
          set(ws.lineage.sample_ids()) == before)
    check("prior model and samples still verifiable",
          assess(network, s1.concentrations, s1.component_total_concentrations()).ok)

    section("8. Different networks side by side in one process")
    other_ws = Workspace("other")
    iso = (
        NetworkBuilder(temperature=310.0)
        .add_species("X", {"A": 1}).add_species("Y", {"A": 1})
        .add_reaction("iso", {"X": -1, "Y": 1}, K=4.0).build()
    )
    iso_state = other_ws.equilibrate(iso, [1.0])
    check("second network computes independently",
          np.allclose(iso_state.concentrations, (0.2, 0.8), atol=1e-10))
    other_sample = other_ws.prepare_sample(iso, 1.0, {"A": 1.0})
    try:
        ws.mix_samples([(s1, 1.0), (other_sample, 1.0)])
        raise SystemExit("cross-network mixing should be refused")
    except InputError as exc:
        check("cross-network mixing refused with network_mismatch",
              exc.code == "network_mismatch")
    check("workspaces keep separate lineages",
          iso_state.network.fingerprint != network.fingerprint
          and other_sample.sample_id not in ws.lineage.sample_ids())

    print(f"\nALL {CHECK_COUNT} DELIVERY CHECKS PASSED")


if __name__ == "__main__":
    main()
