"""补加组分并改变最终体积：旧样品不变，新样品物料、体积、溯源都正确。"""

import numpy as np

from chem_equilibrium_core import (SampleOperationError, Workspace,
                                   add_component, equilibrate_totals,
                                   make_sample)


def test_add_component_mass_balance(model):
    s0 = make_sample(equilibrate_totals(model, [1.0, 1.5], volume_l=1.0),
                     "s0")
    # 投入 0.5 mol 纯 B，稀释/定容到 2 L
    result = add_component(s0, {"B": 0.5}, final_volume_l=2.0)
    bA = 1.0 * 1.0 / 2.0
    bB = (1.5 * 1.0 + 0.5) / 2.0
    np.testing.assert_allclose(result.total_concentrations, [bA, bB],
                               atol=1e-12)
    assert result.volume_l == 2.0
    assert result.origin.operation == "add_component"
    assert result.origin.parents == ("s0",)

    c = dict(zip(result.species, result.concentrations))
    assert abs(c["A"] + c["AB"] + c["AB2"] - bA) < 1e-9
    assert abs(c["B"] + c["AB"] + 2 * c["AB2"] - bB) < 1e-9
    assert abs(c["AB"] / (c["A"] * c["B"]) - 10.0) < 1e-7
    assert abs(c["AB2"] / (c["AB"] * c["B"]) - 5.0) < 1e-7
    assert result.verification.passed


def test_addition_units_mmol(model):
    s0 = make_sample(equilibrate_totals(model, [1.0, 1.5], volume_l=1.0))
    r1 = add_component(s0, {"B": 500.0}, addition_unit="mmol",
                       final_volume_l=2.0)
    r2 = add_component(s0, {"B": 0.5}, final_volume_l=2.0)
    np.testing.assert_allclose(r1.concentrations, r2.concentrations,
                               atol=1e-13)


def test_old_sample_remains_queryable_after_addition(model):
    ws = Workspace()
    ws.add_model(model)
    s0 = ws.equilibrate(model, [1.0, 1.5], volume_l=1.0, sample_id="base")
    before = ws.get_sample("base").concentrations.copy()

    s1 = ws.add_component("base", {"A": 1.0}, final_volume_l=1.5,
                          sample_id_out="fed")
    np.testing.assert_array_equal(ws.get_sample("base").concentrations, before)
    assert ws.get_sample("base").volume_l == 1.0
    assert ws.lineage("fed")["parents"][0]["sample_id"] == "base"
    assert s1.origin.detail["additions"] == {"A": 1.0}


def test_add_unknown_component_rejected(model):
    s0 = make_sample(equilibrate_totals(model, [1.0, 1.5]))
    try:
        add_component(s0, {"X": 1.0}, final_volume_l=1.0)
        raise AssertionError("体系外组分必须被拒绝")
    except SampleOperationError as exc:
        assert "X" in str(exc)


def test_add_then_mix_chain(model):
    ws = Workspace()
    ws.add_model(model)
    a = ws.equilibrate(model, [1.0, 1.0], volume_l=1.0, sample_id="a")
    b = ws.equilibrate(model, [1.0, 2.0], volume_l=1.0, sample_id="b")
    fed = ws.add_component("b", {"B": 0.5}, final_volume_l=1.0,
                           sample_id_out="b-fed")
    m = ws.mix(["a", "b-fed"], volumes_l=[1.0, 1.0], sample_id="mix")
    bA = 1.0
    bB = (1.0 + 2.5) / 2.0
    np.testing.assert_allclose(m.total_concentrations, [bA, bB], atol=1e-12)
    assert m.verification.passed
    tree = ws.lineage("mix")
    parents = {n["sample_id"] for n in tree["parents"]}
    assert parents == {"a", "b-fed"}
