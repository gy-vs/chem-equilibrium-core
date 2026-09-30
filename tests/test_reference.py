"""对照计算：复现用户给出的四个平衡浓度，并独立复核收支与平衡。"""

import numpy as np

from chem_equilibrium_core import (build_model, equilibrate_totals,
                                   make_sample, verify_composition)
from tests.conftest import (R1_AB, R2_AB2, REFERENCE_CONCENTRATIONS,
                            REFERENCE_TOTALS)


def test_reference_concentrations(model):
    result = equilibrate_totals(model, REFERENCE_TOTALS, volume_l=1.0)
    np.testing.assert_allclose(result.concentrations,
                               REFERENCE_CONCENTRATIONS, atol=5e-7)


def test_result_carries_system_metadata(model):
    result = equilibrate_totals(model, REFERENCE_TOTALS, volume_l=2.0)
    assert result.temperature_k == model.temperature_k
    assert result.standard_concentration == 1.0
    assert result.species == model.species
    assert result.components == model.components
    assert result.system_id == model.system_id
    assert result.network_id == model.network_id
    assert result.volume_l == 2.0


def test_budget_is_public_and_componentwise(model):
    result = equilibrate_totals(model, {"A": 1.0, "B": 1.5})
    report = result.verification
    assert report.passed
    # 逐组分收支，而不是只看总量“差不多”
    by_comp = {r.component: r for r in report.balance_records}
    assert by_comp["A"].supplied == 1.0
    assert abs(by_comp["A"].accounted - (result.concentrations[0]
                                         + result.concentrations[2]
                                         + result.concentrations[3])) < 1e-12
    assert abs(by_comp["B"].accounted - (result.concentrations[1]
                                         + result.concentrations[2]
                                         + 2 * result.concentrations[3])) < 1e-12
    for r in report.balance_records:
        assert r.relative_residual < 1e-10
    # 两条反应各自满足 K
    by_rxn = {r.reaction_id: r for r in report.equilibrium_records}
    assert abs(by_rxn["A+B=AB"].ln_k - np.log(10)) < 1e-12
    assert abs(by_rxn["AB+B=AB2"].ln_k - np.log(5)) < 1e-12
    for r in report.equilibrium_records:
        assert abs(r.residual) < 1e-9
        assert abs(r.factor_residual - 1.0) < 1e-8


def test_external_program_can_recheck_independently(model):
    """不依赖库的 passed 标志：另一段程序按公开数据自己算。"""

    result = equilibrate_totals(model, [1.0, 1.5])
    c = dict(zip(result.species, result.concentrations))
    assert abs(c["A"] + c["AB"] + c["AB2"] - 1.0) < 1e-9
    assert abs(c["B"] + c["AB"] + 2 * c["AB2"] - 1.5) < 1e-9
    assert abs(c["AB"] / (c["A"] * c["B"]) - 10.0) < 1e-7
    assert abs(c["AB2"] / (c["AB"] * c["B"]) - 5.0) < 1e-7

    # 直接调用公开复核函数也应得到相同结论
    rep = verify_composition(model, result.concentrations,
                             result.total_concentrations)
    assert rep.passed


def test_as_dict_serializable(model):
    result = equilibrate_totals(model, REFERENCE_TOTALS)
    data = result.as_dict()
    assert data["species"] == ["A", "B", "AB", "AB2"]
    assert data["verification"]["passed"] is True
    assert len(data["verification"]["balance"]) == 2
    s = make_sample(result, "s0")
    assert s.concentration_map["AB2"] == s.concentrations[3]
