"""混合：体积加权平均守恒但不是新平衡，库必须重新平衡且保留物料与体积。"""

import numpy as np

from chem_equilibrium_core import (SampleOperationError, Workspace,
                                   equilibrate_totals, make_sample, mix_samples,
                                   verify_composition)


def test_mix_2_to_1_rebalances(model):
    s1 = make_sample(equilibrate_totals(model, {"A": 1.0, "B": 1.5},
                                        volume_l=2.0), "s1")
    s2 = make_sample(equilibrate_totals(model, {"A": 0.5, "B": 3.0},
                                        volume_l=1.0), "s2")

    result = mix_samples([s1, s2], volumes_l=[2.0, 1.0])
    # 2:1 混合后的总量
    bA = (2.0 * 1.0 + 1.0 * 0.5) / 3.0
    bB = (2.0 * 1.5 + 1.0 * 3.0) / 3.0
    assert result.volume_l == 3.0
    np.testing.assert_allclose(result.total_concentrations, [bA, bB],
                               atol=1e-12)

    c = dict(zip(result.species, result.concentrations))
    # 独立复核（不经过库的模型矩阵）
    assert abs(c["A"] + c["AB"] + c["AB2"] - bA) < 1e-9
    assert abs(c["B"] + c["AB"] + 2 * c["AB2"] - bB) < 1e-9
    assert abs(c["AB"] / (c["A"] * c["B"]) - 10.0) < 1e-7
    assert abs(c["AB2"] / (c["AB"] * c["B"]) - 5.0) < 1e-7
    assert result.verification.passed


def test_naive_concentration_average_is_not_equilibrium_but_conserves(model):
    """用户强调的点：平均旧浓度守恒，却不满足混合后平衡。"""

    s1 = make_sample(equilibrate_totals(model, [1.0, 1.5], volume_l=2.0))
    s2 = make_sample(equilibrate_totals(model, [0.5, 3.0], volume_l=1.0))
    v1, v2 = 2.0, 1.0
    naive = (v1 * s1.concentrations + v2 * s2.concentrations) / (v1 + v2)

    b = model.formula_matrix @ naive
    bA, bB = (2.0 * 1.0 + 0.5) / 3.0, (2.0 * 1.5 + 3.0) / 3.0
    np.testing.assert_allclose(b, [bA, bB], atol=1e-12)  # 组分守恒

    report = verify_composition(model, naive, b)
    assert not report.passed                              # 但不是平衡
    assert report.max_equilibrium_residual > 1e-3

    # 库给出的结果必须与朴素平均不同
    rebalanced = mix_samples([s1, s2], volumes_l=[v1, v2]).concentrations
    assert np.max(np.abs(rebalanced - naive)) > 1e-4


def test_input_samples_are_not_mutated(model):
    s1 = make_sample(equilibrate_totals(model, [1.0, 1.5], volume_l=2.0), "s1")
    s2 = make_sample(equilibrate_totals(model, [0.5, 3.0], volume_l=1.0), "s2")
    c1_before = s1.concentrations.copy()
    v1_before = s1.volume_l
    mix_samples([s1, s2])
    np.testing.assert_array_equal(s1.concentrations, c1_before)
    assert s1.volume_l == v1_before
    try:
        s1.concentrations[0] = 999.0  # type: ignore[misc]
        raise AssertionError("样品浓度应当不可写")
    except ValueError:
        pass


def test_mix_provenance_and_overdraw(model):
    ws = Workspace()
    ws.add_model(model)
    s1 = ws.equilibrate(model, [1.0, 1.5], volume_l=0.5, sample_id="p1")
    s2 = ws.equilibrate(model, [1.0, 1.5], volume_l=0.5, sample_id="p2")

    try:
        ws.mix(["p1", "p2"], volumes_l=[0.6, 0.5])
        raise AssertionError("超过保存体积的取出必须被拒绝")
    except SampleOperationError:
        pass

    mixed = ws.mix(["p1", "p2"], sample_id="m")
    assert mixed.origin.operation == "mix"
    assert mixed.origin.parents == ("p1", "p2")
    tree = ws.lineage("m")
    assert [n["sample_id"] for n in tree["parents"]] == ["p1", "p2"]
    # 旧样品继续可查
    assert ws.get_sample("p1").concentrations is not None
    assert ws.has_sample("p1") and ws.has_sample("m")


def test_default_volumes_mix_whole_samples(model):
    s1 = make_sample(equilibrate_totals(model, [1.0, 1.5], volume_l=2.0))
    s2 = make_sample(equilibrate_totals(model, [1.0, 1.5], volume_l=4.0))
    result = mix_samples([s1, s2])
    assert result.volume_l == 6.0
    np.testing.assert_allclose(result.total_concentrations, [1.0, 1.5],
                               atol=1e-12)
