"""反应网络的重新表达：拆步、补等价反应不得改变真实平衡组成；
不一致的“等价”常数必须在建模阶段被拒绝。
"""

import numpy as np
import pytest

from chem_equilibrium_core import (ModelRejectionError, Reaction, build_model,
                                   equilibrate_totals, mix_samples,
                                   make_sample)
from tests.conftest import (R1_AB, R2_AB2, R_OVERALL, R_OVERALL_BAD,
                            REFERENCE_CONCENTRATIONS)


def _build(composition, reactions):
    return build_model(composition, reactions,
                       temperature=298.15, standard_concentration=1.0)


def test_same_system_id_across_representations(composition):
    m_step = _build(composition, [R1_AB(), R2_AB2()])
    m_overall = _build(composition, [R1_AB(), R_OVERALL()])
    m_all = _build(composition, [R1_AB(), R2_AB2(), R_OVERALL()])

    assert m_step.system_id == m_overall.system_id == m_all.system_id
    # 反应集合不同，network_id 不同
    assert m_step.network_id != m_all.network_id


def test_representations_give_identical_equilibrium(composition):
    models = [
        _build(composition, [R1_AB(), R2_AB2()]),
        _build(composition, [R1_AB(), R_OVERALL()]),
        _build(composition, [R_OVERALL(), R2_AB2()]),
        _build(composition, [R1_AB(), R2_AB2(), R_OVERALL()]),
    ]
    answers = []
    for m in models:
        res = equilibrate_totals(m, [1.0, 1.5])
        np.testing.assert_allclose(res.concentrations,
                                   REFERENCE_CONCENTRATIONS, atol=5e-7)
        answers.append(res.concentrations)
    for c in answers[1:]:
        np.testing.assert_allclose(c, answers[0], atol=1e-10)


def test_overall_splitting_matches_two_step(composition):
    """有人把整体反应拆成两步：拆开后的两步模型必须被接纳并与带等价反应的
    完整写法给出相同组成；而只留整体反应（丢掉中间步）是欠描述，应拒绝。"""

    m_two = _build(composition, [R1_AB(), R2_AB2()])
    assert m_two.n_independent_reactions == 2
    res_two = equilibrate_totals(m_two, [1.0, 1.5])

    m_full = _build(composition, [R1_AB(), R2_AB2(), R_OVERALL()])
    res_full = equilibrate_totals(m_full, [1.0, 1.5])
    np.testing.assert_allclose(res_two.concentrations,
                               res_full.concentrations, atol=1e-12)

    with pytest.raises(ModelRejectionError) as exc_info:
        _build(composition, [R_OVERALL()])
    assert any(r.code == "UNDERDESCRIBED_NETWORK"
               for r in exc_info.value.reasons)


def test_inconsistent_equivalent_reaction_rejected_before_solving(composition):
    bad_duplicate = Reaction("AB2-form-again",
                             {"AB2": 1, "A": -1, "B": -2}, k=30.0)
    with pytest.raises(ModelRejectionError) as exc_info:
        _build(composition, [R1_AB(), R2_AB2(), bad_duplicate])
    err = exc_info.value
    codes = [r.code for r in err.reasons]
    assert "INCONSISTENT_DUPLICATE_REACTION" in codes
    reason = next(r for r in err.reasons
                  if r.code == "INCONSISTENT_DUPLICATE_REACTION")
    # 必须关联到调用方给出的反应与具体常数
    assert "AB2-form-again" in reason.reactions
    assert reason.details["given_ln_k"] == pytest.approx(np.log(30))
    assert reason.details["implied_ln_k"] == pytest.approx(np.log(50))

    # 被拒绝后不能产出任何模型对象去做配液：用 R_OVERALL_BAD
    # （整体 K=30）补到两步模型里，它对两步反应是冗余且矛盾的
    with pytest.raises(ModelRejectionError):
        _build(composition, [R1_AB(), R2_AB2(), R_OVERALL_BAD()])

    # 反过来：整体反应 + 第二步反应也是一对独立反应，整体 K=30 与之自洽，
    # 此时描述的是另一个真实化学条件（隐含第一步 K=6），应当接纳
    m_alt = _build(composition, [R_OVERALL_BAD(), R2_AB2()])
    assert m_alt.n_independent_reactions == 2


def test_inconsistency_is_not_a_numerical_failure(composition):
    """化学条件冲突必须在数值迭代之前报告，而不是“不收敛”。"""

    with pytest.raises(ModelRejectionError) as exc_info:
        _build(composition, [
            Reaction("r1", {"AB": 1, "A": -1, "B": -1}, k=10),
            Reaction("r2", {"AB2": 1, "AB": -1, "B": -1}, k=5),
            Reaction("r3", {"AB2": 1, "A": -1, "B": -2}, k=500),
        ])
    msg = str(exc_info.value)
    assert "不" in msg and "ln K" in msg


def test_reaction_not_conserving_components_rejected_with_ids(composition):
    # 伪造的“反应”净生成一个 A：A + B -> 2AB
    bogus = Reaction("bogus", {"AB": 2, "A": -1, "B": -1}, k=2)
    with pytest.raises(ModelRejectionError) as exc_info:
        _build(composition, [R1_AB(), bogus])
    codes = {r.code: r for r in exc_info.value.reasons}
    assert "REACTION_NOT_CONSERVATIVE" in codes
    r = codes["REACTION_NOT_CONSERVATIVE"]
    assert r.reactions == ("bogus",)
    assert "A" in r.components


def test_missing_reaction_underdescribed_is_rejected(composition):
    """只给整体反应：存在未被描述的组成变化方向（AB 与 A/B 的互换）。"""

    with pytest.raises(ModelRejectionError) as exc_info:
        _build(composition, [R_OVERALL()])
    codes = [r.code for r in exc_info.value.reasons]
    assert "UNDERDESCRIBED_NETWORK" in codes
    reason = next(r for r in exc_info.value.reasons
                  if r.code == "UNDERDESCRIBED_NETWORK")
    assert "AB" in reason.species
    assert reason.details["expected_independent_reactions"] == 2


def test_unknown_species_and_zero_k(composition):
    with pytest.raises(ModelRejectionError) as exc_info:
        _build(composition, [
            R1_AB(),
            Reaction("rbad", {"C": 1, "A": -1}, k=3),
        ])
    assert any(r.code == "UNKNOWN_SPECIES_IN_REACTION"
               for r in exc_info.value.reasons)

    with pytest.raises(ValueError):
        Reaction("r", {"AB": 1, "A": -1, "B": -1}, k=0)


def test_redundant_checks_recorded(composition):
    m = _build(composition, [R1_AB(), R2_AB2(), R_OVERALL()])
    assert len(m.redundant_checks) == 1
    check = m.redundant_checks[0]
    assert check.reaction_id == R_OVERALL().id
    assert check.given_ln_k == pytest.approx(np.log(50))
    assert check.implied_ln_k == pytest.approx(np.log(10) + np.log(5))


def test_mix_samples_from_different_representations(composition):
    m1 = _build(composition, [R1_AB(), R2_AB2()])
    m2 = _build(composition, [R1_AB(), R_OVERALL()])  # 同体系，写法不同
    s1 = make_sample(equilibrate_totals(m1, [1.0, 1.5]))
    s2 = make_sample(equilibrate_totals(m2, [0.5, 3.0]))
    res = mix_samples([s1, s2], volumes_l=[1.0, 1.0])
    assert res.verification.passed
    assert res.origin.detail["solver_model_id"] == m1.model_id
