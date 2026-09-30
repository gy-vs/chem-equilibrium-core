"""结果确认与失败隔离：公开复核能否决裁剪解；一次失败不污染台账。"""

import numpy as np
import pytest

from chem_equilibrium_core import (EquilibriumSolveError, SampleOperationError,
                                   VerificationError, Workspace,
                                   equilibrate_totals, make_sample,
                                   verify_composition)
from chem_equilibrium_core import sample as sample_module


def test_verification_rejects_negative_composition(model):
    b = np.array([1.0, 1.5])
    bad = np.array([0.17, 0.22, 0.39, -0.001])
    report = verify_composition(model, bad, b)
    assert not report.passed
    assert report.concentration_residual < 0
    with pytest.raises(VerificationError) as exc_info:
        report.raise_if_failed()
    assert exc_info.value.report is report
    assert "负浓度" in str(exc_info.value)


def test_verification_rejects_imbalanced_composition(model):
    """浓度都为正、也满足反应商，但组分账对不上——必须被否决。"""

    res = equilibrate_totals(model, [1.0, 1.5])
    wrong_b = np.array([1.1, 1.5])  # 声称投入更多 A
    report = verify_composition(model, res.concentrations, wrong_b,
                                balance_tol=1e-10)
    assert not report.passed
    assert any(r.component == "A" and not r.relative_residual == 0
               for r in report.balance_records)


def test_verification_rejects_wrong_equilibrium(model):
    """满足收支但平衡常数不满足的候选（例如朴素平均）也必须被否决。"""

    s1 = make_sample(equilibrate_totals(model, [1.0, 1.5]))
    s2 = make_sample(equilibrate_totals(model, [0.5, 3.0]))
    naive = (s1.concentrations + s2.concentrations) / 2
    b = model.formula_matrix @ naive
    report = verify_composition(model, naive, b, equilibrium_tol=1e-6)
    assert not report.passed
    assert report.max_equilibrium_residual > 1e-3
    with pytest.raises(VerificationError):
        report.raise_if_failed()


def test_failed_calculation_does_not_pollute_workspace(model, monkeypatch):
    ws = Workspace()
    ws.add_model(model)
    good = ws.equilibrate(model, [1.0, 1.5], sample_id="good")

    def broken_solver(*args, **kwargs):
        raise EquilibriumSolveError("模拟内核失败", residual=None)

    monkeypatch.setattr(sample_module, "solve_candidate", broken_solver)
    with pytest.raises(EquilibriumSolveError):
        ws.equilibrate(model, [0.8, 2.0], sample_id="would-be")
    monkeypatch.undo()

    # 失败 id 未入册；旧样品原样可查；修复后可继续算
    assert not ws.has_sample("would-be")
    assert ws.get_sample("good") is good
    again = ws.equilibrate(model, [0.8, 2.0], sample_id="again")
    assert again.verification.passed


def test_zero_component_total_rejected(model):
    with pytest.raises(SampleOperationError):
        equilibrate_totals(model, [0.0, 1.5])


def test_strongly_binding_case_positive_without_clipping():
    """K 很大时自由物种很小，但仍必须严格为正，不能出现裁剪出来的零。"""

    from chem_equilibrium_core import Reaction, build_model

    m = build_model(
        {"A": {"A": 1}, "B": {"B": 1}, "AB": {"A": 1, "B": 1}},
        [Reaction("tight", {"AB": 1, "A": -1, "B": -1}, k=1e8)])
    res = equilibrate_totals(m, [1.0, 1.0], equilibrium_tol=1e-8)
    assert np.all(res.concentrations > 0)
    assert res.verification.min_concentration > 0
    assert res.verification.passed
    # 自由 A、B 约 sqrt(1/K) = 1e-4 mol/L，结合物接近 1 mol/L
    np.testing.assert_allclose(res.concentrations[2], 1.0, atol=2e-4)
