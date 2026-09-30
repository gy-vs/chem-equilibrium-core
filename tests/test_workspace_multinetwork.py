"""同一进程内多网络并存、并发计算，以及温度/标准浓度等体系参数。"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from chem_equilibrium_core import (Reaction, SampleOperationError, Workspace,
                                   build_model, equilibrate_totals, make_sample,
                                   mix_samples)


def _ab_network(k=10.0, temperature=298.15, c0=1.0):
    return build_model(
        {"A": {"A": 1}, "B": {"B": 1}, "AB": {"A": 1, "B": 1}},
        [Reaction("A+B=AB", {"AB": 1, "A": -1, "B": -1}, k=k)],
        temperature=temperature, standard_concentration=c0)


def test_two_networks_coexist_and_disagree_as_expected():
    m10 = _ab_network(k=10.0)
    m100 = _ab_network(k=100.0)
    assert m10.system_id == m100.system_id  # 同体系不同常数
    assert m10.network_id != m100.network_id
    r10 = equilibrate_totals(m10, [1.0, 1.0])
    r100 = equilibrate_totals(m100, [1.0, 1.0])
    # 结合更强时自由 A 更低
    assert r100.concentrations[0] < r10.concentrations[0]
    assert r10.verification.passed and r100.verification.passed


def test_mixing_samples_across_different_systems_rejected():
    ab = _ab_network()
    other = build_model(
        {"X": {"X": 1}, "Y": {"Y": 1}, "XY": {"X": 1, "Y": 1}},
        [Reaction("X+Y=XY", {"XY": 1, "X": -1, "Y": -1}, k=2)])
    s1 = make_sample(equilibrate_totals(ab, [1.0, 1.0]))
    s2 = make_sample(equilibrate_totals(other, [1.0, 1.0]))
    with pytest.raises(SampleOperationError) as exc_info:
        mix_samples([s1, s2])
    assert "化学体系" in str(exc_info.value)


def test_concurrent_equilibration_is_isolated():
    ws = Workspace()
    models = [_ab_network(k=k) for k in (1.0, 10.0, 100.0, 1000.0)]
    for m in models:
        ws.add_model(m)

    def job(i):
        m = models[i % len(models)]
        bA, bB = 0.5 + 0.1 * i, 1.0 + 0.2 * i
        return ws.equilibrate(m, [bA, bB], volume_l=1.0,
                              sample_id=f"s{i}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        samples = list(ex.map(job, range(40)))

    assert len({s.sample_id for s in samples}) == 40
    assert len(ws.samples()) == 40
    # 每个样品按其网络常数达标
    for s in samples:
        c = s.concentrations
        k = {r.ln_k for r in s.model.reactions}
        rep = s.verification
        assert rep.passed
        # 逐样品独立核对
        assert abs(np.log(c[2] / (c[0] * c[1])) - next(iter(k))) < 1e-8


def test_standard_concentration_changes_dimensionful_answer():
    """c° 改变（同时给出对应条件下的无量纲 K）应改变数值答案并被携带。"""

    m1 = _ab_network(k=10.0, c0=1.0)
    m2 = _ab_network(k=10.0, c0=0.1)  # 例如 c°=0.1 mol/L
    assert m1.system_id != m2.system_id
    r1 = equilibrate_totals(m1, [1.0, 1.0])
    r2 = equilibrate_totals(m2, [1.0, 1.0])
    assert r1.standard_concentration == 1.0
    assert r2.standard_concentration == 0.1
    assert abs(r1.concentrations[0] - r2.concentrations[0]) > 1e-4
    assert r1.verification.passed and r2.verification.passed


def test_temperature_celsius_is_recorded_in_kelvin():
    m = _ab_network(temperature=25.0 + 273.15)
    m_c = build_model(
        {"A": {"A": 1}, "B": {"B": 1}, "AB": {"A": 1, "B": 1}},
        [Reaction("r", {"AB": 1, "A": -1, "B": -1}, k=10)],
        temperature=25.0, temperature_unit="°C")
    assert m.system_id == m_c.system_id


def test_three_component_network_generic():
    """非四物种专用：三组分、三元络合物 ABC 的通用网络。"""

    comp = {
        "A": {"A": 1},
        "B": {"B": 1},
        "C": {"C": 1},
        "AB": {"A": 1, "B": 1},
        "ABC": {"A": 1, "B": 1, "C": 1},
    }
    m = build_model(comp, [
        Reaction("AB", {"AB": 1, "A": -1, "B": -1}, k=8),
        Reaction("ABC", {"ABC": 1, "AB": -1, "C": -1}, k=4),
    ])
    res = equilibrate_totals(m, {"A": 1.0, "B": 1.0, "C": 0.6})
    assert res.verification.passed
    c = res.concentrations
    np.testing.assert_allclose(m.formula_matrix @ c, [1.0, 1.0, 0.6],
                               atol=1e-9)
