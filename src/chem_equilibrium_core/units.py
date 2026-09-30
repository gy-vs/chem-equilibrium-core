"""轻量单位处理。

范围刻意保持很小：本项目只处理固定温度下单液相的体积摩尔浓度，因此只接受
温度（K / °C）、浓度（mol/L 等）和体积（L / mL）的常见写法。不引入额外依赖，
也不做物理量的泛型量纲系统。
"""

from __future__ import annotations

from dataclasses import dataclass

# 各浓度单位到 mol/L 的换算
_CONCENTRATION_TO_MOL_PER_L = {
    "mol/L": 1.0,
    "mol/l": 1.0,
    "M": 1.0,
    "mmol/L": 1e-3,
    "mmol/l": 1e-3,
    "mM": 1e-3,
    "umol/L": 1e-6,
    "uM": 1e-6,
}

# 各体积单位到 L 的换算
_VOLUME_TO_L = {
    "L": 1.0,
    "l": 1.0,
    "mL": 1e-3,
    "ml": 1e-3,
    "uL": 1e-6,
    "ul": 1e-6,
}


@dataclass(frozen=True)
class Quantity:
    """带单位的标量。内部一律换算为 SI 风格的基准单位保存。"""

    value: float
    unit: str


def temperature_to_kelvin(value: float, unit: str = "K") -> float:
    """把温度换算为开尔文。"""

    if unit == "K":
        return float(value)
    if unit in ("degC", "°C", "C"):
        return float(value) + 273.15
    raise ValueError(f"不支持的温度单位 {unit!r}，仅接受 K、°C")


def concentration_to_mol_per_l(value: float, unit: str = "mol/L") -> float:
    """把浓度换算为 mol/L。"""

    if unit not in _CONCENTRATION_TO_MOL_PER_L:
        raise ValueError(
            f"不支持的浓度单位 {unit!r}，支持：{sorted(set(_CONCENTRATION_TO_MOL_PER_L))}"
        )
    return float(value) * _CONCENTRATION_TO_MOL_PER_L[unit]


def volume_to_liters(value: float, unit: str = "L") -> float:
    """把体积换算为升。"""

    if unit not in _VOLUME_TO_L:
        raise ValueError(
            f"不支持的体积单位 {unit!r}，支持：{sorted(set(_VOLUME_TO_L))}"
        )
    return float(value) * _VOLUME_TO_L[unit]
