"""Core dynamical model of the soil-health-community loop.

State variables, all scaled 0-1:
    S  soil condition (inverse of the contaminant stock in exposed layers)
    H  population health and human capital
    C  community capacity (cohesion, investment, care of land)

    dS/dt = r (1 - S) - p (1 - C) S
    dH/dt = k (S - H)
    dC/dt = b H^n / (h^n + H^n) - d C
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class Params:
    r: float = 0.02   # natural soil recovery rate (1/yr) = 1 / contaminant residence time
    p: float = 0.06   # degradation pressure on uncared land (1/yr)
    k: float = 0.5    # health response rate (1/yr)
    d: float = 0.1    # community decay rate (1/yr); build-up rate b is set equal to d
    n: float = 4.0    # steepness of the community threshold
    h: float = 0.4    # health level at the community threshold

    @property
    def b(self) -> float:
        return self.d

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


# Classes of contaminant residence time in the exposed soil layers (see paper, Table 1).
# Residence time depends on the element AND on the soil (pH, organic matter, sorption,
# drainage): e.g. Pb in organic horizons 60-250+ yr; Cd about 170 yr in average
# EU cropland topsoils (leaching only) but decades in strongly leached soils.
PRESETS = {
    "long residence (>250 yr)": dict(r=0.003),
    "intermediate (70-250 yr)": dict(r=0.008),
    "short residence (<70 yr)": dict(r=0.018),
}
# Literature-anchored ranges used in the paper
ANCHORED_RANGES = dict(r=(0.001, 0.023), p=(0.005, 0.15), k=(0.04, 0.1),
                       d=(0.02, 0.1), n=(2.0, 8.0), h=(0.3, 0.6))


def hill(H, n, h):
    H = np.clip(H, 0.0, 1.0)
    return H**n / (h**n + H**n)


def rhs(t, y, P: Params, pressure_factor=None):
    S, H, C = y
    p = P.p * (pressure_factor(t) if pressure_factor else 1.0)
    return [P.r * (1 - S) - p * (1 - C) * S,
            P.k * (S - H),
            P.b * hill(H, P.n, P.h) - P.d * C]


def equilibria(P: Params, grid: int = 4001) -> list[float]:
    """Soil condition at the equilibria (H* = S*, C* = hill(S*))."""
    f = lambda S: P.r * (1 - S) - P.p * (1 - hill(S, P.n, P.h)) * S
    g = np.linspace(1e-6, 1.0, grid)
    v = f(g)
    return [brentq(f, g[i], g[i + 1]) for i in range(grid - 1) if v[i] * v[i + 1] < 0]


def regime(P: Params) -> str:
    """'bistable', 'trap only' or 'healthy only'."""
    e = equilibria(P)
    if len(e) == 3:
        return "bistable"
    return "trap only" if e and e[0] < 0.5 else "healthy only"


def state_at(S: float, P: Params) -> list[float]:
    return [S, S, hill(S, P.n, P.h)]


def simulate(P: Params, y0, years: float = 300, land_care: float = 0.0,
             land_care_years: float = 0.0, points: int = 600, max_step: float = 1.0):
    """Integrate the model. land_care is the fractional reduction of pressure
    maintained for land_care_years from t = 0."""
    pf = (lambda t: (1 - land_care) if t < land_care_years else 1.0) if land_care else None
    sol = solve_ivp(rhs, (0, years), y0, args=(P, pf), method="LSODA", max_step=max_step,
                    t_eval=np.linspace(0, years, points))
    return sol.t, sol.y
