"""Core dynamical model of the soil-health-community loop.

State variables, all scaled 0-1:
    S  soil condition (inverse of the contaminant stock in exposed layers)
    H  population health and human capital
    C  community capacity (cohesion, investment, care of land)

    dS/dt = r (1 - S) - p (1 - C) S
    dH/dt = k (S - H)
    dC/dt = b f(H) - d C,   f(H) = c0 + (1 - c0) H^n / (h^n + H^n)

S is the complement of a normalized contaminant load Q = 1 - S in the soil
layers people are exposed to (Q = 1 is the maximum load the territory can
accumulate). In terms of Q the soil equation reads dQ/dt = p (1 - C)(1 - Q) - r Q:
a saturating input from neglected land and a first-order loss from the exposed
compartment (leaching, burial, dilution), with r = 1 / residence time.
c0 is a baseline community capacity that does not depend on health
(institutions, external investment); the paper's default is c0 = 0.
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
    k: float = 0.07   # health response rate (1/yr); half-life ln(2)/k, about 10 yr
    d: float = 0.1    # community decay rate (1/yr); build-up rate b is set equal to d
    n: float = 4.0    # steepness of the community threshold
    h: float = 0.4    # health level at the community threshold
    c0: float = 0.0   # baseline community capacity independent of health

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


def response(H, P: "Params"):
    """Community capacity sustained by health level H at equilibrium."""
    return P.c0 + (1 - P.c0) * hill(H, P.n, P.h)


def rhs(t, y, P: Params, pressure_factor=None):
    S, H, C = y
    p = P.p * (pressure_factor(t) if pressure_factor else 1.0)
    return [P.r * (1 - S) - p * (1 - C) * S,
            P.k * (S - H),
            P.b * response(H, P) - P.d * C]


def equilibria(P: Params, grid: int = 4001) -> list[float]:
    """Soil condition at the equilibria (H* = S*, C* = hill(S*))."""
    f = lambda S: P.r * (1 - S) - P.p * (1 - response(S, P)) * S
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
    return [S, S, float(response(S, P))]


def jacobian(y, P: Params, p=None):
    S, H, C = y
    p = P.p if p is None else p
    dhill = (1 - P.c0) * P.n * P.h**P.n * H**(P.n - 1) / (P.h**P.n + H**P.n) ** 2 if H > 0 else 0.0
    return np.array([[-P.r - p * (1 - C), 0.0, p * S],
                     [P.k, -P.k, 0.0],
                     [0.0, P.b * dhill, -P.d]])


def stable(S_eq: float, P: Params) -> bool:
    """Linear stability of the equilibrium with soil condition S_eq."""
    return bool(np.max(np.linalg.eigvals(jacobian(state_at(S_eq, P), P)).real) < 0)


def attractor(y, P: Params, tol: float = 0.05, vtol: float = 1e-3):
    """Classify a state as 'healthy', 'trap' or 'unresolved': it must lie within `tol`
    of a stable equilibrium (Euclidean distance in S, H, C) and the vector field
    there must be small (norm below `vtol`), so that slow transients near a fold or
    near the separatrix are not mistaken for a settled outcome."""
    e = [s for s in equilibria(P) if stable(s, P)]
    if not e:
        return "unresolved"
    lo, hi = min(e), max(e)
    v = np.linalg.norm(rhs(0, y, P))
    dist = {lab: np.linalg.norm(np.array(y) - np.array(state_at(s, P))) for lab, s in (("trap", lo), ("healthy", hi))}
    lab = min(dist, key=dist.get)
    if len(e) == 1:
        lab = "trap" if lo < 0.5 else "healthy"
    return lab if dist[min(dist, key=dist.get)] < tol and v < vtol else "unresolved"


def simulate(P: Params, y0, years: float = 300, land_care: float = 0.0,
             land_care_years: float = 0.0, points: int = 600, max_step: float = 1.0):
    """Integrate the model. land_care is the fractional reduction of pressure
    maintained for land_care_years from t = 0."""
    t_eval = np.linspace(0, years, points)
    if not land_care or land_care_years <= 0:
        sol = solve_ivp(rhs, (0, years), y0, args=(P,), method="LSODA", max_step=max_step,
                        t_eval=t_eval, rtol=1e-7, atol=1e-9)
        return sol.t, sol.y
    # integrate in two pieces so that the end of land care is not smoothed over
    t1 = min(land_care_years, years)
    P1 = P.with_(p=P.p * (1 - land_care))
    a = solve_ivp(rhs, (0, t1), y0, args=(P1,), method="LSODA", max_step=max_step,
                  t_eval=t_eval[t_eval <= t1], rtol=1e-7, atol=1e-9)
    if t1 >= years:
        return a.t, a.y
    y1 = solve_ivp(rhs, (0, t1), y0, args=(P1,), method="LSODA", max_step=max_step,
                   rtol=1e-7, atol=1e-9).y[:, -1]
    b = solve_ivp(rhs, (t1, years), y1, args=(P,), method="LSODA", max_step=max_step,
                  t_eval=t_eval[t_eval > t1], rtol=1e-7, atol=1e-9)
    return np.concatenate([a.t, b.t]), np.concatenate([a.y, b.y], axis=1)
