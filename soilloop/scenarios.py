"""Intervention scenarios: cleanup and land care."""
from __future__ import annotations
import numpy as np
from .model import Params, equilibria, state_at, simulate, attractor, stable


def _trap_and_threshold(P: Params):
    e = equilibria(P)
    if len(e) != 3:
        return None
    return state_at(e[0], P), e[1]


UNRESOLVED = []   # parameter sets whose outcome was not settled within the horizon
UNRESOLVED_AS = "trap"   # how unresolved runs are counted; "healthy" for the sensitivity check


def outcome(P: Params, cleanup: float, land_care: float = 0.0, land_care_years: float = 0.0,
            horizon: float = 3000, max_step: float = 1.0) -> str:
    """'healthy', 'trap' or 'unresolved' after a cleanup closing `cleanup` (0-1) of the
    soil gap, optionally with land care. The horizon is doubled once if unresolved."""
    tt = _trap_and_threshold(P)
    if tt is None:
        raise ValueError("Escape is defined only for bistable parameter sets.")
    trap, _ = tt
    y0 = [trap[0] + cleanup * (1 - trap[0]), trap[1], trap[2]]
    for T in (horizon, 2 * horizon):
        _, y = simulate(P, y0, years=T, land_care=land_care,
                        land_care_years=land_care_years, points=2, max_step=max_step)
        lab = attractor(y[:, -1], P)
        if lab != "unresolved":
            return lab
    UNRESOLVED.append((P, cleanup, land_care, land_care_years))
    return lab


def escapes(P: Params, cleanup: float, land_care: float = 0.0,
            land_care_years: float = 0.0, horizon: float = 3000, max_step: float = 1.0) -> bool:
    """True if the territory ends in the healthy attractor (see `outcome`)."""
    o = outcome(P, cleanup, land_care, land_care_years, horizon, max_step)
    return o == "healthy" or (o == "unresolved" and UNRESOLVED_AS == "healthy")


def minimum_cleanup(P: Params, land_care: float = 0.0, land_care_years: float = 0.0,
                    tol: float = 0.005, max_step: float = 1.0) -> float:
    """Smallest cleanup fraction allowing escape. Returns nan if none suffices
    (full cleanup still relapses) and 0 if no cleanup is needed."""
    if escapes(P, 0.0, land_care, land_care_years, max_step=max_step):
        return 0.0
    if not escapes(P, 1.0, land_care, land_care_years, max_step=max_step):
        return float("nan")
    a, b = 0.0, 1.0
    while b - a > tol:
        m = (a + b) / 2
        a, b = (a, m) if escapes(P, m, land_care, land_care_years, max_step=max_step) else (m, b)
    return b


def compare_strategies(P: Params, cleanup: float, land_care: float, years: float):
    """Outcome of cleanup alone, land care alone and both together."""
    out = {}
    for name, (x, u, t) in {"cleanup only": (cleanup, 0, 0),
                            "land care only": (0, land_care, years),
                            "both": (cleanup, land_care, years)}.items():
        out[name] = escapes(P, x, u, t)
    return out


def sample_parameters(ranges: dict, n: int, seed: int = 0, log_rates=("r", "p", "k", "d"), fixed=None):
    """Latin hypercube sample of parameter sets (rates log-uniform)."""
    from scipy.stats import qmc
    keys = list(ranges)
    U = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    out = []
    for u in U:
        kw = {}
        for i, key in enumerate(keys):
            lo, hi = ranges[key]
            kw[key] = 10 ** (np.log10(lo) + u[i] * (np.log10(hi) - np.log10(lo))) if key in log_rates \
                else lo + u[i] * (hi - lo)
        kw.update(fixed or {})
        out.append(Params(**kw))
    return out


def minimum_land_care(P: Params, cleanup: float, years: float, tol: float = 0.005,
                      max_step: float = 1.0) -> float:
    """Smallest fractional pressure reduction, held for `years`, that allows escape
    together with `cleanup`. nan if even a complete reduction is not enough."""
    if escapes(P, cleanup, 0.0, 0.0, max_step=max_step):
        return 0.0
    if not escapes(P, cleanup, 1.0, years, max_step=max_step):
        return float("nan")
    a, b = 0.0, 1.0
    while b - a > tol:
        m = (a + b) / 2
        a, b = (a, m) if escapes(P, cleanup, m, years, max_step=max_step) else (m, b)
    return b
