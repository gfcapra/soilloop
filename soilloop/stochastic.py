"""Stochastic version: random shocks on soil condition and community capacity."""
from __future__ import annotations
import numpy as np
from .model import Params, hill


def simulate_shocks(P: Params, S0, H0, C0, years=400, dt=0.1, sigma=0.01, seed=0):
    """Euler-Maruyama integration of many replicates at once (arrays in, arrays out)."""
    rng = np.random.default_rng(seed)
    S, H, C = (np.array(v, dtype=float) for v in (S0, H0, C0))
    for _ in range(int(years / dt)):
        dS = P.r * (1 - S) - P.p * (1 - C) * S
        dH = P.k * (S - H)
        dC = P.b * hill(H, P.n, P.h) - P.d * C
        S = np.clip(S + dS * dt + sigma * np.sqrt(dt) * rng.standard_normal(S.shape), 0, 1)
        C = np.clip(C + dC * dt + sigma * np.sqrt(dt) * rng.standard_normal(C.shape), 0, 1)
        H = np.clip(H + dH * dt, 0, 1)
    return S, H, C
