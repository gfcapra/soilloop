"""Stochastic version: random shocks on soil condition and community capacity."""
from __future__ import annotations
import numpy as np
from .model import Params, response, equilibria, stable, state_at


def _drift(S, H, C, P, p=None):
    p = P.p if p is None else p
    return (P.r * (1 - S) - p * (1 - C) * S, P.k * (S - H), P.b * response(H, P) - P.d * C)


def simulate_shocks(P: Params, S0, H0, C0, years=400, dt=0.1, sigma=0.01, seed=0,
                    noise="additive", record_every=None):
    """Euler-Maruyama integration of many replicates at once (arrays in, arrays out).
    noise='additive': Gaussian shocks on S and C, states kept in [0, 1] by clipping.
    noise='multiplicative': shocks scaled by S(1-S) and C(1-C), no clipping needed.
    If record_every (years) is given, also returns the list of recorded states."""
    rng = np.random.default_rng(seed)
    S, H, C = (np.array(v, dtype=float) for v in (S0, H0, C0))
    rec, every = [], (int(round(record_every / dt)) if record_every else None)
    for i in range(int(round(years / dt))):
        dS, dH, dC = _drift(S, H, C, P)
        eS = rng.standard_normal(S.shape); eC = rng.standard_normal(C.shape)
        if noise == "multiplicative":
            S = S + dS * dt + sigma * S * (1 - S) * np.sqrt(dt) * eS
            C = C + dC * dt + sigma * C * (1 - C) * np.sqrt(dt) * eC
            S, C = np.clip(S, 0, 1), np.clip(C, 0, 1)
        else:
            S = np.clip(S + dS * dt + sigma * np.sqrt(dt) * eS, 0, 1)
            C = np.clip(C + dC * dt + sigma * np.sqrt(dt) * eC, 0, 1)
        H = np.clip(H + dH * dt, 0, 1)
        if every and (i + 1) % every == 0:
            rec.append((S.copy(), H.copy(), C.copy()))
    return (S, H, C, rec) if record_every else (S, H, C)


def basin(P: Params, S, H, C, years=3000, dt=0.5, p=None):
    """Deterministic basin of attraction of each state: True = healthy attractor.
    States are continued without noise and assigned to the nearest stable equilibrium."""
    S, H, C = (np.array(v, dtype=float).copy() for v in (S, H, C))
    for _ in range(int(years / dt)):
        dS, dH, dC = _drift(S, H, C, P, p)
        S, H, C = S + dS * dt, H + dH * dt, C + dC * dt
    PP = P if p is None or np.ndim(p) == 0 else None
    if PP is not None:
        e = [s for s in equilibria(PP if p is None else P.with_(p=float(p))) if stable(s, P if p is None else P.with_(p=float(p)))]
        lo, hi = min(e), max(e)
        return np.abs(S - hi) < np.abs(S - lo)
    return S > 0.5


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n; den = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / den; w = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return (float(c - w), float(c + w))
