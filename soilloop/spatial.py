"""Spatial version: neighborhoods on a lattice, heterogeneous pressure, social coupling."""
from __future__ import annotations
import numpy as np
from .model import Params, response, equilibria, stable


def laplacian(C, L):
    G = C.reshape(L, L)
    return (np.roll(G, 1, 0) + np.roll(G, -1, 0) + np.roll(G, 1, 1) + np.roll(G, -1, 1) - 4 * G).ravel()


def step(S, H, C, P: Params, p, dt, sigma, rng, D=0.0, L=None):
    dS = P.r * (1 - S) - p * (1 - C) * S
    dH = P.k * (S - H)
    dC = P.b * response(H, P) - P.d * C
    if D and L:
        dC = dC + D * laplacian(C, L)
    S = np.clip(S + dS * dt + sigma * np.sqrt(dt) * rng.standard_normal(S.shape), 0, 1)
    C = np.clip(C + dC * dt + sigma * np.sqrt(dt) * rng.standard_normal(C.shape), 0, 1)
    H = np.clip(H + dH * dt, 0, 1)
    return S, H, C


def pressure_field(p, cv, z):
    return p * np.clip(1 + cv * z, 0.05, None)


def hysteresis_sweep(P: Params, pressures, L=12, cv=0.0, D=0.0, sigma=0.0, years_per_step=200,
                     dt=0.1, seed=3):
    """Raise pressure step by step from a healthy city, then lower it back.
    Returns mean soil condition on the rising and falling branches."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(L * L)
    S = np.full(L * L, 0.95); H = S.copy(); C = np.full(L * L, 0.95)
    up, down = [], []
    for branch, seq in ((up, pressures), (down, pressures[::-1])):
        for p in seq:
            pv = pressure_field(p, cv, z)
            for _ in range(int(years_per_step / dt)):
                S, H, C = step(S, H, C, P, pv, dt, sigma, rng, D, L)
            branch.append(S.mean())
    return np.array(up), np.array(down[::-1])


def trapped_share(P: Params, p, L=12, cv=0.3, D=0.0, sigma=0.01, years=600, dt=0.1, reps=6, seed=5):
    """Share of neighborhoods in the trap (S < 0.45) after `years`, from random starts."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        z = rng.standard_normal(L * L)
        pv = pressure_field(p, cv, z)
        S = rng.random(L * L); H = S.copy(); C = rng.random(L * L)
        for _ in range(int(years / dt)):
            S, H, C = step(S, H, C, P, pv, dt, sigma, rng, D, L)
        out.append(np.mean(S < 0.45))
    return float(np.mean(out))


def cell_states(P: Params, pv, S):
    """Classify each neighborhood as trapped (True) or healthy (False) by the nearest
    stable equilibrium of its own local pressure."""
    out = np.zeros(len(S), bool)
    for i, p in enumerate(pv):
        e = [s for s in equilibria(P.with_(p=float(p))) if stable(s, P.with_(p=float(p)))]
        lo, hi = min(e), max(e)
        out[i] = abs(S[i] - lo) < abs(S[i] - hi) if len(e) > 1 else lo < 0.5
    return out


def morans_i(x, L):
    """Moran's I with rook neighbors on a periodic L x L lattice."""
    G = x.reshape(L, L) - x.mean()
    num = sum((G * np.roll(G, s, a)).sum() for s, a in ((1, 0), (-1, 0), (1, 1), (-1, 1)))
    return float((len(x) / (4 * len(x))) * num / (G**2).sum()) if (G**2).sum() > 0 else float("nan")


def spatial_metrics(P: Params, p, L=12, cv=0.3, D=0.0, sigma=0.01, years=600, dt=0.1, reps=6, seed=5,
                    start="random"):
    """Trapped share, between-neighborhood dispersion of S, concordance (share of
    neighborhoods in the majority state) and Moran's I, averaged over replicates.
    start: 'random' (uniform 0-1) or 'middle' (all at 0.5 with small noise)."""
    rng = np.random.default_rng(seed)
    acc = dict(trapped=[], sd=[], concordance=[], moran=[])
    for _ in range(reps):
        z = rng.standard_normal(L * L)
        pv = pressure_field(p, cv, z)
        if start == "random":
            S = rng.random(L * L); C = rng.random(L * L)
        else:
            S = np.clip(0.5 + 0.05 * rng.standard_normal(L * L), 0, 1); C = S.copy()
        H = S.copy()
        for _ in range(int(years / dt)):
            S, H, C = step(S, H, C, P, pv, dt, sigma, rng, D, L)
        tr = cell_states(P, pv, S)
        acc["trapped"].append(tr.mean()); acc["sd"].append(S.std())
        acc["concordance"].append(max(tr.mean(), 1 - tr.mean())); acc["moran"].append(morans_i(S, L))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}
