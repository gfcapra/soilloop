"""Section 5: false-positive rate and power of the bimodality screen.

Null datasets have no feedback; loop datasets come from the stochastic model.
Each setting is replicated 200 times with 300 units unless stated."""
import json, numpy as np, pandas as pd
from scipy.stats import norm, beta, multivariate_t
from soilloop import Params, simulate_shocks
from soilloop.diagnostics import bimodality_test

REPS = 200
def corr_normals(rng, n, rho=0.7):
    Sig = np.full((3, 3), rho); np.fill_diagonal(Sig, 1)
    return rng.multivariate_normal(np.zeros(3), Sig, n)

def gen(name, rng, n=300):
    if name == "unimodal skewed (beta copula)":
        Z = corr_normals(rng, n); X = beta(5, 2).ppf(norm.cdf(Z))
    elif name == "heavy-tailed unimodal (t, df=3)":
        Sig = np.full((3, 3), 0.7); np.fill_diagonal(Sig, 1)
        X = 0.6 + 0.08 * multivariate_t(np.zeros(3), Sig, df=3).rvs(n, random_state=rng)
    elif name == "continuous spatial gradient":
        g = rng.random(n); X = 0.2 + 0.6 * g[:, None] + rng.normal(0, 0.08, (n, 3))
    elif name == "common driver, moderate effect":
        z = rng.standard_normal(n); X = 1 / (1 + np.exp(-1.5 * z))[:, None] + rng.normal(0, 0.08, (n, 3))
    elif name == "common driver, steep effect":
        z = rng.standard_normal(n); X = 1 / (1 + np.exp(-4 * z))[:, None] + rng.normal(0, 0.08, (n, 3))
    elif name.startswith("loop model"):
        P = Params(p=0.07)
        S, H, C = simulate_shocks(P, rng.random(n), rng.random(n), rng.random(n), years=600, sigma=0.01, seed=int(rng.integers(1e9)))
        X = np.column_stack([S, H, C]) + rng.normal(0, 0.08, (n, 3))
    return pd.DataFrame(np.clip(X, 0, 1), columns=["S", "H", "C"])

settings = [("unimodal skewed (beta copula)", 300), ("heavy-tailed unimodal (t, df=3)", 300),
            ("continuous spatial gradient", 300), ("common driver, moderate effect", 300),
            ("common driver, steep effect", 300), ("loop model", 50), ("loop model", 100), ("loop model", 300)]
out = {}
for si, (name, n) in enumerate(settings):
    rng = np.random.default_rng(1000 + si)
    bic_only = bic_d = full = 0
    for _ in range(REPS):
        t = bimodality_test(gen(name, rng, n), n_init=3)
        bic_only += t["delta_bic"] > 10; bic_d += (t["delta_bic"] > 10 and t["ashman_d"] > 2); full += t["two_groups_preferred"]
    out[f"{name}, n={n}"] = dict(bic_only=bic_only / REPS, bic_and_D=bic_d / REPS, bic_D_dip=full / REPS)
    print(name, n, out[f"{name}, n={n}"], flush=True)
json.dump(out, open("part_d.json", "w"), indent=1)
