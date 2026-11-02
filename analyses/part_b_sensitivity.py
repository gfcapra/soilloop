"""Section 4.3 and Fig. S1: parameter sweeps (anchored and wide ranges)."""
import json, sys, numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from soilloop import ANCHORED_RANGES, sample_parameters, regime, minimum_cleanup, minimum_land_care

WIDE = dict(r=(0.005, 0.05), p=(0.02, 0.15), k=(0.2, 2.0), d=(0.05, 0.5), n=(2, 8), h=(0.3, 0.6))

def anchored_job(P):
    out = dict(tS=minimum_cleanup(P, max_step=5))
    for tau in (10, 20, 40):
        out[f"u{tau}"] = minimum_land_care(P, 0.0, tau, max_step=5)
    for tau in (20, 40, 80):
        out[f"x_lc{tau}"] = minimum_cleanup(P, land_care=0.5, land_care_years=tau, max_step=5)
    return out

def wide_job(P):
    return minimum_cleanup(P, max_step=2)

def summary(v):
    v = np.array(v, float)
    return dict(median=float(np.nanmedian(v)), q25=float(np.nanpercentile(v, 25)), q75=float(np.nanpercentile(v, 75)), none=float(np.isnan(v).mean()))

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "anchored":
        S = sample_parameters(ANCHORED_RANGES, 3000, seed=2)
        reg = np.array([regime(P) for P in S]); r = np.array([P.r for P in S])
        res = dict(all={k: float(np.mean(reg == k)) for k in ("healthy only", "bistable", "trap only")})
        for lab, m in [("long", r < 1/250), ("intermediate", (r >= 1/250) & (r < 1/70)), ("short", r >= 1/70)]:
            res[lab] = {k: float(np.mean(reg[m] == k)) for k in ("healthy only", "bistable", "trap only")}
        idx = np.where(reg == "bistable")[0]; idx = np.random.default_rng(0).choice(idx, 250, replace=False)
        with Pool() as pool: R = pool.map(anchored_job, [S[i] for i in idx], chunksize=4)
        res["cleanup"] = summary([o["tS"] for o in R])
        for tau in (10, 20, 40):
            u = np.array([o[f"u{tau}"] for o in R], float); res[f"land_care_alone_{tau}"] = float(np.mean(~np.isnan(u)))
        for tau in (20, 40, 80):
            res[f"cleanup_with_lc{tau}"] = summary([o[f"x_lc{tau}"] for o in R])
        json.dump(res, open("part_b_anchored.json", "w"), indent=1)
    else:
        S = sample_parameters(WIDE, 4000, seed=1, log_rates=())
        reg = np.array([regime(P) for P in S])
        res = dict(all={k: float(np.mean(reg == k)) for k in ("healthy only", "bistable", "trap only")})
        X = np.array([[P.r, P.p, P.k, P.d, P.n, P.h] for P in S]); B = reg == "bistable"
        res["spearman_bistable"] = {n: float(spearmanr(X[:, i], B).correlation) for i, n in enumerate("r p k d n h".split())}
        r = X[:, 0]; bins = np.linspace(0.005, 0.05, 7)
        res["by_r"] = {k: [float(np.mean(reg[(r >= a) & (r < b)] == k)) for a, b in zip(bins[:-1], bins[1:])] for k in ("healthy only", "bistable", "trap only")}
        with Pool() as pool: T = pool.map(wide_job, [S[i] for i in np.where(B)[0]], chunksize=8)
        res["cleanup"] = summary(T); res["cleanup_values"] = [float(t) for t in T]
        json.dump(res, open("part_b_wide.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cleanup_values"}, indent=1))
