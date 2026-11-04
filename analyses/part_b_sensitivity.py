"""Section 4.3 and Fig. S1: parameter sweeps (anchored and wide ranges)."""
import json, sys, numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from soilloop import ANCHORED_RANGES, sample_parameters, regime, minimum_cleanup, minimum_land_care, equilibria
from soilloop import scenarios

WIDE = dict(r=(0.005, 0.05), p=(0.02, 0.15), k=(0.2, 2.0), d=(0.05, 0.5), n=(2, 8), h=(0.3, 0.6))

def anchored_job(P, as_=None):
    if as_: scenarios.UNRESOLVED_AS = as_
    scenarios.UNRESOLVED.clear()
    out = dict(tS=minimum_cleanup(P, max_step=5))
    for tau in (10, 20, 40):
        out[f"u{tau}"] = minimum_land_care(P, 0.0, tau, max_step=5)
    for tau in (20, 40, 80):
        out[f"x_lc{tau}"] = minimum_cleanup(P, land_care=0.5, land_care_years=tau, max_step=5)
    out["unresolved"] = len(scenarios.UNRESOLVED)
    return out

def wide_job(P):
    scenarios.UNRESOLVED.clear()
    return minimum_cleanup(P, max_step=2), len(scenarios.UNRESOLVED)

def summary(v):
    """Denominators are explicit: n_sets evaluated, n_feasible with a finite cleanup;
    the median and quartiles are conditional on feasibility."""
    v = np.array(v, float); ok = ~np.isnan(v)
    return dict(n_sets=int(v.size), n_feasible=int(ok.sum()), n_none=int((~ok).sum()), none=float((~ok).mean()),
                median_if_feasible=float(np.median(v[ok])) if ok.any() else None,
                q25_if_feasible=float(np.percentile(v[ok], 25)) if ok.any() else None,
                q75_if_feasible=float(np.percentile(v[ok], 75)) if ok.any() else None)

def regimes(S):
    reg = np.array([regime(P) for P in S])
    return reg, {k: float(np.mean(reg == k)) for k in ("healthy only", "bistable", "trap only")}

def importance(S, reg):
    X = np.array([[P.r, P.p, P.k, P.d, P.n, P.h, P.p / P.r] for P in S]); names = "r p k d n h p/r".split()
    trap_present = np.array([len(equilibria(P)) == 3 or (len(equilibria(P)) == 1 and equilibria(P)[0] < 0.5) for P in S])
    out = {}
    for lab, y in (("bistable", reg == "bistable"), ("trap_present", trap_present), ("trap_only", reg == "trap only")):
        out[lab] = {n: float(spearmanr(X[:, i], y).correlation) for i, n in enumerate(names)}
    # partial rank correlation (PRCC) for the six free parameters
    from scipy.stats import rankdata
    R = np.column_stack([rankdata(X[:, i]) for i in range(6)])
    for lab, y in (("bistable", reg == "bistable"), ("trap_present", trap_present)):
        yr = rankdata(y); pr = {}
        for i, n in enumerate(names[:6]):
            Z = np.column_stack([np.ones(len(yr)), np.delete(R, i, axis=1)])
            ex = R[:, i] - Z @ np.linalg.lstsq(Z, R[:, i], rcond=None)[0]
            ey = yr - Z @ np.linalg.lstsq(Z, yr, rcond=None)[0]
            pr[n] = float(np.corrcoef(ex, ey)[0, 1])
        out["prcc_" + lab] = pr
    return out

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "anchored":
        S = sample_parameters(ANCHORED_RANGES, 3000, seed=2)
        reg, shares = regimes(S); r = np.array([P.r for P in S])
        res = dict(n_sets=3000, counts={k: int(np.sum(reg == k)) for k in ("healthy only", "bistable", "trap only")}, all=shares)
        res["importance"] = importance(S, reg)
        # sampling and structural alternatives (regimes only)
        alt = {}
        alt["uniform rates"] = regimes(sample_parameters(ANCHORED_RANGES, 3000, seed=2, log_rates=()))[1]
        narrow = dict(ANCHORED_RANGES, n=(3.0, 6.0), h=(0.35, 0.55))
        alt["narrower social ranges"] = regimes(sample_parameters(narrow, 3000, seed=2))[1]
        for c0 in (0.1, 0.2, 0.3):
            alt[f"baseline capacity c0={c0}"] = regimes(sample_parameters(ANCHORED_RANGES, 3000, seed=2, fixed=dict(c0=c0)))[1]
        alt["saturating response n=1"] = regimes(sample_parameters(dict(ANCHORED_RANGES, n=(1.0, 1.0001)), 3000, seed=2))[1]
        # r and p positively correlated (persistent contaminants in more neglected areas), rank correlation about 0.5
        rng = np.random.default_rng(4); S2 = sample_parameters(ANCHORED_RANGES, 3000, seed=2)
        rr = np.argsort(np.argsort([P.r for P in S2])); pp = sorted(P.p for P in S2)
        noise = rr + rng.normal(0, 900, len(S2)); order = np.argsort(np.argsort(-noise))
        S2 = [P.with_(p=pp[order[i]]) for i, P in enumerate(S2)]
        alt["p higher where residence is longer"] = regimes(S2)[1]
        res["alternatives"] = alt
        for lab, m in [("long", r < 1/250), ("intermediate", (r >= 1/250) & (r < 1/70)), ("short", r >= 1/70)]:
            res[lab] = {k: float(np.mean(reg[m] == k)) for k in ("healthy only", "bistable", "trap only")}
        idx = np.where(reg == "bistable")[0]   # all bistable sets, no subsampling
        with Pool() as pool: R = pool.map(anchored_job, [S[i] for i in idx], chunksize=4)
        res["cleanup"] = summary([o["tS"] for o in R])
        for tau in (10, 20, 40):
            u = np.array([o[f"u{tau}"] for o in R], float); res[f"land_care_alone_{tau}"] = dict(n_sets=int(u.size), n_success=int(np.sum(~np.isnan(u))), share=float(np.mean(~np.isnan(u))))
        for tau in (20, 40, 80):
            res[f"cleanup_with_lc{tau}"] = summary([o[f"x_lc{tau}"] for o in R])
        res["unresolved_runs"] = int(sum(o["unresolved"] for o in R))
        res["cleanup_values"] = [float(o["tS"]) for o in R]
        res["per_set"] = R   # all per-set results, for paired comparisons
        # sensitivity: count unresolved runs as escapes instead of relapses
        from functools import partial
        with Pool() as pool: R2 = pool.map(partial(anchored_job, as_="healthy"), [S[i] for i in idx], chunksize=4)
        res["unresolved_as_escape"] = dict(cleanup=summary([o["tS"] for o in R2]),
            **{f"cleanup_with_lc{t}": summary([o[f"x_lc{t}"] for o in R2]) for t in (20, 40, 80)})
        json.dump(res, open("part_b_anchored.json", "w"), indent=1)
    else:
        S = sample_parameters(WIDE, 4000, seed=1, log_rates=())
        reg = np.array([regime(P) for P in S])
        res = dict(all={k: float(np.mean(reg == k)) for k in ("healthy only", "bistable", "trap only")})
        X = np.array([[P.r, P.p, P.k, P.d, P.n, P.h] for P in S]); B = reg == "bistable"
        res["spearman_bistable"] = {n: float(spearmanr(X[:, i], B).correlation) for i, n in enumerate("r p k d n h".split())}
        r = X[:, 0]; bins = np.linspace(0.005, 0.05, 7)
        res["by_r"] = {k: [float(np.mean(reg[(r >= a) & (r < b)] == k)) for a, b in zip(bins[:-1], bins[1:])] for k in ("healthy only", "bistable", "trap only")}
        with Pool() as pool: TT = pool.map(wide_job, [S[i] for i in np.where(B)[0]], chunksize=8)
        T = [t for t, _ in TT]; res["unresolved_runs"] = int(sum(u for _, u in TT))
        res["cleanup"] = summary(T); res["cleanup_values"] = [float(t) for t in T]
        json.dump(res, open("part_b_wide.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cleanup_values"}, indent=1))
