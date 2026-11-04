"""Section 4.4: shocks, heterogeneity, connection between neighborhoods.

All runs use the illustrative set (every parameter inside the anchored ranges of
Table 1, p = 0.06), plus a check on a sample of anchored bistable sets.
Outcomes are classified by basin of attraction (deterministic continuation of the
final state), not by a threshold on S alone.
"""
import json, sys, numpy as np
from soilloop import Params, equilibria, state_at, simulate_shocks, minimum_cleanup, regime, sample_parameters, ANCHORED_RANGES
from soilloop.stochastic import basin, wilson
from soilloop import spatial

part = sys.argv[1] if len(sys.argv) > 1 else "all"
P = Params()   # illustrative set, p = 0.06
lo, mid, hi = equilibria(P)
trap, healthy = state_at(lo, P), state_at(hi, P)
out = json.load(open("part_c.json")) if part != "all" else {}
out["params"] = P.as_dict(); out["equilibria"] = [lo, mid, hi]
out["deterministic_min_cleanup"] = minimum_cleanup(P)
R = 400; YEARS = 400

def escape_curve(sigma, noise="additive", dt=0.1, xs=None, seed0=0):
    rows = []
    for j, x in enumerate(xs):
        S, H, C, rec = simulate_shocks(P, np.full(R, trap[0] + x * (1 - trap[0])), np.full(R, trap[1]), np.full(R, trap[2]),
                                       years=YEARS, sigma=sigma, seed=seed0 + j, noise=noise, dt=dt, record_every=20)
        final = basin(P, S, H, C)
        ever = np.zeros(R, bool)
        for (s, h, c) in rec:
            ever |= basin(P, s, h, c, years=1500, dt=1.0)
        k = int(final.sum())
        rows.append(dict(x=float(x), healthy_at_400=k / R, ci=wilson(k, R),
                         ever_healthy=float(ever.mean()), returned=float((ever & ~final).mean())))
    return rows

if part in ("all", "shocks"):
    xs = np.round(np.arange(0, 0.91, 0.05), 2); out["xs"] = xs.tolist(); out["escape"] = {}
    for i, s in enumerate((0.0, 0.01, 0.02, 0.04)):
        out["escape"][str(s)] = escape_curve(s, xs=xs, seed0=100 * i)
    # sensitivity: time step and noise type, at sigma = 0.02
    sub = np.array([0.2, 0.4, 0.6])
    out["dt_check"] = dict(dt01=escape_curve(0.02, xs=sub, dt=0.1, seed0=900), dt005=escape_curve(0.02, xs=sub, dt=0.05, seed0=900))
    out["multiplicative"] = escape_curve(0.08, noise="multiplicative", xs=sub, seed0=950)
    out["healthy_stays"] = {}
    for s in (0.01, 0.02, 0.04):
        S, H, C = simulate_shocks(P, np.full(R, healthy[0]), np.full(R, healthy[1]), np.full(R, healthy[2]), years=YEARS, sigma=s, seed=7)
        k = int(basin(P, S, H, C).sum()); out["healthy_stays"][str(s)] = dict(share=k / R, ci=wilson(k, R))
    json.dump(out, open("part_c.json", "w"), indent=1)

if part in ("all", "anchored"):
    # 40 anchored bistable sets: cleanup = deterministic minimum + 10 points, sigma = 0.02
    S_all = sample_parameters(ANCHORED_RANGES, 3000, seed=2)
    bis = [Q for Q in S_all if regime(Q) == "bistable"]
    rng = np.random.default_rng(11); pick = [bis[i] for i in rng.choice(len(bis), 60, replace=False)]
    res = []
    for m, Q in enumerate(pick):
        x = minimum_cleanup(Q, max_step=5)
        if np.isnan(x) or x > 0.9:
            continue
        e = equilibria(Q); t = state_at(e[0], Q); x2 = min(1.0, x + 0.10)
        S, H, C = simulate_shocks(Q, np.full(200, t[0] + x2 * (1 - t[0])), np.full(200, t[1]), np.full(200, t[2]),
                                  years=YEARS, sigma=0.02, seed=500 + m)
        res.append(float(basin(Q, S, H, C).mean()))
        if len(res) == 40:
            break
    out["anchored_shocks"] = dict(n_sets=len(res), median=float(np.median(res)), q25=float(np.percentile(res, 25)),
                                  q75=float(np.percentile(res, 75)), values=res)
    json.dump(out, open("part_c.json", "w"), indent=1)

if part in ("all", "hysteresis"):
    from soilloop.model import stable
    ps = np.round(np.arange(0.03, 0.1201, 0.0025), 4); out["ps"] = ps.tolist(); out["sweeps"] = {}
    P0 = Params()
    folds = [float(p) for p in np.arange(0.03, 0.12, 0.0001) if len(equilibria(P0.with_(p=p))) == 3]
    out["static_folds"] = [min(folds), max(folds)]
    runs = [(0, 0, 0, 50), (0, 0, 0, 200), (0, 0, 0, 1000), (0, 0, 0, 5000),
            (0, 0, 0.01, 200), (0.3, 0, 0.01, 200), (0.3, 0.05, 0.01, 200), (0.6, 0.05, 0.01, 200),
            (0.3, 0.05, 0.03, 200), (0.3, 0.05, 0.01, 1000), (0.6, 0.05, 0.01, 1000)]
    for cv, D, s, T in runs:
        L = 1 if (cv == 0 and s == 0) else 12
        up, down = spatial.hysteresis_sweep(P0, ps, L=L, cv=cv, D=D, sigma=s, years_per_step=T)
        thr = 0.5
        coll = next((float(p) for p, v in zip(ps, up) if v < thr), None)
        reco = next((float(p) for p, v in zip(ps[::-1], down[::-1]) if v > thr), None)
        out["sweeps"][f"{cv}_{D}_{s}_{T}"] = dict(up=up.tolist(), down=down.tolist(),
            mean_vertical_gap=float(np.trapezoid(up - down, ps) / (ps[-1] - ps[0])), collapse_p=coll, recovery_p=reco)
    json.dump(out, open("part_c.json", "w"), indent=1)

if part in ("all", "sync"):
    P0 = Params(); out["sync"] = {}
    for p in (0.056, 0.059, 0.062, 0.065, 0.068):
        out["sync"][str(p)] = {f"{lab}_{st}": spatial.spatial_metrics(P0, p, D=D, start=st)
                               for lab, D in (("isolated", 0.0), ("connected", 0.05)) for st in ("random", "middle")}
    json.dump(out, open("part_c.json", "w"), indent=1)

print(json.dumps({k: v for k, v in out.items() if k not in ("sweeps",)}, indent=0, default=str)[:4000])
