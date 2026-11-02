"""Section 4.4: shocks, heterogeneity, connection between neighbourhoods."""
import json, numpy as np
from soilloop import Params, equilibria, state_at, simulate_shocks
from soilloop import spatial
P = Params().with_(p=0.07)
lo, mid, hi = equilibria(P)
trap, healthy = state_at(lo, P), state_at(hi, P)
out = dict(mid=mid)
xs = np.round(np.arange(0, 0.91, 0.05), 2); out["xs"] = xs.tolist(); out["escape"] = {}
for i, s in enumerate((0.0, 0.01, 0.02, 0.04)):
    pr = []
    for j, x in enumerate(xs):
        R = 300
        S, H, C = simulate_shocks(P, np.full(R, trap[0] + x * (1 - trap[0])), np.full(R, trap[1]), np.full(R, trap[2]),
                                  years=400, sigma=s, seed=100 * i + j)
        pr.append(float(np.mean(S > mid)))
    out["escape"][str(s)] = pr
out["healthy_stays"] = {}
for s in (0.01, 0.02, 0.04):
    S, H, C = simulate_shocks(P, np.full(300, healthy[0]), np.full(300, healthy[1]), np.full(300, healthy[2]), years=400, sigma=s, seed=7)
    out["healthy_stays"][str(s)] = float(np.mean(S > mid))
P0 = Params()
ps = np.linspace(0.03, 0.12, 19); out["ps"] = ps.tolist(); out["sweeps"] = {}
for cv, D, s in [(0, 0, 0), (0, 0, 0.01), (0.3, 0, 0.01), (0.3, 0.05, 0.01), (0.6, 0.05, 0.01), (0.3, 0.05, 0.03)]:
    up, down = spatial.hysteresis_sweep(P0, ps, cv=cv, D=D, sigma=s)
    out["sweeps"][f"{cv}_{D}_{s}"] = dict(up=up.tolist(), down=down.tolist(), gap=float(np.trapezoid(up - down, ps) / (ps[-1] - ps[0])))
out["sync"] = {}
for p in (0.056, 0.059, 0.062, 0.068):
    out["sync"][str(p)] = dict(isolated=spatial.trapped_share(P0, p, D=0.0), connected=spatial.trapped_share(P0, p, D=0.05))
json.dump(out, open("part_c.json", "w"), indent=1)
print("mid", round(mid, 3)); [print(k, np.round(v, 2)) for k, v in out["escape"].items()]
print("healthy stays", out["healthy_stays"]); print({k: round(v["gap"], 3) for k, v in out["sweeps"].items()}); print(out["sync"])
