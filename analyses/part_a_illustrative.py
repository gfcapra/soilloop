"""Sections 4.1-4.2: illustrative parameter set."""
import json, sys, numpy as np
sys.setrecursionlimit(10000)
from soilloop import Params, equilibria, regime, minimum_cleanup, escapes, minimum_land_care, simulate, state_at
P = Params()
out = {}
e = equilibria(P); out["equilibria"] = e
out["bistable_p_range"] = [float(p) for p in np.arange(0.01, 0.12, 0.0005) if len(equilibria(P.with_(p=p))) == 3][::len([1])]
ps = [p for p in np.arange(0.01, 0.12, 0.0005) if len(equilibria(P.with_(p=p))) == 3]
out["bistable_p_min_max"] = [float(min(ps)), float(max(ps))]
out["min_cleanup"] = minimum_cleanup(P)
out["relapse_20_28"] = [escapes(P, 0.20), escapes(P, 0.28)]
from soilloop.model import stable, jacobian
out["stability"] = [stable(s, P) for s in e]
out["eigenvalues"] = [sorted(np.linalg.eigvals(jacobian(state_at(s, P), P)).real.tolist()) for s in e]
# persistence
rs = np.round(np.arange(0.0100, 0.0251, 0.0005), 4)
out["persistence"] = [(float(r), regime(P.with_(r=r)), (minimum_cleanup(P.with_(r=r)) if regime(P.with_(r=r)) == "bistable" else None)) for r in rs]
# community recovery
ds = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
out["community"] = [(1 / d, minimum_cleanup(P.with_(d=d))) for d in ds]
# complementarity
out["complementarity"] = dict(cleanup_only=escapes(P, 0.20), land_care_only=escapes(P, 0, 0.45, 10), both=escapes(P, 0.20, 0.45, 10))
# frontiers
xs = np.round(np.arange(0, 0.311, 0.01), 3)
out["frontier"] = {tau: [(float(x), minimum_land_care(P, x, tau)) for x in xs] for tau in (5, 10, 20)}
json.dump(out, open("part_a.json", "w"), default=float, indent=1)
print("eq", np.round(e, 3), "bistable p", out["bistable_p_min_max"], "min cleanup", out["min_cleanup"], out["relapse_20_28"], out["stability"])
print("persistence", [(r, g[:4], None if t is None else round(t, 3)) for r, g, t in out["persistence"]])
print("community", [(round(a, 1), round(b, 3)) for a, b in out["community"]])
print("complementarity", out["complementarity"])
for tau, f in out["frontier"].items(): print(tau, [(x, None if np.isnan(u) else round(u, 2)) for x, u in f][:3], '...', [(x, round(u, 2)) for x, u in f if not np.isnan(u)][-4:])
