"""Quick start: scenarios and diagnostics."""
import pandas as pd
from soilloop import Params, PRESETS, regime, equilibria, minimum_cleanup, compare_strategies
from soilloop import diagnostics as dg

# 1. Scenario explorer -------------------------------------------------------
P = Params()                                  # illustrative set used in the paper
print("Regime:", regime(P), "| equilibria (soil condition):", [round(e, 3) for e in equilibria(P)])
print("Minimum one-off cleanup:", round(minimum_cleanup(P), 3))
print("Cleanup 12.5% + land care -45% for 10 yr:", compare_strategies(P, 0.125, 0.45, 10))
for name, kw in PRESETS.items():
    Q = Params(p=0.03, k=0.07, d=0.05, n=5, h=0.45).with_(**kw)
    print(f"{name:26s} -> {regime(Q)}")

# 2. Diagnostics on real (here: synthetic) data -----------------------------
df = pd.read_csv("synthetic_municipalities.csv").set_index("unit")
scores = dg.build_scores(df,
                         soil={"soil_Pb_mg_kg": True},
                         health={"premature_mortality_per_1000": True},
                         community={"deprivation_index": True})
test = dg.bimodality_test(scores)
print({k: v for k, v in test.items() if k not in ("model", "order")})
groups = dg.classify(scores, test)
print(groups.value_counts())
