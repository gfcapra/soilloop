"""Create a synthetic dataset of 300 municipalities to try the diagnostics.
The data are invented and generated from the model itself."""
import numpy as np, pandas as pd
from soilloop import Params, simulate_shocks

rng = np.random.default_rng(42)
n = 300
P = Params(r=0.02, p=0.07, k=0.5, d=0.1, n=4, h=0.4)
S, H, C = simulate_shocks(P, rng.random(n), rng.random(n), rng.random(n), years=600, sigma=0.01, seed=1)
df = pd.DataFrame({
    "unit": [f"M{i:03d}" for i in range(n)],
    # higher = worse
    "soil_Pb_mg_kg": np.round(30 + 900 * (1 - S) * rng.lognormal(0, 0.15, n), 1),
    "premature_mortality_per_1000": np.round(2 + 8 * (1 - H) + rng.normal(0, 0.6, n), 2),
    "deprivation_index": np.round(-2 + 4 * (1 - C) + rng.normal(0, 0.3, n), 2),
})
df.to_csv("synthetic_municipalities.csv", index=False)
print(df.describe().round(2))
