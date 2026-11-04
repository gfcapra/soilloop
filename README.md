# soilloop 0.1

A dynamical model and screening diagnostics for the loop between contaminated soils,
human health and community capacity in marginal areas.

Accompanies: Capra, G.F. *Breaking the loop: a dynamical model of contaminated soils,
human health and community decline in marginal areas* (in preparation).

## What it does

**Scenario explorer.** For a parameter set, or the literature-anchored classes of
contaminant residence time (long, intermediate, short; residence time depends on both
the element and the soil), it tells you whether the system
has a degraded stable state (alone or alongside a healthy one), how large a one-off
cleanup must be to reach the healthy state, and how cleanup and land care (a temporary
reduction of pressure) combine. Outcomes are classified by the attractor reached
(distance from a linearly stable equilibrium and a small vector field), not by a
threshold on soil condition alone.

**Diagnostics for real territories.** From a table of territorial units with soil,
health and community indicators, it builds 0-1 condition scores, tests whether the
units split into two groups and assigns each unit to a provisional group. Two groups
are flagged only if BIC prefers them by more than 10, Ashman's D > 2 and Hartigan's
dip test rejects unimodality (p < 0.05). In simulations this rule kept false positives
on unimodal data at 2% or less, but a common driver with a steep effect and no
feedback still produced two groups: bimodality is a screening signal, not evidence
of the loop. A function for early-warning
indicators on time series is also included.

## What it does not do

The model is conceptual. Its social parameters are not calibrated, so outputs are
**qualitative and exploratory**: regimes, thresholds as orders of magnitude, the
relative merit of strategies. It does not prescribe the cleanup needed in a specific
place. A two-group pattern in real data is *consistent* with the loop but does not
prove it. Group labels describe conditions imposed on territories, never the people
who live there, and should not be used to rank or stigmatize communities.

## Install and run

```bash
pip install .
cd examples
python make_synthetic_data.py      # invented data for testing
python quickstart.py
pytest ../tests                     # checks the numbers reported in the paper
```

## Reproducing the paper

```bash
cd analyses
python part_a_illustrative.py          # Sections 4.1-4.2 (minutes)
python part_b_sensitivity.py anchored  # Section 4.3 and Table S2 (tens of minutes)
python part_b_sensitivity.py wide      # Fig. S1
python part_c_robustness.py            # Section 4.4 (shocks, hysteresis, spatial)
python part_d_diagnostics.py           # Section 5 and Table S1
python make_figures.py                 # Figures 2-5, S1 and S2
```

## Model

    dS/dt = r (1 - S) - p (1 - C) S          soil condition
    dH/dt = k (S - H)                        health and human capital
    dC/dt = b [c0 + (1 - c0) H^n / (h^n + H^n)] - d C    community capacity (b = d)

S is the complement of a normalized contaminant load Q = 1 - S in the exposed soil
layers; neither is a measured concentration. `r` is the rate at which that load
declines (1 / residence time); `p` the pressure on land the community no longer cares
for; `c0` an optional baseline community capacity (0 by default). The default
illustrative set (r = 0.02, p = 0.06, k = 0.07, d = 0.1, n = 4, h = 0.4) lies inside
the ranges of the paper's Table 1. See the paper for ranges, sources and the
structural assumption behind bistability (a sigmoidal community response; with
n = 1 the model has no alternative states).

## Browser version

`web/soilloop_explorer.html` runs the same model and a simplified diagnosis
(diagonal-covariance mixtures, without the dip test) in any browser, with no installation.

## License and citation

MIT license. Please cite the article (see `CITATION.cff`).
Software archive: https://doi.org/10.5281/zenodo.23890631
