# soilloop 0.1

A dynamical model and screening diagnostics for the loop between contaminated soils,
human health and community capacity in marginal areas.

Accompanies: Capra, G.F. *Breaking the loop: a dynamical model of contaminated soils,
human health and community decline in marginal areas* (in preparation).

## What it does

**Scenario explorer.** For a parameter set, or the literature-anchored classes of
contaminant residence time (long, intermediate, short; residence time depends on both
the element and the soil), it tells you whether the system
has a degradation trap, how large a one-off cleanup must be to escape it, and how
cleanup and land care (a temporary reduction of pressure) combine.

**Diagnostics for real territories.** From a table of territorial units with soil,
health and community indicators, it builds 0-1 condition scores, tests whether the
units split into two groups (an empirical signature of alternative stable states)
and assigns each unit to a provisional group. A function for early-warning
indicators on time series is also included.

## What it does not do

The model is conceptual. Its social parameters are not calibrated, so outputs are
**qualitative and exploratory**: regimes, thresholds as orders of magnitude, the
relative merit of strategies. It does not prescribe the cleanup needed in a specific
place. A two-group pattern in real data is *consistent* with the loop but does not
prove it. Group labels describe conditions imposed on territories, never the people
who live there, and should not be used to rank or stigmatise communities.

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
python part_b_sensitivity.py anchored  # Section 4.3 (tens of minutes)
python part_b_sensitivity.py wide      # Fig. S1
python part_c_robustness.py            # Section 4.4
python make_figures.py                 # Figures 2-5 and S1
```

## Model

    dS/dt = r (1 - S) - p (1 - C) S          soil condition
    dH/dt = k (S - H)                        health and human capital
    dC/dt = b H^n / (h^n + H^n) - d C        community capacity (b = d)

`r` is the inverse of the contaminant residence time; `p` the pressure on land the
community no longer cares for. See the paper, Table 1, for ranges and sources.

## Browser version

`web/soilloop_explorer.html` runs the same model and a simplified diagnosis
(diagonal-covariance mixtures) in any browser, with no installation.

## Licence and citation

MIT licence. Please cite the article (see `CITATION.cff`).
