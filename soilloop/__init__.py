"""soilloop: a dynamical model and screening diagnostics for the
contaminated soil - health - community loop in marginal areas.

Version 0.1 (exploratory). Accompanies: Capra, G.F. Breaking the loop: a dynamical
model of contaminated soils, human health and community decline in marginal areas.
"""
from .model import Params, PRESETS, ANCHORED_RANGES, equilibria, regime, simulate, state_at
from .scenarios import escapes, minimum_cleanup, minimum_land_care, compare_strategies, sample_parameters
from . import spatial
from .stochastic import simulate_shocks
from . import diagnostics

__version__ = "0.1.0"
