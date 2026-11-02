"""Checks that the package reproduces values reported in the paper."""
import math
from soilloop import Params, regime, equilibria, minimum_cleanup, compare_strategies

P = Params()

def test_illustrative_bistable():
    e = equilibria(P)
    assert regime(P) == "bistable"
    assert abs(e[0] - 0.318) < 0.01 and abs(e[1] - 0.400) < 0.01 and abs(e[2] - 0.898) < 0.01

def test_cleanup_threshold_21pct():
    assert abs(minimum_cleanup(P) - 0.21) < 0.02

def test_complementarity():
    out = compare_strategies(P, 0.125, 0.45, 10)
    assert out == {"cleanup only": False, "land care only": False, "both": True}

def test_persistence_regimes():
    assert regime(P.with_(r=0.010)) == "trap only"
    assert regime(P.with_(r=0.025)) == "healthy only"
