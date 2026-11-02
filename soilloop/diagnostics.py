"""Diagnostics for real territories.

Input: one row per territorial unit (neighbourhood, municipality) with indicators
of soil contamination, population health and community condition. The functions
normalise the indicators to 0-1 'condition' scores (1 = good), test whether units
split into two groups (the first empirical signature of the loop) and assign each
unit to a provisional group.

These are screening tools. A bimodal pattern is consistent with the loop but does
not prove it, and group labels describe conditions, never the people who live there.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def condition_score(x: pd.Series, higher_is_worse: bool, robust: bool = True) -> pd.Series:
    """Rescale an indicator to 0-1 where 1 = best condition."""
    x = x.astype(float)
    lo, hi = (x.quantile(0.02), x.quantile(0.98)) if robust else (x.min(), x.max())
    z = ((x - lo) / (hi - lo)).clip(0, 1)
    return 1 - z if higher_is_worse else z


def build_scores(df: pd.DataFrame, soil: dict, health: dict, community: dict) -> pd.DataFrame:
    """Each argument maps column name -> True if higher values are worse.
    Several columns per dimension are averaged."""
    out = pd.DataFrame(index=df.index)
    for name, spec in {"S": soil, "H": health, "C": community}.items():
        cols = [condition_score(df[c], worse) for c, worse in spec.items()]
        out[name] = pd.concat(cols, axis=1).mean(axis=1)
    return out


def bimodality_test(scores: pd.DataFrame, seed: int = 0) -> dict:
    """Compare one- and two-group Gaussian mixtures on the joint scores (BIC).
    A lower BIC for two groups, with well separated means, is consistent with
    alternative stable states."""
    X = scores[["S", "H", "C"]].to_numpy()
    g1 = GaussianMixture(1, covariance_type="full", random_state=seed).fit(X)
    g2 = GaussianMixture(2, covariance_type="full", n_init=10, random_state=seed).fit(X)
    bic1, bic2 = g1.bic(X), g2.bic(X)
    means = g2.means_
    order = np.argsort(means.mean(axis=1))
    sep = float(np.linalg.norm(means[order[1]] - means[order[0]]))
    return dict(bic_one_group=float(bic1), bic_two_groups=float(bic2),
                delta_bic=float(bic1 - bic2), two_groups_preferred=bool(bic2 < bic1 - 10),
                group_means={"low": means[order[0]].round(3).tolist(),
                             "high": means[order[1]].round(3).tolist()},
                separation=round(sep, 3), model=g2, order=order)


def classify(scores: pd.DataFrame, test: dict, uncertain_band=(0.2, 0.8)) -> pd.Series:
    """Provisional group for each unit from the two-group mixture:
    'low-condition group', 'high-condition group' or 'uncertain'."""
    X = scores[["S", "H", "C"]].to_numpy()
    prob_low = test["model"].predict_proba(X)[:, test["order"][0]]
    lab = np.where(prob_low >= uncertain_band[1], "low-condition group",
                   np.where(prob_low <= uncertain_band[0], "high-condition group", "uncertain"))
    return pd.Series(lab, index=scores.index, name="group")


def early_warning(series: pd.Series, window: int) -> pd.DataFrame:
    """Rolling variance and lag-1 autocorrelation of a detrended time series.
    Rising values before a decline are a fourth signature of the loop."""
    x = series.astype(float)
    resid = x - x.rolling(window, center=True, min_periods=window // 2).mean()
    return pd.DataFrame({
        "variance": resid.rolling(window).var(),
        "autocorrelation": resid.rolling(window).apply(lambda w: pd.Series(w).autocorr(1), raw=False),
    })
