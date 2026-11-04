"""Diagnostics for real territories.

Input: one row per territorial unit (neighborhood, municipality) with indicators
of soil contamination, population health and community condition. The functions
normalize the indicators to 0-1 'condition' scores (1 = good), test whether units
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


def ashman_d(model) -> float:
    """Ashman's D of a two-component mixture, measured along the axis that joins
    the two means. D > 2 indicates components separated enough to form two modes;
    a lower BIC alone does not (skewed or heavy-tailed unimodal data are often
    better fitted by two Gaussians)."""
    m = model.means_; v = m[1] - m[0]; u = v / np.linalg.norm(v)
    s2 = [float(u @ c @ u) for c in model.covariances_]
    return float(np.sqrt(2) * abs(u @ v) / np.sqrt(s2[0] + s2[1]))


def bimodality_test(scores: pd.DataFrame, seed: int = 0, n_init: int = 10) -> dict:
    """Compare one- and two-group Gaussian mixtures on the joint scores (BIC) and
    measure the separation of the two groups (Ashman's D) and the multimodality of
    the data along the axis joining them (Hartigan's dip test, if the optional
    package diptest is installed). Two groups are flagged only if BIC prefers them
    by more than 10, D > 2 and the dip test rejects unimodality (p < 0.05). Even then the pattern is
    consistent with alternative stable states but does not prove them: a common
    driver with a steep effect can produce it without any feedback."""
    X = scores[["S", "H", "C"]].to_numpy()
    g1 = GaussianMixture(1, covariance_type="full", random_state=seed).fit(X)
    g2 = GaussianMixture(2, covariance_type="full", n_init=n_init, random_state=seed).fit(X)
    bic1, bic2 = g1.bic(X), g2.bic(X)
    means = g2.means_
    order = np.argsort(means.mean(axis=1))
    sep = float(np.linalg.norm(means[order[1]] - means[order[0]]))
    D = ashman_d(g2)
    # Hartigan's dip test on the projection onto the axis joining the two means
    v = g2.means_[1] - g2.means_[0]; proj = X @ (v / np.linalg.norm(v))
    try:
        import diptest
        dip, dip_p = diptest.diptest(proj)
    except ImportError:          # optional dependency
        dip, dip_p = float("nan"), float("nan")
    flagged = bool(bic2 < bic1 - 10 and D > 2 and (np.isnan(dip_p) or dip_p < 0.05))
    return dict(bic_one_group=float(bic1), bic_two_groups=float(bic2),
                delta_bic=float(bic1 - bic2), ashman_d=round(D, 3), dip_p=float(dip_p),
                two_groups_preferred=flagged,
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
