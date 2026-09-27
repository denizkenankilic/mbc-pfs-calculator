"""Performance measures for survival risk models on a held-out trial, bootstrap SEs and meta-analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.util import Surv

HORIZONS = (6.0, 12.0)
TAU = 12.0


class Calibrator:
    """Maps a risk score to survival probabilities with a one-covariate Cox model fitted on development data."""

    def fit(self, score, dev):
        s = np.asarray(score, float)
        self.mu, self.sd = s.mean(), s.std() if s.std() > 0 else 1.0
        z = (s - self.mu) / self.sd
        self.cph = CoxPHFitter().fit(pd.DataFrame({"z": z, "time": dev.time.values, "event": dev.event.values}),
                                     "time", "event")
        self.beta = float(self.cph.params_["z"])
        return self

    def lp(self, score):
        return self.beta * (np.asarray(score, float) - self.mu) / self.sd

    def surv(self, score, times):
        z = (np.asarray(score, float) - self.mu) / self.sd
        sf = self.cph.predict_survival_function(pd.DataFrame({"z": z}), times=times)
        return sf.values.T  # n x len(times)


def metrics(dev, test, score_test, cal: Calibrator):
    """All per-trial measures (IPCW metrics weighted by the held-out censoring distribution)."""
    y_dev = Surv.from_arrays(dev.event.values, dev.time.values)
    y_te = Surv.from_arrays(test.event.values, test.time.values)
    risk = np.asarray(score_test, float)
    out = {"harrell_c": concordance_index_censored(test.event.values, test.time.values, risk)[0]}
    # restrict IPCW metrics to times observable in both sets
    # IPCW weights use the censoring distribution of the held-out trial itself (external validation)
    tmax = test.time[test.event].max() if test.event.any() else test.time.max()
    tau = min(TAU, tmax * 0.999)
    try:
        out["uno_c"] = concordance_index_ipcw(y_te, y_te, risk, tau=tau)[0]
    except Exception:
        out["uno_c"] = np.nan
    for h in HORIZONS:
        try:
            out[f"auc_{int(h)}"] = cumulative_dynamic_auc(y_te, y_te, risk, [h])[0][0] if h < tmax else np.nan
        except Exception:
            out[f"auc_{int(h)}"] = np.nan
    # integrated Brier score over 1..12 months (within follow-up of test set)
    grid = np.linspace(1, min(12.0, test.time.max() * 0.95), 23)
    try:
        S = cal.surv(risk, grid)
        out["ibs_12"] = integrated_brier_score(y_te, y_te, S, grid)
    except Exception:
        out["ibs_12"] = np.nan
    # calibration slope
    lp = cal.lp(risk)
    try:
        c = CoxPHFitter().fit(pd.DataFrame({"lp": lp, "time": test.time.values, "event": test.event.values}),
                              "time", "event")
        out["cal_slope"] = float(c.params_["lp"])
    except Exception:
        out["cal_slope"] = np.nan
    # observed / expected 6-month progression risk
    km = KaplanMeierFitter().fit(test.time, test.event)
    obs6 = 1 - float(km.survival_function_at_times(6.0).iloc[0])
    exp6 = float(np.mean(1 - cal.surv(risk, [6.0])[:, 0]))
    out["oe_6"] = obs6 / exp6 if exp6 > 0 else np.nan
    return out


def bootstrap(dev, test, score_test, cal, B=200, seed=2026, keys=None):
    rng = np.random.default_rng(seed)
    n = len(test)
    rows = []
    score_test = np.asarray(score_test, float)
    for _ in range(B):
        i = rng.integers(0, n, n)
        tb = test.iloc[i].reset_index(drop=True)
        if tb.event.sum() < 5:
            continue
        m = metrics(dev, tb, score_test[i], cal)
        rows.append({k: m[k] for k in (keys or m)})
    return pd.DataFrame(rows)


def boot_indices(n, B=200, seed=2026):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(B)]


def random_effects(est, se):
    """DerSimonian–Laird pooled estimate, 95% CI, I^2 and 95% prediction interval."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    k = len(est)
    if k == 0:
        return dict(pooled=np.nan, lo=np.nan, hi=np.nan, i2=np.nan, pi_lo=np.nan, pi_hi=np.nan, k=0)
    w = 1 / se ** 2
    fe = np.sum(w * est) / np.sum(w)
    q = np.sum(w * (est - fe) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 else 0.0
    wr = 1 / (se ** 2 + tau2)
    mu = np.sum(wr * est) / np.sum(wr)
    se_mu = np.sqrt(1 / np.sum(wr))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    from scipy import stats
    tcrit = stats.t.ppf(0.975, k - 2) if k > 2 else np.nan
    pi = tcrit * np.sqrt(tau2 + se_mu ** 2) if k > 2 else np.nan
    return dict(pooled=mu, lo=mu - 1.96 * se_mu, hi=mu + 1.96 * se_mu, i2=i2,
                pi_lo=mu - pi if k > 2 else np.nan, pi_hi=mu + pi if k > 2 else np.nan, k=k, tau2=tau2)


logit = lambda p: np.log(p / (1 - p))
expit = lambda x: 1 / (1 + np.exp(-x))


def net_benefit(time, event, risk6, thresholds, horizon=6.0):
    """Survival decision curve (Vickers et al. 2008): KM-based event proportion among those above threshold."""
    time, event, risk6 = map(np.asarray, (time, event, risk6))
    n = len(time)
    km_all = KaplanMeierFitter().fit(time, event)
    p_all = 1 - float(km_all.survival_function_at_times(horizon).iloc[0])
    rows = []
    for pt in thresholds:
        sel = risk6 >= pt
        if sel.sum() == 0:
            nb = 0.0
        else:
            km = KaplanMeierFitter().fit(time[sel], event[sel])
            p_ev = 1 - float(km.survival_function_at_times(horizon).iloc[0])
            frac = sel.mean()
            nb = p_ev * frac - (1 - p_ev) * frac * pt / (1 - pt)
        rows.append(dict(threshold=pt, net_benefit=nb, treat_all=p_all - (1 - p_all) * pt / (1 - pt)))
    return pd.DataFrame(rows)
