"""Model wrappers with a common interface: fit(dev_df) -> self ; risk(df) -> higher = worse prognosis.

Each wrapper owns its own Preprocessor so that every step is re-fitted inside each development set.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from model_prep import CLIN, LAB, SEED, Preprocessor, surv_y

warnings.filterwarnings("ignore")
N_INNER = 5


def cindex(df, risk):
    return concordance_index_censored(df["event"].values, df["time"].values, np.asarray(risk))[0]


def inner_cv(make, df, grid):
    """Return best params (max mean Harrell C over 5 inner folds of the development data)."""
    kf = KFold(N_INNER, shuffle=True, random_state=SEED)
    best, best_c = None, -1
    for params in grid:
        cs = []
        for tr, va in kf.split(df):
            m = make(**params).fit(df.iloc[tr])
            cs.append(cindex(df.iloc[va], m.risk(df.iloc[va])))
        c = float(np.mean(cs))
        if c > best_c:
            best, best_c = params, c
    return best, best_c


# ---------------------------------------------------------------- M0 score
SCORE_ITEMS = {  # name: (variable, rule)
    "ECOG ≥1": ("ecog_ge1", lambda x: x >= 1),
    "Visceral metastases": ("visceral_mets", lambda x: x >= 1),
    "≥3 organ sites": ("n_met_sites", lambda x: x >= 3),
    "Pretreated setting": ("pretreated", lambda x: x >= 1),
    "Haemoglobin <12 g/dL": ("hgb", lambda x: x < 12),
    "Albumin <3.5 g/dL": ("alb", lambda x: x < 3.5),
    "ALP >1×ULN": ("alp_uln", lambda x: x > 1),
    "Neutrophils >7.5×10⁹/L": ("anc", lambda x: x > 7.5),
}


class PointsScore:
    name = "M0 Points score"

    def __init__(self):
        self.pre = Preprocessor(sorted({v for v, _ in SCORE_ITEMS.values()}))

    def items(self, df):
        R = self.pre.transform_raw(df)
        return pd.DataFrame({k: rule(R[v]).astype(float) for k, (v, rule) in SCORE_ITEMS.items()}, index=df.index)

    def fit(self, df):
        self.pre.fit(df)
        I = self.items(df)
        var = [c for c in I.columns if I[c].std() > 0]  # items constant in development data get 0 points
        cph = CoxPHFitter(penalizer=0.01).fit(I[var].assign(time=df.time.values, event=df.event.values), "time", "event")
        b = cph.params_.reindex(I.columns).fillna(0.0)
        # one point per 0.25 log-hazard units (HR ~1.28); items with non-positive effect receive 0 points
        self.points = (b / 0.25).round().clip(lower=0).astype(int)
        self.coef = b
        return self

    def risk(self, df):
        return (self.items(df) * self.points).sum(axis=1).values.astype(float)


# ---------------------------------------------------------------- Cox
class Cox:
    def __init__(self, features, name, penalizer=0.01):
        self.features, self.name, self.penalizer = features, name, penalizer
        self.pre = Preprocessor(features)

    def fit(self, df):
        X = self.pre.fit(df).transform(df)
        self.keep = [c for c in X.columns if X[c].std() > 1e-8]  # drop predictors constant in development data
        self.cph = CoxPHFitter(penalizer=self.penalizer).fit(
            X[self.keep].assign(time=df.time.values, event=df.event.values), "time", "event")
        return self

    def risk(self, df):
        return self.cph.predict_log_partial_hazard(self.pre.transform(df)[self.keep]).values


class LassoCox:
    name = "M3 LASSO-Cox"

    def __init__(self, alpha=None):
        self.alpha = alpha
        self.features = CLIN + LAB
        self.pre = Preprocessor(self.features)

    def fit(self, df):
        X = self.pre.fit(df).transform(df)
        y = surv_y(df)
        if self.alpha is None:  # tune alpha on the regularisation path by inner CV
            path = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=30).fit(X, y).alphas_
            kf = KFold(N_INNER, shuffle=True, random_state=SEED)
            scores = np.zeros(len(path))
            for tr, va in kf.split(X):
                m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=path).fit(X.iloc[tr], y[tr])
                for j, a in enumerate(path):
                    scores[j] += cindex(df.iloc[va], m.predict(X.iloc[va], alpha=a))
            self.alpha = float(path[int(np.argmax(scores))])
        self.m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[self.alpha]).fit(X, y)
        return self

    def risk(self, df):
        return self.m.predict(self.pre.transform(df))

    def coefs(self):
        return pd.Series(self.m.coef_[:, 0], index=self.features)


# ---------------------------------------------------------------- RSF
class RSF:
    name = "M4 Random survival forest"
    GRID = [dict(min_samples_leaf=l, max_features=f) for l, f in itertools.product([15, 30], ["sqrt", 0.5])]

    def __init__(self, min_samples_leaf=15, max_features="sqrt", n_estimators=300):
        self.kw = dict(min_samples_leaf=min_samples_leaf, max_features=max_features, n_estimators=n_estimators)
        self.features = CLIN + LAB
        self.pre = Preprocessor(self.features)

    def fit(self, df):
        X = self.pre.fit(df).transform(df)
        self.m = RandomSurvivalForest(random_state=SEED, n_jobs=2, **self.kw).fit(X, surv_y(df))
        return self

    def risk(self, df):
        return self.m.predict(self.pre.transform(df))


# ---------------------------------------------------------------- XGBoost Cox
class XGBCox:
    name = "M5 Gradient-boosted Cox"
    GRID = [dict(max_depth=d, n_estimators=n) for d, n in itertools.product([2, 3], [100, 250])]

    def __init__(self, max_depth=2, n_estimators=150, learning_rate=0.03, impute=True):
        self.kw = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
        self.features = CLIN + LAB
        self.pre = Preprocessor(self.features, impute=impute)

    def fit(self, df):
        import xgboost as xgb
        X = self.pre.fit(df).transform(df)
        y = np.where(df.event.values, df.time.values, -df.time.values)
        self.m = xgb.XGBRegressor(objective="survival:cox", subsample=0.8, colsample_bytree=0.8,
                                  min_child_weight=5, reg_lambda=1.0, random_state=SEED, n_jobs=2,
                                  tree_method="hist", **self.kw).fit(X, y)
        return self

    def risk(self, df):
        return np.log(self.m.predict(self.pre.transform(df)))


# ---------------------------------------------------------------- DeepSurv
class DeepSurv:
    name = "M6 DeepSurv"
    GRID = [dict(hidden=h, dropout=d) for h, d in itertools.product([(16,), (32, 16)], [0.1, 0.3])]

    def __init__(self, hidden=(16,), dropout=0.1, weight_decay=1e-3, lr=1e-3, max_epochs=300):
        self.hidden, self.dropout, self.wd, self.lr, self.max_epochs = hidden, dropout, weight_decay, lr, max_epochs
        self.features = CLIN + LAB
        self.pre = Preprocessor(self.features)

    @staticmethod
    def _loss(risk, t, e):
        import torch
        order = torch.argsort(t, descending=True)
        r, ev = risk[order], e[order]
        log_cum = torch.logcumsumexp(r, dim=0)
        return -torch.sum((r - log_cum) * ev) / torch.clamp(ev.sum(), min=1)

    def _net(self, p):
        import torch.nn as nn
        layers, d = [], p
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1, bias=False))
        return nn.Sequential(*layers)

    def fit(self, df):
        import torch
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        X = self.pre.fit(df).transform(df).values.astype(np.float32)
        t = df.time.values.astype(np.float32)
        e = df.event.values.astype(np.float32)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(X))
        nv = int(0.2 * len(X))
        va, tr = idx[:nv], idx[nv:]
        T = lambda a: torch.tensor(a)
        net = self._net(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, patience = np.inf, None, 0
        for ep in range(self.max_epochs):
            net.train()
            opt.zero_grad()
            loss = self._loss(net(T(X[tr])).squeeze(1), T(t[tr]), T(e[tr]))
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                vl = self._loss(net(T(X[va])).squeeze(1), T(t[va]), T(e[va])).item()
            if vl < best - 1e-4:
                best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 30:
                    break
        net.load_state_dict(best_state)
        self.net = net.eval()
        return self

    def risk(self, df):
        import torch
        X = torch.tensor(self.pre.transform(df).values.astype(np.float32))
        with torch.no_grad():
            return self.net(X).squeeze(1).numpy()


def make_models(dev, tune=True):
    """Instantiate and fit all models on a development set (inner-CV tuning where applicable)."""
    fitted, tuning, factories = {}, {}, {}
    factories["M0"] = lambda: PointsScore()
    factories["M1"] = lambda: Cox(CLIN, "M1 Cox (clinical)")
    factories["M2"] = lambda: Cox(CLIN + LAB, "M2 Cox (clinical + blood)")
    for k in ("M0", "M1", "M2"):
        fitted[k] = factories[k]().fit(dev)
    fitted["M3"] = LassoCox().fit(dev)
    tuning["M3"] = dict(alpha=fitted["M3"].alpha)
    a3 = fitted["M3"].alpha
    factories["M3"] = lambda: LassoCox(alpha=a3)
    for key, cls in (("M4", RSF), ("M5", XGBCox), ("M6", DeepSurv)):
        if tune:
            params, c = inner_cv(cls, dev, cls.GRID)
        else:
            params, c = {}, np.nan
        tuning[key] = dict(params=params, inner_c=round(c, 4) if c == c else c)
        fitted[key] = cls(**params).fit(dev)
        factories[key] = (lambda cls=cls, params=params: cls(**params))
    return fitted, tuning, factories


def oof_scores(factory, dev):
    """Out-of-fold development risk scores (5-fold, fixed hyper-parameters) for fitting the calibration map,
    so that the map is not distorted by in-sample overfitting of flexible models."""
    kf = KFold(N_INNER, shuffle=True, random_state=SEED)
    s = np.full(len(dev), np.nan)
    for tr, va in kf.split(dev):
        m = factory().fit(dev.iloc[tr])
        s[va] = m.risk(dev.iloc[va])
    return s


class StratCox(Cox):
    """Cox model stratified by trial during development (sensitivity analysis): coefficients reflect
    within-trial associations only; trial-level 'pretreated' is dropped (absorbed by strata)."""

    def __init__(self, features, name="M2s Cox stratified by trial", penalizer=0.01):
        super().__init__([f for f in features if f != "pretreated"], name, penalizer)

    def fit(self, df):
        X = self.pre.fit(df).transform(df)
        self.cph = CoxPHFitter(penalizer=self.penalizer).fit(
            X.assign(time=df.time.values, event=df.event.values, trial=df.trial.values), "time", "event",
            strata=["trial"])
        return self

    def risk(self, df):
        X = self.pre.transform(df)
        return (X[self.cph.params_.index] @ self.cph.params_).values
