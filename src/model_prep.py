"""Feature definitions and a leakage-safe preprocessor (fit on development data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from common import PROCESSED

SEED = 2026

CLIN = ["age", "bmi", "ecog_ge1", "visceral_mets", "n_met_sites", "measurable", "prior_endocrine",
        "prior_adj_chemo", "prior_taxane", "prior_anthracycline", "n_prior_chemo_adv", "pretreated"]
LAB = ["hgb", "wbc", "anc", "plt", "alb", "alp_uln", "ast_uln", "bili_uln", "creat", "ca"]
LOG_VARS = {"wbc", "anc", "plt", "alp_uln", "ast_uln", "bili_uln", "creat", "alc", "nlr"}
BINARY = {"ecog_ge1", "visceral_mets", "measurable", "prior_endocrine", "prior_adj_chemo",
          "prior_taxane", "prior_anthracycline", "pretreated"}
EXTRA_LYMPH = ["alc", "nlr"]

PRETTY = {"age": "Age", "bmi": "BMI", "ecog_ge1": "ECOG ≥1", "visceral_mets": "Visceral metastases",
          "n_met_sites": "No. organ sites", "measurable": "Measurable disease", "prior_endocrine": "Prior endocrine therapy",
          "prior_adj_chemo": "Prior (neo)adjuvant chemo", "prior_taxane": "Prior taxane",
          "prior_anthracycline": "Prior anthracycline", "n_prior_chemo_adv": "Prior chemo lines (advanced)",
          "pretreated": "Pretreated setting", "hgb": "Haemoglobin", "wbc": "WBC", "anc": "Neutrophils",
          "plt": "Platelets", "alb": "Albumin", "alp_uln": "ALP (×ULN)", "ast_uln": "AST (×ULN)",
          "bili_uln": "Bilirubin (×ULN)", "creat": "Creatinine", "ca": "Calcium", "alc": "Lymphocytes", "nlr": "NLR"}


def load(outcome="pfs") -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "mbc_pooled.csv")
    df["ecog_ge1"] = (df["ecog"] >= 1).astype(float).where(df["ecog"].notna())
    df["pretreated"] = (df["setting"] == "pretreated").astype(float)
    df["time"] = df[f"{outcome}_months"]
    df["event"] = df[f"{outcome}_event"]
    df = df[df.time.notna() & df.event.notna()].copy()
    df["event"] = df["event"].astype(bool)
    return df.reset_index(drop=True)


class Preprocessor:
    """log-transform -> winsorise (1st/99th pct) -> iterative imputation -> standardise; all fitted on dev data."""

    def __init__(self, features, impute=True):
        self.features = list(features)
        self.impute = impute

    def _log(self, X):
        X = X.copy()
        for c in self.features:
            if c in LOG_VARS:
                X[c] = np.log(X[c].clip(lower=1e-3))
        return X

    def fit(self, df):
        X = self._log(df[self.features].astype(float))
        self.lo = X.quantile(0.01)
        self.hi = X.quantile(0.99)
        for c in BINARY & set(self.features):
            self.lo[c], self.hi[c] = 0, 1
        X = X.clip(self.lo, self.hi, axis=1)
        if self.impute:
            self.imp = IterativeImputer(max_iter=20, random_state=SEED, sample_posterior=False)
            Xi = pd.DataFrame(self.imp.fit_transform(X), columns=self.features)
        else:
            Xi = X
        self.mu = Xi.mean()
        self.sd = Xi.std().replace(0, 1)
        return self

    def transform_raw(self, df):
        """Imputed values on the original (clinical) scale — used by the points score."""
        X = self._log(df[self.features].astype(float)).clip(self.lo, self.hi, axis=1)
        if self.impute:
            X = pd.DataFrame(self.imp.transform(X), columns=self.features, index=df.index)
        for c in self.features:
            if c in LOG_VARS:
                X[c] = np.exp(X[c])
            if c in BINARY:
                X[c] = X[c].round().clip(0, 1)
        return X

    def transform(self, df):
        X = self._log(df[self.features].astype(float)).clip(self.lo, self.hi, axis=1)
        if self.impute:
            X = pd.DataFrame(self.imp.transform(X), columns=self.features, index=df.index)
        for c in BINARY & set(self.features):
            X[c] = X[c].round().clip(0, 1)
        return (X - self.mu) / self.sd


def surv_y(df):
    from sksurv.util import Surv
    return Surv.from_arrays(event=df["event"].values, time=df["time"].values)
