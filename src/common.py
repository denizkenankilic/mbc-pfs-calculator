"""Shared helpers: paths, SAS reading, audit logging, time conversions.

Project: Routine blood tests and clinical factors for PFS prediction in
metastatic breast cancer (Project Data Sphere comparator arms, 5 phase III trials).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUT_TAB = ROOT / "outputs" / "tables"
OUT_FIG = ROOT / "outputs" / "figures"
for _p in (INTERIM, PROCESSED, OUT_TAB, OUT_FIG):
    _p.mkdir(parents=True, exist_ok=True)

DAYS_PER_MONTH = 30.4375  # 365.25 / 12, as used by the Lilly ADaM specification

# Baseline window for laboratory / vital data, in days relative to randomization
# (randomization day = 0). Prespecified: last non-missing value from day -35 to day +1.
BASELINE_WINDOW = (-35, 1)

# Trial metadata (comparator arms only, as shared on Project Data Sphere)
TRIALS = {
    "PFE111": dict(assess_interval_days=42, nct="NCT00373113", sponsor="Pfizer", setting="pretreated",
                   control="Capecitabine", pds="Breast_Pfizer_2006_111"),
    "PFE112": dict(assess_interval_days=56, nct="NCT00373256", sponsor="Pfizer", setting="first-line",
                   control="Bevacizumab + paclitaxel", pds="Breast_Pfizer_2006_112"),
    "PFE113": dict(assess_interval_days=42, nct="NCT00435409", sponsor="Pfizer", setting="pretreated",
                   control="Capecitabine", pds="Breast_Pfizer_2007_113"),
    "LLY168": dict(assess_interval_days=42, nct="NCT00703326", sponsor="Eli Lilly", setting="first-line",
                   control="Placebo + docetaxel", pds="Breast_EliLill_2008_168"),
    "SAN135": dict(assess_interval_days=42, nct="NCT00081796", sponsor="Sanofi", setting="pretreated",
                   control="Capecitabine", pds="Breast_SanofiU_2004_135"),
}


def read_sas(path: Path) -> pd.DataFrame:
    """Read a SAS7BDAT file; strips whitespace in text columns and turns '.' / '' into NaN."""
    df = pd.read_sas(path, encoding="latin1")
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            s = df[c].astype("string").str.strip()
            df[c] = s.mask(s.isin(["", ".", "nan", "NaN"])).astype(object)
    return df


@dataclass
class Audit:
    """Collects a row-per-action cleaning log (exported as Supplementary Table)."""
    rows: list = field(default_factory=list)

    def log(self, trial: str, step: str, variable: str, n: int, detail: str = ""):
        self.rows.append(dict(trial=trial, step=step, variable=variable, n_affected=int(n), detail=detail))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


AUDIT = Audit()


def to_months(days) -> pd.Series:
    return pd.Series(days, dtype="float") / DAYS_PER_MONTH


def last_in_window(df: pd.DataFrame, id_col: str, day_col: str, value_col: str,
                   window=BASELINE_WINDOW) -> pd.Series:
    """Per patient, the last non-missing value whose day lies inside `window` (inclusive).
    Ties on the same day are resolved by the median of same-day values."""
    lo, hi = window
    d = df[[id_col, day_col, value_col]].dropna()
    d = d[(d[day_col] >= lo) & (d[day_col] <= hi)]
    if d.empty:
        return pd.Series(dtype=float)
    last_day = d.groupby(id_col)[day_col].transform("max")
    d = d[d[day_col] == last_day]
    return d.groupby(id_col)[value_col].median()


def yn(x) -> float:
    """Map yes/no-like values to 1/0 (NaN otherwise)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ("Y", "YES", "1", "TRUE", "POSITIVE"):
        return 1.0
    if s in ("N", "NO", "0", "FALSE", "NEGATIVE"):
        return 0.0
    return np.nan
