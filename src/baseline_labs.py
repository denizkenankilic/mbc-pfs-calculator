"""Builds one-row-per-patient baseline laboratory values from a harmonised long table.

Input long table columns: id, day (relative to randomisation), analyte, value, unit, uln
Analyte keys follow labs.SPEC; differential percentages use keys anc_pct/alc_pct/amc_pct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import AUDIT, BASELINE_WINDOW, last_in_window
from labs import SPEC, harmonise, uln_ratio

ULN_ANALYTES = ["alp", "ast", "alt", "bili", "creat"]


def build_baseline(long: pd.DataFrame, trial: str, ids) -> pd.DataFrame:
    ids = pd.Index(ids, name="id")
    base = pd.DataFrame(index=ids)
    lo, hi = BASELINE_WINDOW

    # --- absolute analytes -------------------------------------------------
    harmonised = {}
    for a in SPEC:
        sub = long[long.analyte == a]
        if sub.empty:
            continue
        harmonised[a] = harmonise(sub, trial, a)

    # consistency check: an absolute differential count cannot exceed same-day WBC
    # (catches percentages entered under the absolute-count test name)
    if "wbc" in harmonised:
        w = harmonised["wbc"].dropna(subset=["val"]).groupby(["id", "day"]).val.median().rename("wbc_sameday")
        for a in ("anc", "alc", "amc"):
            if a not in harmonised:
                continue
            h = harmonised[a].join(w, on=["id", "day"])
            bad = h["val"].notna() & h["wbc_sameday"].notna() & (h["val"] > 1.02 * h["wbc_sameday"])
            if bad.any():
                AUDIT.log(trial, "lab_inconsistent_with_wbc", a, bad.sum(),
                          "absolute differential > same-day WBC (likely %); set to missing")
            h.loc[bad, "val"] = np.nan
            harmonised[a] = h.drop(columns="wbc_sameday")

    for a in SPEC:
        if a in harmonised:
            base[a] = last_in_window(harmonised[a], "id", "day", "val").reindex(ids)
        else:
            base[a] = np.nan

    # --- WBC differential given as % : absolute = % x same-day WBC -----------
    if "wbc" in harmonised:
        wbc = harmonised["wbc"].dropna(subset=["val"])[["id", "day", "val"]].rename(columns={"val": "wbc"})
        wbc = wbc.groupby(["id", "day"], as_index=False).wbc.median()
        for a in ("anc", "alc", "amc"):
            pct = long[long.analyte == f"{a}_pct"].copy()
            pct_in = pd.DataFrame()
            if a in harmonised:  # differential reported with unit '%' under the absolute test name
                pct_in = harmonised[a][harmonised[a].qc == "percent"][["id", "day", "value"]]
            pct = pd.concat([pct[["id", "day", "value"]], pct_in], ignore_index=True)
            if pct.empty:
                continue
            pct["pct"] = pd.to_numeric(pct["value"], errors="coerce")
            pct = pct[(pct.pct >= 0) & (pct.pct <= 100)]
            m = pct.merge(wbc, on=["id", "day"], how="inner")
            m["val"] = m.pct / 100 * m.wbc
            derived = last_in_window(m, "id", "day", "val").reindex(ids)
            need = base[a].isna() & derived.notna()
            if need.any():
                AUDIT.log(trial, "lab_derived_from_percent", a, need.sum(),
                          "absolute count missing at baseline; derived as % x same-day WBC")
                base.loc[need, a] = derived[need]

    # --- ULN ratios (unit-independent) ---------------------------------------
    for a in ULN_ANALYTES:
        sub = long[(long.analyte == a)].copy()
        if sub.empty or "uln" not in sub:
            base[f"{a}_uln"] = np.nan
            continue
        sub["ratio"] = uln_ratio(sub, "value", "uln")
        base[f"{a}_uln"] = last_in_window(sub, "id", "day", "ratio").reindex(ids)

    # --- window diagnostics ------------------------------------------------
    for a in SPEC:
        n_any = long[(long.analyte == a)].id.nunique()
        n_base = base[a].notna().sum()
        AUDIT.log(trial, "baseline_available", a, n_base,
                  f"patients with value in window [{lo},{hi}] days; {n_any} with any value")
    return base


def add_inflammation_indices(df: pd.DataFrame) -> pd.DataFrame:
    """NLR, PLR, SII, LMR computed only when all components are present (no imputation)."""
    d = df.copy()
    alc = d["alc"].where(d["alc"] > 0)
    amc = d["amc"].where(d["amc"] > 0)
    d["nlr"] = d["anc"] / alc
    d["plr"] = d["plt"] / alc
    d["sii"] = d["plt"] * d["anc"] / alc  # 10^9/L
    d["lmr"] = d["alc"] / amc
    return d
