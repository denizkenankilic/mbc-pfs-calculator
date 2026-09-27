"""Laboratory harmonisation.

Each analyte is converted to one target unit using (1) an explicit unit-label
conversion table and (2) a prespecified physiological plausibility range.
Values whose label-based conversion is missing or implausible go through a
documented "magnitude rescue": candidate conversion factors are tried and the
value is kept only if exactly one candidate lands in the analyte's *typical*
range. Everything else is set to missing. Every action is written to the
audit log (see common.AUDIT).

Upper-limit-of-normal (ULN) ratios are also derived from the record's own
reference range (same unit as the raw value), which is unit-independent and is
the preferred scale for liver enzymes and bilirubin.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import AUDIT

# ---------------------------------------------------------------------------
# Analyte specification
#   target: target unit
#   plaus:  absolute plausibility range in target unit (outside -> missing)
#   typical: narrower range used only to disambiguate magnitude rescue
#   cand:   candidate factors tried during rescue (multiplied with raw value)
# ---------------------------------------------------------------------------
SPEC = {
    "hgb":    dict(label="Haemoglobin", target="g/dL", plaus=(3, 20), typical=(6, 18), cand=[1, 0.1, 1.611]),
    "wbc":    dict(label="White blood cells", target="10^9/L", plaus=(0.3, 150), typical=(1.5, 30), cand=[1, 0.001]),
    "anc":    dict(label="Neutrophils (absolute)", target="10^9/L", plaus=(0.05, 120), typical=(0.5, 25), cand=[1, 0.001]),
    "alc":    dict(label="Lymphocytes (absolute)", target="10^9/L", plaus=(0.02, 50), typical=(0.2, 6), cand=[1, 0.001]),
    "amc":    dict(label="Monocytes (absolute)", target="10^9/L", plaus=(0.0, 5), typical=(0.05, 2.5), cand=[1, 0.001]),
    "plt":    dict(label="Platelets", target="10^9/L", plaus=(5, 2000), typical=(50, 900), cand=[1, 0.001]),
    "alb":    dict(label="Albumin", target="g/dL", plaus=(1.0, 6.5), typical=(2.0, 5.5), cand=[1, 0.1, 0.001]),
    "alp":    dict(label="Alkaline phosphatase", target="U/L", plaus=(10, 6000), typical=(20, 3000), cand=[1, 60]),
    "ast":    dict(label="AST", target="U/L", plaus=(3, 5000), typical=(5, 1000), cand=[1, 60]),
    "alt":    dict(label="ALT", target="U/L", plaus=(2, 5000), typical=(3, 1000), cand=[1, 60]),
    "bili":   dict(label="Total bilirubin", target="mg/dL", plaus=(0.02, 40), typical=(0.05, 10), cand=[1, 1 / 17.104, 0.1]),
    "creat":  dict(label="Creatinine", target="mg/dL", plaus=(0.1, 20), typical=(0.2, 5), cand=[1, 1 / 88.42, 0.1, 1000 / 88.42]),
    "ca":     dict(label="Calcium", target="mg/dL", plaus=(5, 16), typical=(6, 16), cand=[1, 4.008, 2.004, 0.1]),
    "tprot":  dict(label="Total protein", target="g/dL", plaus=(2, 14), typical=(4, 11), cand=[1, 0.1]),
    "na":     dict(label="Sodium", target="mmol/L", plaus=(100, 185), typical=(115, 165), cand=[1]),
    "glu":    dict(label="Glucose", target="mg/dL", plaus=(15, 1200), typical=(40, 600), cand=[1, 18.016]),
}

COUNT_ANALYTES = {"wbc", "anc", "alc", "amc", "plt"}


def _u(unit) -> str:
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return ""
    return re.sub(r"\s+", "", str(unit)).lower()


def count_factor(unit) -> float | None:
    """Factor converting a haematology count unit to 10^9/L; None if unknown; np.inf for percentages."""
    u = _u(unit)
    if u in ("", "."):
        return None
    if u == "%":
        return np.inf
    if "lakh" in u:
        return 100.0
    if re.search(r"10(\*\*|\*|\^|e)?5", u):
        return 100.0
    if re.search(r"10(\*\*|\*|\^|e)?4", u):
        return 10.0
    if re.search(r"10(\*\*|\*|\^|e)?6/l", u):
        return 0.001
    if re.search(r"10(\*\*|\*|\^|e)?2/", u):
        return 0.1
    if re.search(r"10(\*\*|\*|\^|e)?9", u) or "giga" in u or u in ("/nl", "gg/l"):
        return 1.0
    if (re.search(r"10(\*\*|\*|\^|e)?3", u) or u.startswith("k/") or "thou" in u
            or u.startswith("1000/") or u.startswith("t/")):
        return 1.0
    if u in ("/ul", "/mm3", "/mm**3", "/mm*3", "/cumm", "cells/ul", "cells/mcl", "1/mm**3", "cm**3", "/mcl"):
        return 0.001
    return None


# Explicit conversion tables for chemistry (lower-cased, spaces removed)
CHEM = {
    "hgb": {"g/dl": 1, "g%": 1, "gm%": 1, "g/100ml": 1, "g/l": 0.1, "mmol/l": 1.611},
    "alb": {"g/dl": 1, "g%": 1, "gm%": 1, "g/l": 0.1, "mg/dl": 0.001},
    "tprot": {"g/dl": 1, "g%": 1, "gm%": 1, "g/l": 0.1},
    "alp": {"u/l": 1, "iu/l": 1, "mu/ml": 1, "miu/ml": 1, "ukat/l": 60, "umol/l": 60},  # umol/l here = mislabelled ukat/L (values ~1-2)
    "ast": {"u/l": 1, "iu/l": 1, "mu/ml": 1, "ul": 1, "ukat/l": 60},
    "alt": {"u/l": 1, "iu/l": 1, "mu/ml": 1, "ul": 1, "ukat/l": 60},
    "bili": {"mg/dl": 1, "mg%": 1, "umol/l": 1 / 17.104, "mmol/l": 1 / 17.104, "mg/l": 0.1},  # mmol/l = mislabelled umol/L (values 5-20)
    "creat": {"mg/dl": 1, "mg%": 1, "umol/l": 1 / 88.42, "mmol/l": 1000 / 88.42, "mg/l": 0.1},
    "ca": {"mg/dl": 1, "mg%": 1, "mmol/l": 4.008, "meq/l": 2.004, "mg/l": 0.1},
    "na": {"mmol/l": 1, "meq/l": 1},
    "glu": {"mg/dl": 1, "mg%": 1, "mmol/l": 18.016, "g/l": 100},
}


def unit_factor(analyte: str, unit) -> float | None:
    if analyte in COUNT_ANALYTES:
        return count_factor(unit)
    return CHEM.get(analyte, {}).get(_u(unit))


def harmonise(df: pd.DataFrame, trial: str, analyte: str, value_col="value", unit_col="unit") -> pd.DataFrame:
    """Adds columns `val` (target unit) and `qc` (ok / rescued / implausible / unit_unknown / percent)."""
    spec = SPEC[analyte]
    lo, hi = spec["plaus"]
    tlo, thi = spec["typical"]
    out = df.copy()
    raw = pd.to_numeric(out[value_col], errors="coerce")
    fac = out[unit_col].map(lambda u: unit_factor(analyte, u))
    val = np.full(len(out), np.nan)
    qc = np.array(["missing"] * len(out), dtype=object)

    for i, (r, f) in enumerate(zip(raw.values, fac.values)):
        if np.isnan(r):
            continue
        if f is not None and np.isinf(f):
            qc[i] = "percent"  # differential in %, resolved by caller with WBC
            continue
        if f is not None:
            v = r * f
            if lo <= v <= hi:
                val[i], qc[i] = v, "ok"
                continue
        # magnitude rescue
        hits = [r * c for c in spec["cand"] if tlo <= r * c <= thi]
        if len(hits) == 1:
            val[i], qc[i] = hits[0], "rescued"
        else:
            qc[i] = "implausible" if f is not None else "unit_unknown"
    out["val"] = val
    out["qc"] = qc
    counts = pd.Series(qc).value_counts()
    for k in ("rescued", "implausible", "unit_unknown"):
        if counts.get(k, 0):
            AUDIT.log(trial, f"lab_{k}", analyte, counts[k],
                      "all records (any visit); rescued = unique candidate factor in typical range; "
                      "implausible/unit_unknown set to missing")
    return out


def uln_ratio(df: pd.DataFrame, value_col="value", uln_col="uln") -> pd.Series:
    """value / ULN using the record's own reference range (unit-independent)."""
    v = pd.to_numeric(df[value_col], errors="coerce")
    u = pd.to_numeric(df[uln_col], errors="coerce")
    r = v / u.where(u > 0)
    return r.where((r > 0) & (r < 100))
