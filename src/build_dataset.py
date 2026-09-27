"""Runs all extractors and writes the pooled, harmonised, one-row-per-patient analysis dataset.

Outputs
  data/processed/mbc_pooled.csv              analysis dataset (all trials)
  data/processed/mbc_pooled.xlsx             same + data dictionary + cleaning log + flow sheets
  data/interim/tumour_response_long.csv      visit-level responses used for PFS derivation
  outputs/tables/cleaning_log.csv            audit log of every cleaning action
  outputs/tables/patient_flow.csv            CONSORT-style counts
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import extract_lilly
import extract_pfizer
import extract_sanofi
from baseline_labs import add_inflammation_indices
from common import AUDIT, DAYS_PER_MONTH, INTERIM, OUT_TAB, PROCESSED, TRIALS

ORDER = ["PFE111", "PFE113", "SAN135", "PFE112", "LLY168"]

NONLAB_PLAUS = {"age": (18, 100), "height_cm": (120, 210), "weight_kg": (30, 200), "bmi": (12, 70), "ecog": (0, 2)}

DICTIONARY = [
    # name, label, unit/coding, source note
    ("trial", "Trial identifier", "PFE111/PFE112/PFE113/LLY168/SAN135", "PDS dataset"),
    ("nct", "ClinicalTrials.gov identifier", "", "registry"),
    ("sponsor", "Data provider / sponsor", "", "PDS"),
    ("setting", "Treatment setting of the trial", "first-line / pretreated", "protocol"),
    ("control_regimen", "Comparator-arm regimen received", "", "protocol"),
    ("patient_id", "De-identified patient id (trial-prefixed)", "", "PDS"),
    ("age", "Age at baseline", "years", "demographics"),
    ("race", "Race (harmonised)", "WHITE/BLACK/ASIAN/OTHER", "demographics"),
    ("height_cm", "Height", "cm", "screening"),
    ("weight_kg", "Weight", "kg", "screening"),
    ("bmi", "Body-mass index", "kg/m2", "derived weight/height^2"),
    ("ecog", "ECOG performance status at baseline", "0/1/2", "last value day -35..+1"),
    ("visceral_mets", "Visceral metastasis at baseline", "1=yes", "lesion log (LLY168: sponsor stratum)"),
    ("liver_mets", "Liver metastasis", "1=yes", "lesion log; NA in LLY168"),
    ("lung_mets", "Lung/pleural metastasis", "1=yes", "lesion log; NA in LLY168"),
    ("bone_mets", "Bone metastasis", "1=yes", "lesion log; NA in LLY168"),
    ("bone_only", "Bone-only disease", "1=yes", "lesion log; NA in LLY168"),
    ("n_met_sites", "Number of involved organ sites", "count", "lesion log (LLY168: sponsor NBST)"),
    ("measurable", "Measurable disease (>=1 target lesion)", "1=yes", "lesion log"),
    ("prior_taxane", "Any prior taxane", "1=yes", "prior therapy history"),
    ("prior_anthracycline", "Any prior anthracycline", "1=yes", "prior therapy history"),
    ("prior_endocrine", "Any prior endocrine therapy (proxy for HR+)", "1=yes", "prior therapy history"),
    ("prior_her2", "Any prior HER2-targeted therapy", "1=yes", "prior therapy history"),
    ("prior_adj_chemo", "Prior (neo)adjuvant chemotherapy", "1=yes", "prior therapy history"),
    ("n_prior_chemo_adv", "Number of prior chemotherapy regimens for advanced disease", "count", "prior therapy history"),
    ("er_pos", "Oestrogen-receptor positive", "1=yes", "LLY168 & SAN135 only"),
    ("pr_pos", "Progesterone-receptor positive", "1=yes", "LLY168 & SAN135 only"),
    ("her2_pos", "HER2 positive", "1=yes", "LLY168 & SAN135 only"),
    ("tnbc", "Triple-negative", "1=yes", "LLY168 & SAN135 only"),
    ("hgb", "Haemoglobin", "g/dL", "baseline lab"), ("wbc", "White blood cells", "10^9/L", "baseline lab"),
    ("anc", "Absolute neutrophil count", "10^9/L", "baseline lab (or % x WBC)"),
    ("alc", "Absolute lymphocyte count", "10^9/L", "baseline lab (or % x WBC); NA in LLY168"),
    ("amc", "Absolute monocyte count", "10^9/L", "baseline lab (or % x WBC); NA in LLY168"),
    ("plt", "Platelets", "10^9/L", "baseline lab"), ("alb", "Albumin", "g/dL", "baseline lab"),
    ("alp", "Alkaline phosphatase", "U/L", "baseline lab"), ("ast", "AST", "U/L", "baseline lab"),
    ("alt", "ALT", "U/L", "baseline lab"), ("bili", "Total bilirubin", "mg/dL", "baseline lab"),
    ("creat", "Creatinine", "mg/dL", "baseline lab"), ("ca", "Calcium (total)", "mg/dL", "baseline lab"),
    ("tprot", "Total protein", "g/dL", "baseline lab"), ("na", "Sodium", "mmol/L", "baseline lab"),
    ("glu", "Glucose (non-fasting)", "mg/dL", "baseline lab"),
    ("alp_uln", "ALP / upper limit of normal", "ratio", "record-level reference range"),
    ("ast_uln", "AST / ULN", "ratio", "record-level reference range"),
    ("alt_uln", "ALT / ULN", "ratio", "record-level reference range"),
    ("bili_uln", "Bilirubin / ULN", "ratio", "record-level reference range"),
    ("creat_uln", "Creatinine / ULN", "ratio", "record-level reference range"),
    ("nlr", "Neutrophil-to-lymphocyte ratio", "ratio", "anc/alc"),
    ("plr", "Platelet-to-lymphocyte ratio", "ratio", "plt/alc"),
    ("sii", "Systemic immune-inflammation index", "10^9/L", "plt*anc/alc"),
    ("lmr", "Lymphocyte-to-monocyte ratio", "ratio", "alc/amc"),
    ("pfs_months", "Progression-free survival time", "months", "harmonised algorithm (LLY168: sponsor ADTTE)"),
    ("pfs_event", "PFS event indicator", "1=PD or death", ""),
    ("pfs_reason", "Reason for PFS event/censoring", "text", ""),
    ("os_months", "Overall survival time", "months", "PFE111, SAN135, LLY168 only"),
    ("os_event", "Death indicator", "1=death", "PFE111, SAN135, LLY168 only"),
    ("pfs_months_alg", "PFS by harmonised algorithm (validation, LLY168 only)", "months", ""),
    ("pfs_event_alg", "PFS event by harmonised algorithm (validation, LLY168 only)", "1=event", ""),
]


def harmonise_race(r):
    if pd.isna(r):
        return np.nan
    r = str(r).upper()
    for k in ("WHITE", "BLACK", "ASIAN"):
        if k in r:
            return k
    return "OTHER"


def run():
    frames, resp_all, flow = [], [], []
    extractors = {"PFE111": extract_pfizer.extract, "PFE112": extract_pfizer.extract, "PFE113": extract_pfizer.extract,
                  "LLY168": extract_lilly.extract, "SAN135": extract_sanofi.extract}
    for tr in ORDER:
        pt, resp = extractors[tr](tr)
        n0 = len(pt)
        pt = pt.reset_index().rename(columns={"id": "src_id"})
        pt.insert(0, "trial", tr)
        for k in ("nct", "sponsor", "setting"):
            pt[k] = TRIALS[tr][k]
        pt["control_regimen"] = TRIALS[tr]["control"]
        pt["patient_id"] = tr + "-" + pt["src_id"].astype(str).str.replace(r"\s+", "", regex=True)
        resp = resp.assign(trial=tr, patient_id=tr + "-" + resp["id"].astype(str).str.replace(r"\s+", "", regex=True))
        resp_all.append(resp.drop(columns="id"))

        # exclusions ---------------------------------------------------------
        male = pt["sex"].eq("M")
        AUDIT.log(tr, "exclusion", "sex", male.sum(), "male patients excluded (female MBC population)")
        pt = pt[~male]
        flow.append(dict(trial=tr, comparator_arm_in_PDS=n0, excluded_male=int(male.sum()), analysed=len(pt)))
        frames.append(pt)

    df = pd.concat(frames, ignore_index=True)
    df["race"] = df["race"].map(harmonise_race)
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    # non-laboratory plausibility --------------------------------------------
    for c, (lo, hi) in NONLAB_PLAUS.items():
        bad = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
        for tr, n in df.loc[bad, "trial"].value_counts().items():
            AUDIT.log(tr, "implausible_set_missing", c, n, f"outside [{lo},{hi}]")
        df.loc[bad, c] = np.nan
    # BMI missing if height/weight implausible
    df.loc[df["height_cm"].isna() | df["weight_kg"].isna(), "bmi"] = np.nan

    df = add_inflammation_indices(df)

    # endpoints in months
    df["pfs_months"] = df["pfs_days"] / DAYS_PER_MONTH
    df["os_months"] = df["os_days"] / DAYS_PER_MONTH
    if "pfs_days_alg" in df:
        df["pfs_months_alg"] = df["pfs_days_alg"] / DAYS_PER_MONTH

    # duplicates / key integrity
    dup = df.patient_id.duplicated().sum()
    assert dup == 0, f"{dup} duplicated patient ids"

    cols = [c for c, *_ in DICTIONARY if c in df.columns]
    out = df[cols].copy()
    out.to_csv(PROCESSED / "mbc_pooled.csv", index=False)
    pd.concat(resp_all).to_csv(INTERIM / "tumour_response_long.csv", index=False)

    log = AUDIT.frame()
    log.to_csv(OUT_TAB / "cleaning_log.csv", index=False)
    flow = pd.DataFrame(flow)
    flow.to_csv(OUT_TAB / "patient_flow.csv", index=False)
    dd = pd.DataFrame(DICTIONARY, columns=["variable", "label", "unit_or_coding", "source_or_derivation"])
    dd["n_non_missing"] = dd.variable.map(lambda v: int(out[v].notna().sum()) if v in out else 0)
    dd["pct_missing"] = dd.variable.map(lambda v: round(100 * out[v].isna().mean(), 1) if v in out else np.nan)
    dd.to_csv(PROCESSED / "data_dictionary.csv", index=False)
    with pd.ExcelWriter(PROCESSED / "mbc_pooled.xlsx") as xw:
        out.to_excel(xw, sheet_name="data", index=False)
        dd.to_excel(xw, sheet_name="dictionary", index=False)
        log.to_excel(xw, sheet_name="cleaning_log", index=False)
        flow.to_excel(xw, sheet_name="patient_flow", index=False)
    return out, log, flow


if __name__ == "__main__":
    out, log, flow = run()
    print(out.shape)
    print(flow.to_string(index=False))
