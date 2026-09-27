"""Extraction of the Eli Lilly ROSE/TRIO-12 comparator arm (CP12-0606; placebo + docetaxel), CDISC ADaM.

Primary endpoints are taken from the sponsor-derived ADTTE (investigator PFS, OS).
The harmonised PFS algorithm (endpoints.derive_pfs) is additionally applied to
visit-level responses (ADRS PARAMCD=ORINV) to validate the algorithm used for
the Pfizer/Sanofi trials against a sponsor derivation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_labs import build_baseline
from common import AUDIT, DAYS_PER_MONTH, RAW, TRIALS, read_sas, yn
from drugs import HER2_TARGETED, has
from endpoints import derive_pfs

D = RAW / "lilly_168"
LAB_MAP = {"Hemoglobin": "hgb", "Leukocytes": "wbc", "Neutrophils": "anc", "Platelet": "plt",
           "Albumin": "alb", "Alkaline Phosphatase": "alp", "Aspartate Aminotransferase": "ast",
           "Alanine Aminotransferase": "alt", "Bilirubin": "bili", "Creatinine": "creat",
           "Calcium": "ca", "Protein": "tprot", "Sodium": "na", "Glucose": "glu"}
RESP_MAP = {"CR": "CR", "PR": "PR", "SD": "SD", "PD": "PD", "UE": "NE", "NE": "NE"}


def extract(trial="LLY168"):
    t = lambda n: read_sas(D / f"{n}.sas7bdat")
    sl = t("adsl").set_index("USUBJID")
    ids = sl.index.tolist()
    rdt = sl["RANDDT"]
    rel = lambda idcol, dates: (pd.to_datetime(dates) - idcol.map(rdt)).dt.days

    pt = pd.DataFrame(index=pd.Index(ids, name="id"))
    pt["age"] = sl["AGEIC"]
    pt["sex"] = sl["SEX"].str[0]
    pt["race"] = sl["RACE"].replace({"BLACK OR AFRICAN AMERICAN": "BLACK",
                                     "AMERICAN INDIAN OR ALASKA NATIVE": "OTHER"})
    pt["height_cm"] = sl["HGTBLR"]
    pt["weight_kg"] = sl["WGTBLR"]
    pt["country"] = sl["COUNTRY"]
    pt["ecog"] = sl["ECOGBLR"]

    # metastatic disease (sponsor-derived; organ-level flags not shared in ADaM)
    pt["visceral_mets"] = sl["VMETA"].map(yn)
    pt["n_met_sites"] = sl["NBST"]
    pt["measurable"] = (sl["LESTYP"] == "MEASURABLE").astype(float)
    for c in ("liver_mets", "lung_mets", "bone_mets", "bone_only"):
        pt[c] = np.nan
    AUDIT.log(trial, "not_available", "liver/lung/bone mets, bone_only", len(ids),
              "organ-level lesion data not included in shared ADaM; visceral flag and site count used")

    # prior therapy
    pt["prior_taxane"] = sl["PTAX"].map(yn)
    pt["prior_anthracycline"] = sl["PAFL"].map(yn)
    pt["prior_endocrine"] = sl["PHTXFL"].map(yn)
    cm = t("adcm")
    prior = cm[cm.APHASE == "Prior to Randomization/First Dose"]
    adj = prior[(prior.TXTYP == "Chemotherapy") & prior.CMINDALL.isin(["ADJUVANT", "NEOADJUVANT"])]
    pt["prior_adj_chemo"] = pd.Series(1.0, index=adj.USUBJID.unique()).reindex(ids).fillna(0)
    pt["prior_her2"] = prior.groupby("USUBJID").CMDECOD.apply(
        lambda s: float(any(has(x, HER2_TARGETED) for x in s.fillna("")))).reindex(ids).fillna(0)
    pt["n_prior_chemo_adv"] = 0.0  # eligibility: no prior chemotherapy for metastatic disease
    AUDIT.log(trial, "by_design", "n_prior_chemo_adv", len(ids), "first-line trial: no prior chemotherapy for MBC (eligibility)")

    # receptor status (available only in LLY168 and SAN135)
    pt["er_pos"] = sl["ERST"].map(yn)
    pt["pr_pos"] = sl["PRST"].map(yn)
    pt["her2_pos"] = sl["HERST"].map(lambda x: np.nan if pd.isna(x) else float(str(x).startswith("AMPLIFIED")))
    pt["tnbc"] = sl["TRPNEGFL"].map(yn)

    # laboratory (serum/blood only)
    lb = t("adlb")
    lb = lb[lb.PARAM.isin(LAB_MAP) & ~lb.LBSPEC.fillna("").str.upper().str.contains("URINE")]
    long = pd.DataFrame({"id": lb.USUBJID, "day": rel(lb.USUBJID, lb.ADT), "analyte": lb.PARAM.map(LAB_MAP),
                         "value": lb.AVAL, "unit": lb.LBSTRESU, "uln": lb.ANRHI})
    pt = pt.join(build_baseline(long, trial, ids))

    # sponsor-derived endpoints (months -> days)
    tte = t("adtte")
    tte = tte[tte.PARAMCD == "INV"]
    pfs = tte[tte.PARCAT1.str.startswith("Progression-free")].set_index("USUBJID")
    os_ = tte[tte.PARCAT1.str.startswith("Overall survival")].set_index("USUBJID")
    pt["pfs_days"] = (pfs["AVAL"] * DAYS_PER_MONTH).reindex(ids).clip(lower=1)
    pt["pfs_event"] = (1 - pfs["CNSR"]).reindex(ids)
    pt["pfs_reason"] = np.where(pt.pfs_event == 1, "event (sponsor ADTTE)",
                                "censored: " + pfs["CNSRDSC"].reindex(ids).fillna("sponsor").astype(str))
    pt["os_days"] = (os_["AVAL"] * DAYS_PER_MONTH).reindex(ids).clip(lower=1)
    pt["os_event"] = (1 - os_["CNSR"]).reindex(ids)
    pt["deaths_known"] = sl["DEATHFL"].map(yn).fillna(0)

    # --- validation of harmonised algorithm on ADRS --------------------------
    rs = t("adrs")
    rs = rs[(rs.PARAMCD == "ORINV") & rs.ADT.notna()]
    resp = pd.DataFrame({"id": rs.USUBJID, "day": rel(rs.USUBJID, rs.ADT), "resp": rs.AVALC.map(RESP_MAP).fillna("NE")})
    deaths = rel(sl.index.to_series(), sl["DEATHDT"]).dropna()
    post = cm[(cm.APHASE == "Post Discontinuation of Treatment") &
              cm.TXTYP.isin(["Chemotherapy", "Hormonal", "Small Molecule", "Antiangiogenic", "Biologic", "Biologics", "Investigational"])]
    newtx = rel(post.USUBJID, post.CMSTDT).groupby(post.USUBJID).min()
    newtx = newtx[newtx > 1]
    max_gap = 2 * TRIALS[trial]["assess_interval_days"] + 14
    alg = derive_pfs(ids, resp, deaths, newtx, max_gap)
    pt["pfs_days_alg"] = alg["pfs_days"]
    pt["pfs_event_alg"] = alg["pfs_event"]
    return pt, resp
