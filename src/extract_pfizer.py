"""Extraction of the three Pfizer comparator-arm datasets (A6181107 / A6181094 / A6181099).

Pfizer day variables (COLLDAY, EFDAY, ...) are study days relative to first dose;
all days are re-expressed relative to randomisation (RANDDAY in random.sas7bdat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_labs import build_baseline
from common import AUDIT, RAW, TRIALS, last_in_window, read_sas
from drugs import ANTHRACYCLINES, CHEMO, ENDOCRINE, HER2_TARGETED, TAXANES, has
from endpoints import derive_pfs, median_gap

FOLDER = {"PFE111": "pfizer_111", "PFE112": "pfizer_112", "PFE113": "pfizer_113"}

LAB_MAP = {
    "HEMOGLOBIN": "hgb", "WHITE BLOOD CELLS": "wbc", "NEUTROPHILS (ABSOLUTE)": "anc",
    "LYMPHOCYTES (ABSOLUTE)": "alc", "MONOCYTES (ABSOLUTE)": "amc", "PLATELETS": "plt",
    "NEUTROPHILS (%)": "anc_pct", "LYMPHOCYTES (%)": "alc_pct", "MONOCYTES (%)": "amc_pct",
    "ALBUMIN": "alb", "ALKALINE PHOSPHATASE": "alp", "ASPARTATE AMINOTRANSFERASE (AST)": "ast",
    "ALANINE AMINOTRANSFERASE (ALT)": "alt", "BILIRUBIN (TOTAL)": "bili", "CREATININE": "creat",
    "CALCIUM": "ca", "PROTEIN (TOTAL)": "tprot", "SODIUM": "na", "GLUCOSE": "glu",
}
VISCERAL = {"LIVER", "LUNG", "PLEURA", "PLEURAL EFFUSION", "ASCITES", "BRAIN", "OVARY"}
RESP_MAP = {"COMPLETE RESPONSE": "CR", "PARTIAL RESPONSE": "PR", "STABLE DISEASE": "SD",
            "PROGRESSIVE DISEASE": "PD", "INDETERMINATE": "NE", "NOT ASSESSED": "NE"}


def extract(trial: str):
    d = RAW / FOLDER[trial]
    t = lambda name: read_sas(d / f"{name}.sas7bdat")
    rnd = t("random").set_index("PID_A")["RANDDAY"]
    ids = rnd.index.tolist()
    rel = lambda df, col: df[col] - df["PID_A"].map(rnd)

    pt = pd.DataFrame(index=pd.Index(ids, name="id"))

    # ---- demographics ------------------------------------------------------
    dm = t("demog").drop_duplicates("PID_A").set_index("PID_A")
    pt["age"] = dm["AGE"]
    pt["sex"] = dm["SEXC"].str.upper().str[0]
    pt["race"] = dm["RACESC"].str.upper()
    pt["height_cm"] = dm["HT"]
    pt["weight_kg"] = dm["WT"]
    reg = t("random").set_index("PID_A")["COUNTRY"]
    pt["country"] = reg

    # ---- ECOG --------------------------------------------------------------
    pf = t("pfm_p")
    pf["day"] = rel(pf, "EFDAY")
    pt["ecog"] = last_in_window(pf.rename(columns={"PID_A": "id"}), "id", "day", "PFMECOG").reindex(ids)
    pt["ecog"] = pt["ecog"].round()

    # ---- metastatic sites (investigator lesion log at baseline) -------------
    tm = t("tmm_p")
    tm["day"] = rel(tm, "EFDAY")
    tb = tm[(tm.day >= -42) & (tm.day <= 1)]
    # pleural effusion counted with pleura as one organ site (as in EFC6089)
    sites = tb.groupby("PID_A")["TMMDIS"].apply(lambda s: set(s.dropna().str.upper().replace({"PLEURAL EFFUSION": "PLEURA"})))
    sites = sites.reindex(ids)
    has_site = lambda k: sites.map(lambda s: np.nan if not isinstance(s, set) else float(any(x in s for x in k)))
    pt["liver_mets"] = has_site({"LIVER"})
    pt["lung_mets"] = has_site({"LUNG", "PLEURA", "PLEURAL EFFUSION"})
    pt["bone_mets"] = has_site({"BONE"})
    pt["visceral_mets"] = has_site(VISCERAL)
    pt["n_met_sites"] = sites.map(lambda s: np.nan if not isinstance(s, set) else len(s))
    pt["bone_only"] = sites.map(lambda s: np.nan if not isinstance(s, set) else float(s == {"BONE"}))
    pt["measurable"] = tb[tb.LESTYPE == "Target"].groupby("PID_A").size().reindex(ids).notna().astype(float)
    pt.loc[sites.isna(), "measurable"] = np.nan
    AUDIT.log(trial, "baseline_tumour_sites", "sites", sites.notna().sum(),
              "lesion log records within day -42..+1; visceral = liver/lung/pleura/effusion/ascites/brain/ovary")

    # ---- prior systemic therapy (screening history) -------------------------
    cd = t("cd_b_p")
    prior = cd[cd.CPEVENT.isin(["SCRN_SYSTEMIC", "SCREENING"])].copy()
    prior["drug"] = prior["CMDECOD"].fillna("")
    g = prior.groupby("PID_A")
    pt["prior_taxane"] = g["drug"].apply(lambda s: float(any(has(x, TAXANES) for x in s))).reindex(ids).fillna(0)
    pt["prior_anthracycline"] = g["drug"].apply(lambda s: float(any(has(x, ANTHRACYCLINES) for x in s))).reindex(ids).fillna(0)
    pt["prior_endocrine"] = g["drug"].apply(lambda s: float(any(has(x, ENDOCRINE) for x in s))).reindex(ids).fillna(0)
    pt["prior_her2"] = g["drug"].apply(lambda s: float(any(has(x, HER2_TARGETED) for x in s))).reindex(ids).fillna(0)
    adj = prior[prior.ONCCTYP.isin(["ADJUVANT", "NEOADJUVANT"]) & prior.drug.map(lambda x: has(x, CHEMO))]
    pt["prior_adj_chemo"] = pd.Series(1.0, index=adj.PID_A.unique()).reindex(ids).fillna(0)
    met = prior[(prior.ONCCTYP == "ADVANCED/METASTATIC") & prior.drug.map(lambda x: has(x, CHEMO))]
    pt["n_prior_chemo_adv"] = met.groupby("PID_A")["CDRGRNM"].nunique().reindex(ids).fillna(0)

    # ---- laboratory --------------------------------------------------------
    lb = t("lab_safe")
    lb = lb[lb.LBTEST.isin(LAB_MAP)].copy()
    long = pd.DataFrame({"id": lb.PID_A, "day": rel(lb, "COLLDAY"), "analyte": lb.LBTEST.map(LAB_MAP),
                         "value": lb.LABVALUE, "unit": lb.LABUNITR, "uln": lb.MAX_NORM})
    labs = build_baseline(long, trial, ids)
    pt = pt.join(labs)

    # ---- deaths -------------------------------------------------------------
    ae = t("adverse")
    g5 = ae[ae.AEGRADE == 5].copy()
    g5["dday"] = rel(g5, "AETDAY").fillna(rel(g5, "AEFDAY"))
    fin = t("final")
    fd = fin[fin.FINSTATC == "SUBJECT DIED"].copy()
    fd["dday"] = rel(fd, "COLLDAY")
    death_src = [g5[["PID_A", "dday"]], fd[["PID_A", "dday"]]]
    srv = None
    if (d / "srv_p.sas7bdat").exists():
        srv = t("srv_p")
        s_d = srv[srv.DEATHDAY.notna()].copy()
        s_d["dday"] = rel(s_d, "DEATHDAY")
        death_src.append(s_d[["PID_A", "dday"]])
    deaths = pd.concat(death_src).dropna().groupby("PID_A")["dday"].min()

    # ---- new anticancer therapy after randomisation -------------------------
    post = cd[cd.CPEVENT.isin(["FOLLOW_UP_SYSTEM", "LOG_SYSTEMIC"])].copy()
    post["day"] = rel(post, "CDBPFDAY")
    post = post[post.CMDECOD.map(lambda x: has(x, CHEMO + ENDOCRINE + HER2_TARGETED + ["BEVACIZUMAB", "LAPATINIB", "SUNITINIB"]))]
    post = post[post.day > 1]
    newtx = post.groupby("PID_A")["day"].min()

    # ---- tumour response & PFS ---------------------------------------------
    io = t("iota_p")
    resp = pd.DataFrame({"id": io.PID_A, "day": rel(io, "EFDAY"), "resp": io.IOTAALL.map(RESP_MAP).fillna("NE")})
    interval = TRIALS[trial]["assess_interval_days"]
    max_gap = 2 * interval + 14
    AUDIT.log(trial, "pfs_rule", "max_gap_days", max_gap,
              f"protocol assessment interval {interval} d (observed median {median_gap(resp):.0f} d); "
              "PD/death > 2 x interval + 14 d after last adequate assessment censored")
    pfs = derive_pfs(ids, resp, deaths, newtx, max_gap)
    pt = pt.join(pfs)

    # ---- overall survival (only where long-term follow-up exists) ----------
    if srv is not None:
        # last known alive: latest day in any patient-level record
        days = [rel(srv, "SRVDAY"), rel(lb, "COLLDAY"), rel(io, "EFDAY"), rel(fin, "COLLDAY"), rel(pf, "EFDAY")]
        idcols = [srv.PID_A, lb.PID_A, io.PID_A, fin.PID_A, pf.PID_A]
        last = pd.concat([pd.DataFrame({"id": i, "day": dd}) for i, dd in zip(idcols, days)]).groupby("id")["day"].max()
        pt["os_event"] = pt.index.map(lambda i: 1.0 if i in deaths.index else 0.0)
        pt["os_days"] = [deaths[i] if i in deaths.index else last.get(i, np.nan) for i in pt.index]
        pt["os_days"] = pt["os_days"].clip(lower=1)
    else:
        pt["os_event"] = np.nan
        pt["os_days"] = np.nan

    pt["deaths_known"] = pt.index.map(lambda i: 1.0 if i in deaths.index else 0.0)
    return pt, resp
