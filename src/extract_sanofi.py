"""Extraction of the Sanofi XRP9881B-3001 / EFC6089 comparator arm (capecitabine), raw CRF-level SAS data.

Study days (xxDY) are relative to randomisation day 1 (RNDY = 1) -> re-expressed so that
randomisation = day 0. Deaths are recorded only at week precision (DEATHWK); death day is
set to the mid-point of the recorded week (7 x week - 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_labs import build_baseline
from common import AUDIT, RAW, TRIALS, last_in_window, read_sas
from drugs import ANTHRACYCLINES, CHEMO, ENDOCRINE, HER2_TARGETED, TAXANES, has
from endpoints import derive_pfs

D = RAW / "sanofi_135"
LAB_MAP = {"Hemoglobin": "hgb", "Leukocytes": "wbc", "Neutrophils": "anc", "Lymphocytes": "alc",
           "Monocytes": "amc", "Platelets": "plt", "Albumin": "alb", "Alkaline phosphatase": "alp",
           "SGOT": "ast", "SGPT": "alt", "Total bilirubin": "bili", "Creatinine": "creat",
           "Calcium": "ca", "Total protein": "tprot", "Sodium": "na", "Glucose (random)": "glu"}
VISCERAL = {"Liver", "Lung", "Pleura", "Pleural effusion", "Ascites", "Omentum/peritoneum", "Adrenal glands",
            "Brain", "Ovary", "Other visceral", "Kidneys", "Stomach", "Pancreas", "Heart", "Pericardium",
            "Gallbladder/biliary tract", "Retroperitoneum", "Diaphragm", "Thyroid"}
RESP_MAP = {"Complete response": "CR", "Partial response": "PR", "Stable disease": "SD",
            "Progressive disease": "PD", "Not evaluable": "NE"}


def _status(vals):
    v = set(x for x in vals if isinstance(x, str))
    if "Positive" in v:
        return 1.0
    if "Negative" in v:
        return 0.0
    return np.nan


def extract(trial="SAN135"):
    t = lambda n: read_sas(D / f"{n}.sas7bdat")
    rnd = t("random").set_index("RUSUBJID")["RNDY"]
    ids = rnd.index.tolist()
    rel = lambda df, col: df[col] - df["RUSUBJID"].map(rnd)

    pt = pd.DataFrame(index=pd.Index(ids, name="id"))
    dm = t("demo").set_index("RUSUBJID")
    pt["age"] = dm["AGE"]
    pt["sex"] = dm["SEX"].str[0]
    pt["race"] = dm["RACE"].str.upper()
    pt["country"] = dm["REGION"]
    hw = t("htwt")
    hw["day"] = rel(hw, "HTWTDY")
    hw0 = hw[hw.CYCLE == 0].drop_duplicates("RUSUBJID").set_index("RUSUBJID")
    pt["height_cm"] = hw0["HTDV"]
    pt["weight_kg"] = hw0["WTDV"]

    ps = t("perfstat")
    ps["day"] = rel(ps, "PSDY")
    ps.loc[ps.day.isna() & (ps.CYCLE == 0), "day"] = 0
    pt["ecog"] = last_in_window(ps.rename(columns={"RUSUBJID": "id"}), "id", "day", "PSSTAT").reindex(ids).round()

    # metastatic sites at baseline (investigator tumour assessments, cycle 0)
    ta = t("tumasses")
    ta["day"] = rel(ta, "TADY")
    tb = ta[ta.CYCLE == 0]
    sites = tb.groupby("RUSUBJID")["TASITECD"].apply(lambda s: set(s.dropna())).reindex(ids)
    f = lambda k: sites.map(lambda s: np.nan if not isinstance(s, set) else float(any(x in s for x in k)))
    pt["liver_mets"] = f({"Liver"})
    pt["lung_mets"] = f({"Lung", "Pleura", "Pleural effusion"})
    pt["bone_mets"] = f({"Bone", "Bone marrow"})
    pt["visceral_mets"] = f(VISCERAL)
    # organ-group count comparable to Pfizer/Lilly categories
    grp = {"Liver": "liver", "Lung": "lung", "Pleura": "pleura", "Pleural effusion": "pleura", "Lymph node": "lymph node",
           "Bone": "bone", "Bone marrow": "bone", "Skin": "skin", "Breast": "breast", "Brain": "brain",
           "Ascites": "peritoneum", "Omentum/peritoneum": "peritoneum", "Ovary": "ovary"}
    pt["n_met_sites"] = sites.map(lambda s: np.nan if not isinstance(s, set) or not s else len({grp.get(x, "other") for x in s}))
    pt["bone_only"] = sites.map(lambda s: np.nan if not isinstance(s, set) or not s else float(s <= {"Bone", "Bone marrow"}))
    pt["measurable"] = tb[tb.TAREMES == "Measurable (target)"].groupby("RUSUBJID").size().reindex(ids).notna().astype(float)
    pt.loc[sites.isna(), "measurable"] = np.nan

    # prior therapy
    ch = t("chemo")
    pc = ch[ch.CYCLE == 0].copy()
    pc["drug"] = pc["CTTERM"].fillna("") + " " + pc["CTMODIFY"].fillna("")
    g = pc.groupby("RUSUBJID")["drug"]
    pt["prior_taxane"] = g.apply(lambda s: float(any(has(x, TAXANES) for x in s))).reindex(ids).fillna(0)
    pt["prior_anthracycline"] = g.apply(lambda s: float(any(has(x, ANTHRACYCLINES) for x in s))).reindex(ids).fillna(0)
    adj = pc[pc.INTENT.isin(["Adjuvant", "Neo-adjuvant"]) & pc.drug.map(lambda x: has(x, CHEMO))]
    pt["prior_adj_chemo"] = pd.Series(1.0, index=adj.RUSUBJID.unique()).reindex(ids).fillna(0)
    met = pc[(pc.INTENT == "Metastatic") & pc.drug.map(lambda x: has(x, CHEMO))]
    pt["n_prior_chemo_adv"] = met.groupby("RUSUBJID")["REGNUM"].nunique().reindex(ids).fillna(0)
    tt = t("tumther")
    tp = tt[tt.CYCLE == 0].copy()
    tp["drug"] = tp["TTMODIFY"].fillna("") + " " + tp["TTREGDR"].fillna("")
    endo_ids = set(tp[(tp.TTTYP == "Hormonotherapy") | tp.drug.map(lambda x: has(x, ENDOCRINE))].RUSUBJID)
    pt["prior_endocrine"] = [1.0 if i in endo_ids else 0.0 for i in ids]
    her_ids = set(tp[tp.drug.map(lambda x: has(x, HER2_TARGETED))].RUSUBJID) | set(pc[pc.drug.map(lambda x: has(x, HER2_TARGETED))].RUSUBJID)
    pt["prior_her2"] = [1.0 if i in her_ids else 0.0 for i in ids]

    # receptors
    hr = t("hormrec")
    by = lambda typ: hr[hr.HRRECTYP.isin(typ)].groupby("RUSUBJID")["HRRECSTA"].apply(_status).reindex(ids)
    pt["er_pos"] = by(["Estrogen Receptors"])
    pt["pr_pos"] = by(["Progesterone Receptors"])
    pt["her2_pos"] = by(["HER-2 Receptors", "FISH", "Herceptest"])
    pt["tnbc"] = ((pt.er_pos == 0) & (pt.pr_pos == 0) & (pt.her2_pos == 0)).astype(float)
    pt.loc[pt[["er_pos", "pr_pos", "her2_pos"]].isna().any(axis=1), "tnbc"] = np.nan

    # laboratory
    lb = t("labs")
    lb = lb[lb.LBTEST.isin(LAB_MAP)].copy()
    lb["day"] = rel(lb, "LBDY")
    miss = lb.day.isna() & (lb.CYCLE == 0)
    lb.loc[miss, "day"] = 0
    AUDIT.log(trial, "lab_day_imputed", "all", miss.sum(), "screening (cycle 0) records without study day assigned day 0")
    long = pd.DataFrame({"id": lb.RUSUBJID, "day": lb.day, "analyte": lb.LBTEST.map(LAB_MAP),
                         "value": lb.ORGRES, "unit": lb.ORGUNIT, "uln": lb.ORGNRHI})
    pt = pt.join(build_baseline(long, trial, ids))

    # deaths (week precision)
    od = t("oncdeath")
    od = od[od.DEATHWK.notna()]
    deaths = (od.set_index("RUSUBJID")["DEATHWK"] * 7 - 3).groupby(level=0).min()
    AUDIT.log(trial, "death_day_from_week", "death", len(deaths), "death day = 7 x DEATHWK - 3 (mid-week)")

    # new anticancer therapy (follow-up pages 6xx)
    fu = ch[ch.CYCLE >= 600].copy()
    fu["day"] = rel(fu, "CTSDY")
    ft = tt[tt.CYCLE >= 600].copy()
    ft["day"] = rel(ft, "TTSDY")
    ft = ft[ft.TTTYP.isin(["Hormonotherapy", "Targeted therapy", "Immunotherapy"])]
    newtx = pd.concat([fu[["RUSUBJID", "day"]], ft[["RUSUBJID", "day"]]]).dropna()
    newtx = newtx[newtx.day > 1].groupby("RUSUBJID")["day"].min()

    # visit-level response: overall response per cycle, dated by that cycle's tumour-assessment day
    ov = t("ovrsp")
    aday = ta[ta.CYCLE > 0].groupby(["RUSUBJID", "CYCLE"])["day"].median()
    ov = ov[ov.OVRSP.notna()].join(aday, on=["RUSUBJID", "CYCLE"])
    # cycles without a dated tumour assessment: use the population median assessment day of that cycle
    cyc_day = ta[ta.CYCLE > 0].groupby("CYCLE")["day"].median()
    undated = ov["day"].isna()
    ov.loc[undated, "day"] = ov.loc[undated, "CYCLE"].map(cyc_day)
    AUDIT.log(trial, "response_day_imputed", "ovrsp", undated.sum(),
              "cycle-level response without patient-level assessment day: population median day of that cycle")
    resp = pd.DataFrame({"id": ov.RUSUBJID, "day": ov["day"], "resp": ov.OVRSP.map(RESP_MAP).fillna("NE")})
    # PD date recorded on the response page (investigator) overrides cycle-dated PD
    rp = t("response")
    pdd = rp[rp.RPDPDY.notna()].copy()
    pdd["day"] = rel(pdd, "RPDPDY")
    pd_first = pdd.groupby("RUSUBJID")["day"].min()
    # cycle-dated PD kept only for patients without a dated PD on the response page
    resp = resp[~((resp.resp == "PD") & resp.id.isin(pd_first.index))]
    resp = pd.concat([resp.dropna(subset=["day"]),
                      pd.DataFrame({"id": pd_first.index, "day": pd_first.values, "resp": "PD"})], ignore_index=True)
    AUDIT.log(trial, "response_undated", "ovrsp", ov["day"].isna().sum(), "responses still undated after imputation (dropped)")

    max_gap = 2 * TRIALS[trial]["assess_interval_days"] + 14
    AUDIT.log(trial, "pfs_rule", "max_gap_days", max_gap, "protocol interval 42 d")
    pt = pt.join(derive_pfs(ids, resp, deaths, newtx, max_gap))

    # overall survival
    pf = t("pafustat")
    pf["day"] = rel(pf, "PFDY")
    alive_days = [pf[pf.PFPTSTAT == "Alive"][["RUSUBJID", "day"]], ta[["RUSUBJID", "day"]],
                  lb[["RUSUBJID", "day"]], ps[["RUSUBJID", "day"]]]
    last = pd.concat(alive_days).dropna().groupby("RUSUBJID")["day"].max()
    pt["os_event"] = [1.0 if i in deaths.index else 0.0 for i in ids]
    pt["os_days"] = [deaths[i] if i in deaths.index else last.get(i, np.nan) for i in ids]
    pt["os_days"] = pt["os_days"].clip(lower=1)
    pt["deaths_known"] = pt["os_event"]
    return pt, resp
