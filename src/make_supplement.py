"""Builds Supplementary File S1 (Tables S1-S9) as Excel and as JSON for the Word version (build_supplement_docx.js)."""
import json
import re

import numpy as np
import pandas as pd

from common import OUT_TAB, ROOT

OUT = ROOT / "outputs" / "manuscript"
LAB = {"M0": "M0 Points score", "M1": "M1 Cox, clinical", "M2": "M2 Cox, clinical + blood", "M3": "M3 LASSO-Cox",
       "M4": "M4 Random survival forest", "M5": "M5 Gradient-boosted Cox", "M6": "M6 DeepSurv"}
MET = {"harrell_c": "Harrell C", "uno_c": "Uno C (τ=12 mo)", "auc_6": "AUC 6 mo", "auc_12": "AUC 12 mo",
       "ibs_12": "IBS 1–12 mo", "cal_slope": "Calibration slope", "oe_6": "O/E 6 mo"}
rd = lambda f: pd.read_csv(OUT_TAB / f)


def f3(x):
    return "" if pd.isna(x) else f"{x:.3f}"


def per_trial(outcome):
    p = rd(f"iecv_per_trial_{outcome}.csv")
    p["value"] = [("NA" if pd.isna(e) else f"{e:.3f} ({l:.3f}–{h:.3f})") for e, l, h in zip(p.estimate, p.boot_lo, p.boot_hi)]
    t = p.pivot_table(index=["model", "metric"], columns="trial", values="value", aggfunc="first").reset_index()
    t["model"] = t.model.map(LAB); t["metric"] = t.metric.map(MET)
    return t.rename(columns={"model": "Model", "metric": "Metric"})


def pooled(outcome):
    p = rd(f"iecv_pooled_{outcome}.csv")
    return pd.DataFrame({"Model": p.model.map(LAB), "Metric": p.metric.map(MET), "Trials (k)": p.k,
                         "Pooled (95% CI)": [f"{a:.3f} ({b:.3f}–{c:.3f})" for a, b, c in zip(p.pooled, p.lo, p.hi)],
                         "I²": [f"{100 * i:.0f}%" for i in p.i2],
                         "95% PI": ["NA" if (pd.isna(a) or k < 4) else f"{a:.3f}–{b:.3f}" for a, b, k in zip(p.pi_lo, p.pi_hi, p.k)]})


def cdiff(outcome):
    d = rd(f"iecv_cdiff_pooled_{outcome}.csv")
    return pd.DataFrame({"Model": d.model.map(LAB), "Reference": d.reference.map(LAB),
                         "ΔC pooled (95% CI)": [f"{a:+.3f} ({b:+.3f} to {c:+.3f})".replace("-", "−") for a, b, c in zip(d.pooled, d.lo, d.hi)],
                         "I²": [f"{100 * i:.0f}%" for i in d.i2]})


NICE = {"Unnamed: 0": "Predictor", "pooled_c": "Pooled C", "lo": "95% CI low", "hi": "95% CI high", "i2": "I²",
        "harrell_c": "Harrell C", "analysis": "Analysis", "model": "Model", "setting": "Setting", "k": "Trials (k)",
        "grid": "Grid", "trial": "Trial", "estimate": "Estimate", "HR_per_SD": "HR per SD", "p": "p",
        "SD_original_scale": "SD (original scale)", "scale": "Scale", "test_statistic": "Test statistic",
        "-log2(p)": "−log2(p)", "item": "Item", "log_HR": "log HR", "points": "Points", "HR": "HR",
        "n": "n", "events": "Events", "median_pfs": "Median PFS (months)", "pfs6": "6-month PFS", "pfs12": "12-month PFS",
        "exp(coef)": "HR", "exp(coef) lower 95%": "95% CI low", "exp(coef) upper 95%": "95% CI high",
        "expected_shrinkage_vanhouwelingen": "Heuristic shrinkage of fitted model (van Houwelingen)",
        "lr_chi2": "LR χ²", "df": "df", "c_clin_strat": "C clinical (stratified)", "c_full_strat": "C clinical+blood (stratified)",
        "c_final_apparent": "C final model (apparent)", "variable": "Variable", "coef_original_scale": "Coefficient (original scale)",
        "HR_original_scale": "HR (original scale)", "mean_abs_shap": "Mean |SHAP|", "feature": "Feature",
        "trial ": "Trial", "step": "Step", "n_affected": "n affected", "detail": "Detail"}


NICE.update({"R2_CS_apparent": "Cox–Snell R² (apparent)", "R2_CS_adjusted": "Cox–Snell R² (optimism-adjusted)",
             "required_n_criterion_i": "Required n, criterion (i) (adjusted R²)", "required_n_criterion_ii": "Required n, criterion (ii) (adjusted R²)",
             "required_n_criterion_i_apparentR2": "Required n, criterion (i) (apparent R²)",
             "required_n_criterion_ii_apparentR2": "Required n, criterion (ii) (apparent R²)",
             "events_per_parameter": "Events per parameter", "parameters": "Parameters", "LR_chi2": "LR χ² (final model)",
             "n_test": "n (held-out)", "se": "SE", "pfs_events": "PFS events", "derived_median_pfs": "Derived median PFS",
             "published_median_pfs": "Published median PFS", "derived_median_os": "Derived median OS",
             "published_median_os": "Published median OS", "pfs_followup_reverse_km": "PFS follow-up (reverse KM, months)",
             "reference": "Reference", "note": "Note", "high": "High", "inter": "Intermediate"})


def _num(v, k):
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return str(int(round(v))) if float(v).is_integer() and abs(v) >= 1 else f"{v:.{k}f}".replace("-", "−")
    if isinstance(v, str):
        return re.sub(r"(?<=\d)-(?=\d)", "–", v)
    return v


def tidy(df, k=3):
    df = df.copy()
    for c in df.columns:
        lc = str(c)
        if lc in ("p",) or lc.endswith("_p") or lc == "p_value":
            df[c] = [("<0.001" if (isinstance(v, (int, float)) and v < 0.001) else (f"{v:.3f}" if isinstance(v, (int, float)) else v)) for v in df[c]]
        elif lc in ("i2", "I²") and df[c].dtype.kind == "f":
            df[c] = [f"{100 * v:.0f}%" if pd.notna(v) else "" for v in df[c]]
        else:
            df[c] = [_num(v, k) for v in df[c]]
    if df.shape[1] and str(df.columns[0]) in ("Group", "covariate"):
        df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: NICE.get(x, x))
    if "Quantity" in df.columns:
        df["Quantity"] = df["Quantity"].map(lambda x: NICE.get(x, x))
    return df.rename(columns=lambda c: NICE.get(str(c), str(c)).replace("_", " "))


def rnd(df, k=3):
    return df  # formatting is done in tidy()


def build():
    ex = json.load(open(OUT_TAB / "extra_analyses_summary.json"))
    sub = json.load(open(OUT_TAB / "subanalyses_summary.json"))
    m = json.load(open(OUT_TAB / "final_model_M2.json"))
    hr = rd("final_cox_hr_per_sd.csv").rename(columns={"Unnamed: 0": "Predictor"})
    bs = m["baseline_survival"]
    bs = pd.DataFrame({"Months": list(bs.keys()), "Baseline S0(t)": list(bs.values())}) if isinstance(bs, dict) else pd.DataFrame({"S0": [str(bs)]})
    prep = pd.DataFrame({"Feature": m["features"], "Log-transformed": ["yes" if f in m["log_vars"] else "no" for f in m["features"]],
                         "Winsor low": m["lo"], "Winsor high": m["hi"], "Mean": m["mu"], "SD": m["sd"],
                         "Coefficient (standardised)": m["coef"]}) if isinstance(m["mu"], list) else \
        pd.DataFrame({"Feature": list(m["mu"].keys()), "Log-transformed": ["yes" if f in m["log_vars"] else "no" for f in m["mu"]],
                      "Winsor low": [m["lo"][f] for f in m["mu"]], "Winsor high": [m["hi"][f] for f in m["mu"]],
                      "Mean": list(m["mu"].values()), "SD": [m["sd"][f] for f in m["mu"]],
                      "Coefficient (standardised)": [m["coef"][f] for f in m["mu"]]})
    rec = sub["receptor"]
    tables = [
        ("Table S1", "Data-cleaning audit log and patient flow",
         [("Patient flow", rd("patient_flow.csv")), ("Audit log (one row per trial × cleaning step × variable)", rd("cleaning_log.csv"))]),
        ("Table S2", "Baseline characteristics, missingness and endpoint validation",
         [("Baseline characteristics by trial", rd("table1_baseline.csv")), ("Missing values (%) by trial", rd("missingness_by_trial.csv")),
          ("Endpoint validation against published comparator-arm results", rd("endpoint_validation.csv"))]),
        ("Table S3", "Per-trial performance in held-out trials, estimate (bootstrap 95% CI)",
         [("Progression-free survival", per_trial("pfs")), ("Overall survival", per_trial("os"))]),
        ("Table S4", "Random-effects pooled performance and paired C-index differences",
         [("PFS, pooled performance", pooled("pfs")), ("PFS, paired ΔC", cdiff("pfs")),
          ("OS, pooled performance", pooled("os")), ("OS, paired ΔC", cdiff("os"))]),
        ("Table S5", "Final model (M2) fitted to all 1298 patients and final points score",
         [("Hazard ratios per SD", rnd(hr)), ("Coefficients on the original scale", rnd(rd("final_model_M2_coefficients_original_scale.csv"), 5)),
          ("Preprocessing constants and standardised coefficients (calculator equation: S(t|x) = S0(t)^exp(lp), lp = Σ coef·(clip(x) − mean)/SD, x log-transformed where indicated)", rnd(prep, 5)),
          ("Baseline survival", rnd(bs, 5)), ("Points score", rnd(rd("final_points_score.csv")))]),
        ("Table S6", "Missing-data robustness (pooled Harrell C)",
         [("Pooled (pooling of analysis-specific bootstrap SEs; I² may differ by 1 point from Table S4)", rnd(rd("missing_robustness_pooled.csv"))), ("Per trial", rnd(rd("missing_robustness_per_trial.csv")))]),
        ("Table S7", "Lymphocyte/NLR, laboratory-block and receptor-status analyses",
         [("Lymphocytes not recorded in ROSE/TRIO-12 (LLY168)", rnd(rd("lymphocyte_scenario_LLY168.csv"))),
          ("NLR added value in the four trials with lymphocytes", rnd(rd("nlr_added_value_4trials.csv"))),
          ("Likelihood-ratio test of the laboratory block (trial-stratified Cox, all data)", pd.DataFrame([{"LR χ²": round(sub["lrt_blood"]["lr_chi2"], 1), "df": int(sub["lrt_blood"]["df"]), "p": sub["lrt_blood"]["p"], "C clinical (stratified)": sub["lrt_blood"]["c_clin_strat"], "C clinical+blood (stratified)": sub["lrt_blood"]["c_full_strat"], "C final model (apparent)": sub["lrt_blood"]["c_final_apparent"]}])),
          ("ER status (LLY168 + SAN135)", pd.DataFrame([{"n": rec["n"], "LR χ²": round(rec["lr_chi2"], 2), "p": "<0.001" if rec["p"] < 0.001 else f"{rec['p']:.3f}",
                                                          "HR ER+ (95% CI)": f"{rec['hr_er'][0]:.2f} ({rec['hr_er'][1]:.2f}–{rec['hr_er'][2]:.2f})",
                                                          "C without ER": round(rec["c_without"], 3), "C with ER": round(rec["c_with"], 3)}])),
          ("Prior endocrine therapy (rows) vs ER status (columns); 0 = no/negative, 1 = yes/positive", rd("prior_endocrine_vs_er_crosstab.csv"))]),
        ("Table S8", "Tuning sensitivity, non-linearity, proportional hazards and SHAP importance",
         [("Hyper-parameter grid sensitivity (pooled Harrell C)", rnd(rd("tuning_sensitivity_pooled.csv"))),
          ("Cubic B-spline Cox vs linear Cox (analysis-specific pooling; I² may differ by 1 point from Table S4)", rnd(rd("nonlinearity_spline_cox.csv"))),
          ("Scaled Schoenfeld residual tests (final M2)", rnd(rd("ph_test_schoenfeld.csv").rename(columns={"Unnamed: 0": "Predictor"}), 4)),
          ("Mean |SHAP| (gradient-boosted Cox, all data)", rnd(rd("final_xgb_shap_importance.csv"), 4))]),
        ("Table S9", "Sample size, discrimination by treatment setting and risk groups",
         [("Sample-size assessment (Riley et al.; criteria use the optimism-adjusted R²)", pd.DataFrame({"Quantity": list(ex["sample_size"].keys()), "Value": [(f"{v:.4f}" if isinstance(v, float) and v < 1 else str(v)) for v in ex["sample_size"].values()]})),
          ("Pooled Harrell C by treatment setting", rnd(rd("cindex_by_setting.csv"))),
          ("Tertile risk groups from held-out M2 predictions", rnd(pd.DataFrame(ex["risk_groups"]).T.reset_index().rename(columns={"index": "Group"}))),
          ("Trial-stratified hazard ratios vs low-risk group", rnd(rd("risk_groups_hr_M2.csv").rename(columns={"covariate": "Group"})))]),
    ]
    tables = [(c, t, [(l, tidy(d, 5 if any(w in l for w in ("Coefficients", "Preprocessing", "Baseline survival")) else (1 if "Missing values" in l else 3))) for l, d in parts]) for c, t, parts in tables]
    # renumber in order of first citation in the manuscript
    ren = {"Table S5": "Table S9", "Table S6": "Table S5", "Table S7": "Table S8", "Table S8": "Table S6", "Table S9": "Table S7"}
    tables = sorted([(ren.get(c, c), t, p) for c, t, p in tables], key=lambda x: int(x[0].split("S")[1]))
    # Excel
    with pd.ExcelWriter(OUT / "Supplementary_File_S1.xlsx") as xw:
        for code, title, parts in tables:
            r = 0
            pd.DataFrame({f"{code}. {title}": []}).to_excel(xw, sheet_name=code, index=False, startrow=r); r = 2
            for lab, df in parts:
                pd.DataFrame({lab: []}).to_excel(xw, sheet_name=code, index=False, startrow=r); r += 1
                df.to_excel(xw, sheet_name=code, index=False, startrow=r); r += len(df) + 3
    # JSON for Word
    js = []
    for code, title, parts in tables:
        js.append(dict(code=code, title=title, parts=[dict(label=lab, columns=[str(c) for c in df.columns],
                                                           rows=[["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row]
                                                                 for row in df.itertuples(index=False)]) for lab, df in parts]))
    json.dump(js, open(OUT / "supplement_tables.json", "w", encoding="utf8"), ensure_ascii=False)
    print("written", [t[0] for t in tables])


if __name__ == "__main__":
    build()
