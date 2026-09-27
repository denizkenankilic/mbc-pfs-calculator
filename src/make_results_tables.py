"""Collects modelling outputs into manuscript-ready tables (Excel, one sheet per table)."""
import json

import pandas as pd

from common import OUT_TAB

LAB = {"M0": "M0 Points score", "M1": "M1 Cox clinical", "M2": "M2 Cox clinical+blood", "M3": "M3 LASSO-Cox",
       "M4": "M4 Random survival forest", "M5": "M5 Gradient-boosted Cox", "M6": "M6 DeepSurv"}
MET = ["harrell_c", "uno_c", "auc_6", "auc_12", "ibs_12", "cal_slope", "oe_6"]
MNAME = {"harrell_c": "Harrell C", "uno_c": "Uno C (τ=12 mo)", "auc_6": "AUC 6 mo", "auc_12": "AUC 12 mo",
         "ibs_12": "IBS 0–12 mo", "cal_slope": "Calibration slope", "oe_6": "O/E 6 mo"}


def perf_table(outcome):
    p = pd.read_csv(OUT_TAB / f"iecv_pooled_{outcome}.csv")
    rows = []
    for m in LAB:
        r = {"Model": LAB[m]}
        for k in MET:
            x = p[(p.model == m) & (p.metric == k)].iloc[0]
            r[MNAME[k]] = f"{x.pooled:.3f} ({x.lo:.3f}–{x.hi:.3f})"
            if k == "harrell_c":
                r["I² (C)"] = f"{100 * x.i2:.0f}%"
                r["95% PI (C)"] = f"{x.pi_lo:.3f}–{x.pi_hi:.3f}" if pd.notna(x.pi_lo) else "NA"
        rows.append(r)
    return pd.DataFrame(rows)


def diff_table(outcome):
    d = pd.read_csv(OUT_TAB / f"iecv_cdiff_pooled_{outcome}.csv")
    d = d[d.reference == "M2"]
    return pd.DataFrame({"Model vs M2": d.model.map(LAB),
                         "ΔC pooled (95% CI)": [f"{a:+.3f} ({b:+.3f} to {c:+.3f})" for a, b, c in zip(d.pooled, d.lo, d.hi)],
                         "I²": [f"{100 * i:.0f}%" for i in d.i2]})


def per_trial(outcome):
    p = pd.read_csv(OUT_TAB / f"iecv_per_trial_{outcome}.csv")
    p = p[p.metric.isin(["harrell_c", "cal_slope", "oe_6"])]
    p["value"] = [f"{e:.3f} ({l:.3f}–{h:.3f})" for e, l, h in zip(p.estimate, p.boot_lo, p.boot_hi)]
    t = p.pivot_table(index=["metric", "model"], columns="trial", values="value", aggfunc="first")
    t.index = [f"{MNAME[a]} — {LAB[b]}" for a, b in t.index]
    return t


def run():
    sub = json.load(open(OUT_TAB / "subanalyses_summary.json"))
    with pd.ExcelWriter(OUT_TAB / "results_tables.xlsx") as xw:
        perf_table("pfs").to_excel(xw, sheet_name="T2_PFS_pooled", index=False)
        diff_table("pfs").to_excel(xw, sheet_name="T3_PFS_deltaC_vs_M2", index=False)
        per_trial("pfs").to_excel(xw, sheet_name="S_PFS_per_trial")
        perf_table("os").to_excel(xw, sheet_name="S_OS_pooled", index=False)
        diff_table("os").to_excel(xw, sheet_name="S_OS_deltaC_vs_M2", index=False)
        pd.read_csv(OUT_TAB / "final_cox_hr_per_sd.csv").to_excel(xw, sheet_name="T4_final_Cox_HR", index=False)
        pd.read_csv(OUT_TAB / "final_points_score.csv").to_excel(xw, sheet_name="T5_points_score", index=False)
        pd.read_csv(OUT_TAB / "final_xgb_shap_importance.csv").to_excel(xw, sheet_name="S_SHAP_importance", index=False)
        pd.read_csv(OUT_TAB / "missing_robustness_pooled.csv").to_excel(xw, sheet_name="S_missing_robustness", index=False)
        pd.read_csv(OUT_TAB / "lymphocyte_scenario_LLY168.csv").to_excel(xw, sheet_name="S_lymphocyte_LLY168", index=False)
        pd.read_csv(OUT_TAB / "nlr_added_value_4trials.csv").to_excel(xw, sheet_name="S_NLR_4trials", index=False)
        pd.DataFrame([sub["lrt_blood"]]).to_excel(xw, sheet_name="S_LRT_blood_block", index=False)
        pd.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in sub["receptor"].items()}]).to_excel(
            xw, sheet_name="S_receptor_status", index=False)
        pd.read_csv(OUT_TAB / "prior_endocrine_vs_er_crosstab.csv").to_excel(xw, sheet_name="S_endocrine_vs_ER", index=False)
        ex = json.load(open(OUT_TAB / "extra_analyses_summary.json"))
        pd.read_csv(OUT_TAB / "tuning_sensitivity_pooled.csv").to_excel(xw, sheet_name="S_tuning_sensitivity", index=False)
        pd.read_csv(OUT_TAB / "nonlinearity_spline_cox.csv").to_excel(xw, sheet_name="S_spline_Cox", index=False)
        pd.read_csv(OUT_TAB / "ph_test_schoenfeld.csv").to_excel(xw, sheet_name="S_PH_test", index=False)
        pd.DataFrame([ex["sample_size"]]).to_excel(xw, sheet_name="S_sample_size", index=False)
        pd.read_csv(OUT_TAB / "cindex_by_setting.csv").to_excel(xw, sheet_name="S_C_by_setting", index=False)
        pd.DataFrame(ex["risk_groups"]).T.to_excel(xw, sheet_name="S_risk_groups")
        pd.read_csv(OUT_TAB / "risk_groups_hr_M2.csv").to_excel(xw, sheet_name="S_risk_groups_HR", index=False)
        pd.read_csv(OUT_TAB / "final_model_M2_coefficients_original_scale.csv").to_excel(xw, sheet_name="T_final_model_equation", index=False)


if __name__ == "__main__":
    run()
    print(perf_table("pfs").to_string(index=False))
    print(diff_table("pfs").to_string(index=False))
