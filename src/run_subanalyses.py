"""Secondary analyses (SAP §RQ1, RQ4, RQ5, sub-analyses).

1. Added value of blood tests: likelihood-ratio test (M1 vs M2) in all data, trial-stratified Cox.
2. Final models on all data: Cox hazard ratios, points-score table, SHAP (XGBoost) + LASSO coefficients.
3. Missing-data robustness: complete-case vs imputation (M2), native missing handling (XGBoost) — IECV.
4. Systematically missing lymphocytes: NLR-augmented Cox developed without LLY168, applied to LLY168.
5. Receptor status (LLY168 + SAN135): added value of ER / triple-negative status.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

from common import OUT_TAB
from evaluate import Calibrator, boot_indices, metrics, random_effects
from model_prep import CLIN, LAB, PRETTY, Preprocessor, load
from models import SCORE_ITEMS, Cox, PointsScore, StratCox, XGBCox, cindex

B = 200


def lrt_blood(df):
    """Added value of blood block: trial-stratified Cox (clinical vs clinical+blood), all data.
    Hazard-ratio table: final unstratified M2 (the reported model) fitted on all data."""
    clin_s = [c for c in CLIN if c != "pretreated"]  # constant within trial strata
    pre = Preprocessor(CLIN + LAB).fit(df)
    X = pre.transform(df).assign(time=df.time.values, event=df.event.values, trial=df.trial.values)
    f1 = CoxPHFitter(penalizer=0.0).fit(X[clin_s + ["time", "event", "trial"]], "time", "event", strata=["trial"])
    f2 = CoxPHFitter(penalizer=0.0).fit(X[clin_s + LAB + ["time", "event", "trial"]], "time", "event", strata=["trial"])
    lr = 2 * (f2.log_likelihood_ - f1.log_likelihood_)
    p = stats.chi2.sf(lr, len(LAB))
    fm = CoxPHFitter(penalizer=0.01).fit(X[CLIN + LAB + ["time", "event"]], "time", "event")  # same penalty as exported model/calculator
    hr = fm.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
        columns={"exp(coef)": "HR_per_SD", "exp(coef) lower 95%": "lo", "exp(coef) upper 95%": "hi"})
    hr["SD_original_scale"] = [pre.sd[c] for c in hr.index]
    hr["scale"] = ["log" if c in __import__("model_prep").LOG_VARS else "linear" for c in hr.index]
    hr.index = [PRETTY.get(i, i) for i in hr.index]
    return dict(lr_chi2=lr, df=len(LAB), p=p, c_clin_strat=f1.concordance_index_, c_full_strat=f2.concordance_index_,
                c_final_apparent=fm.concordance_index_), hr


def final_models(df):
    out = {}
    ps = PointsScore().fit(df)
    pts = pd.DataFrame({"item": ps.points.index, "log_HR": ps.coef.values, "HR": np.exp(ps.coef.values),
                        "points": ps.points.values})
    out["points"] = pts
    from models import inner_cv
    params, _ = inner_cv(XGBCox, df, XGBCox.GRID)
    out["xgb_params"] = params
    xgb = XGBCox(**params).fit(df)
    import shap
    X = xgb.pre.transform(df)
    sv = shap.TreeExplainer(xgb.m).shap_values(X)
    out["shap_values"] = pd.DataFrame(sv, columns=X.columns)
    out["shap_X"] = X
    imp = pd.Series(np.abs(sv).mean(0), index=[PRETTY[c] for c in X.columns]).sort_values(ascending=False)
    out["shap_importance"] = imp.rename("mean_abs_shap").to_frame()
    return out


def iecv_simple(df, make, name, complete_case_test=False, B=B):
    """IECV C-index for one model factory; optional complete-case restriction."""
    rows = []
    for tr in sorted(df.trial.unique()):
        dev, test = df[df.trial != tr].reset_index(drop=True), df[df.trial == tr].reset_index(drop=True)
        if complete_case_test:
            cc = CLIN + LAB
            dev = dev.dropna(subset=cc).reset_index(drop=True)
            test = test.dropna(subset=cc).reset_index(drop=True)
        m = make().fit(dev)
        s = m.risk(test)
        c = cindex(test, s)
        bs = [cindex(test.iloc[i], s[i]) for i in boot_indices(len(test), B)]
        rows.append(dict(analysis=name, trial=tr, n_test=len(test), harrell_c=c, se=np.std(bs)))
    r = pd.DataFrame(rows)
    lg = lambda p: np.log(p / (1 - p))
    pooled = random_effects(lg(r.harrell_c.values), r.se.values / (r.harrell_c * (1 - r.harrell_c)).values)
    ex = lambda x: 1 / (1 + np.exp(-x))
    return r, dict(analysis=name, pooled_c=ex(pooled["pooled"]), lo=ex(pooled["lo"]), hi=ex(pooled["hi"]), i2=pooled["i2"])


def lymphocyte_scenario(df):
    """Develop on the four trials with lymphocytes, validate in LLY168 (lymphocytes systematically missing)."""
    dev = df[df.trial != "LLY168"].reset_index(drop=True)
    test = df[df.trial == "LLY168"].reset_index(drop=True)
    res = []
    for label, feats in (("M2 (no lymphocytes)", CLIN + LAB), ("M2 + log NLR, NLR imputed in LLY168", CLIN + LAB + ["nlr"])):
        m = Cox(feats, label).fit(dev)
        s = m.risk(test)
        bs = [cindex(test.iloc[i], s[i]) for i in boot_indices(len(test), B)]
        res.append(dict(model=label, harrell_c=cindex(test, s), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    # does NLR add value where it is measured? IECV among the four trials
    four = dev
    r1, p1 = iecv_simple(four, lambda: Cox(CLIN + LAB, "M2"), "M2, 4 trials")
    r2, p2 = iecv_simple(four, lambda: Cox(CLIN + LAB + ["nlr"], "M2+NLR"), "M2 + NLR, 4 trials")
    return pd.DataFrame(res), pd.DataFrame([p1, p2]), pd.concat([r1, r2])


def receptor_subanalysis(df):
    d = df[df.trial.isin(["LLY168", "SAN135"])].copy()
    d = d.dropna(subset=["er_pos"]).reset_index(drop=True)
    pre = Preprocessor(CLIN + LAB).fit(d)
    X = pre.transform(d).assign(time=d.time.values, event=d.event.values, trial=d.trial.values,
                                er_pos=d.er_pos.values)
    X = X.drop(columns=["pretreated"])  # constant within trial (stratified)
    base = [c for c in CLIN + LAB if c != "pretreated"]
    f1 = CoxPHFitter(penalizer=0.01).fit(X[base + ["time", "event", "trial"]], "time", "event", strata=["trial"])
    f2 = CoxPHFitter(penalizer=0.01).fit(X[base + ["er_pos", "time", "event", "trial"]], "time", "event", strata=["trial"])
    lr = 2 * (f2.log_likelihood_ - f1.log_likelihood_)
    hr_er = f2.summary.loc["er_pos", ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]].values
    # agreement between prior endocrine therapy (pooled proxy) and ER status
    agree = pd.crosstab(d.prior_endocrine, d.er_pos)
    return dict(n=len(d), lr_chi2=lr, p=stats.chi2.sf(lr, 1), hr_er=hr_er.tolist(),
                c_without=f1.concordance_index_, c_with=f2.concordance_index_), agree


def run():
    df = load("pfs")
    res = {}
    lrt, hr = lrt_blood(df)
    res["lrt_blood"] = lrt
    hr.to_csv(OUT_TAB / "final_cox_hr_per_sd.csv")
    fm = final_models(df)
    fm["points"].to_csv(OUT_TAB / "final_points_score.csv", index=False)
    fm["shap_importance"].to_csv(OUT_TAB / "final_xgb_shap_importance.csv")
    fm["shap_values"].to_csv(OUT_TAB / "final_xgb_shap_values.csv", index=False)
    fm["shap_X"].to_csv(OUT_TAB / "final_xgb_shap_X.csv", index=False)

    # missing-data robustness
    r_imp, p_imp = iecv_simple(df, lambda: Cox(CLIN + LAB, "M2"), "M2 imputation (primary)")
    r_cc, p_cc = iecv_simple(df, lambda: Cox(CLIN + LAB, "M2"), "M2 complete-case", complete_case_test=True)
    r_xi, p_xi = iecv_simple(df, lambda: XGBCox(), "XGBoost imputed")
    r_xn, p_xn = iecv_simple(df, lambda: XGBCox(impute=False), "XGBoost native missing")
    r_st, p_st = iecv_simple(df, lambda: StratCox(CLIN + LAB), "M2 trial-stratified development")
    miss = pd.DataFrame([p_imp, p_cc, p_xi, p_xn, p_st])
    pd.concat([r_imp, r_cc, r_xi, r_xn, r_st]).to_csv(OUT_TAB / "missing_robustness_per_trial.csv", index=False)
    miss.to_csv(OUT_TAB / "missing_robustness_pooled.csv", index=False)

    ly, ly_pool, ly_per = lymphocyte_scenario(df)
    ly.to_csv(OUT_TAB / "lymphocyte_scenario_LLY168.csv", index=False)
    ly_pool.to_csv(OUT_TAB / "nlr_added_value_4trials.csv", index=False)
    ly_per.to_csv(OUT_TAB / "nlr_added_value_4trials_per_trial.csv", index=False)

    rec, agree = receptor_subanalysis(df)
    res["receptor"] = rec
    agree.to_csv(OUT_TAB / "prior_endocrine_vs_er_crosstab.csv")
    json.dump(res, open(OUT_TAB / "subanalyses_summary.json", "w"), indent=1, default=float)
    return res, hr, fm, miss, ly, ly_pool


if __name__ == "__main__":
    res, hr, fm, miss, ly, ly_pool = run()
    pd.set_option("display.width", 200)
    print(json.dumps(res, indent=1, default=float))
    print(hr.round(3).to_string())
    print(fm["points"].round(3).to_string())
    print(fm["shap_importance"].round(4).to_string())
    print(miss.round(3).to_string())
    print(ly.round(3).to_string()); print(ly_pool.round(3).to_string())
