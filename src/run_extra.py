"""Additional analyses:
 A. Non-linear Cox (cubic B-splines for continuous predictors) under IECV.
 B. Proportional-hazards check (scaled Schoenfeld residuals) for the final M2 model.
 C. Sample-size adequacy (Riley et al., BMJ 2020 criteria for time-to-event models).
 D. Discrimination by treatment setting (pooled per setting from IECV).
 E. Risk groups from held-out M2 predictions (tertiles): KM, HRs.
 F. Final M2 model export (for the equation, nomogram-type table and the decision-support tool).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.preprocessing import SplineTransformer

from common import OUT_FIG, OUT_TAB
from evaluate import random_effects
from model_prep import BINARY, CLIN, LAB, LOG_VARS, PRETTY, Preprocessor, load
from models import Cox
from run_subanalyses import iecv_simple

CONT = [c for c in CLIN + LAB if c not in BINARY and c not in ("n_met_sites", "n_prior_chemo_adv")]


class SplineCox(Cox):
    name = "M2 spline Cox"

    def __init__(self):
        super().__init__(CLIN + LAB, self.name, penalizer=0.05)

    def _expand(self, X, fit=False):
        if fit:
            self.spl = SplineTransformer(n_knots=3, degree=3, include_bias=False, extrapolation="linear").fit(X[CONT])
        S = pd.DataFrame(self.spl.transform(X[CONT]), index=X.index,
                         columns=self.spl.get_feature_names_out(CONT))
        return pd.concat([X.drop(columns=CONT), S], axis=1)

    def fit(self, df):
        X = self._expand(self.pre.fit(df).transform(df), fit=True)
        self.keep = [c for c in X.columns if X[c].std() > 1e-8]
        self.cph = CoxPHFitter(penalizer=self.penalizer).fit(
            X[self.keep].assign(time=df.time.values, event=df.event.values), "time", "event")
        return self

    def risk(self, df):
        return self.cph.predict_log_partial_hazard(self._expand(self.pre.transform(df))[self.keep]).values


def pooled_c(per):
    lg = lambda p: np.log(p / (1 - p))
    ex = lambda x: 1 / (1 + np.exp(-x))
    r = random_effects(lg(per.estimate.values), (per.se / (per.estimate * (1 - per.estimate))).values)
    return ex(r["pooled"]), ex(r["lo"]), ex(r["hi"]), r["i2"]


def main():
    df = load("pfs")
    out = {}

    # A. spline Cox
    r_lin, p_lin = iecv_simple(df, lambda: Cox(CLIN + LAB, "M2"), "M2 linear (reference)")
    r_spl, p_spl = iecv_simple(df, SplineCox, "M2 cubic splines")
    pd.DataFrame([p_lin, p_spl]).to_csv(OUT_TAB / "nonlinearity_spline_cox.csv", index=False)
    out["spline"] = [p_lin, p_spl]

    # B. PH test on final model (all data)
    pre = Preprocessor(CLIN + LAB).fit(df)
    X = pre.transform(df)
    data = X.assign(time=df.time.values, event=df.event.values)
    final = CoxPHFitter(penalizer=0.01).fit(data, "time", "event")
    ph = proportional_hazard_test(final, data, time_transform="rank").summary
    ph.index = [PRETTY.get(i, i) for i in ph.index]
    ph.to_csv(OUT_TAB / "ph_test_schoenfeld.csv")
    out["ph_global_min_p"] = float(ph.p.min())
    out["ph_n_p_lt_0.01"] = int((ph.p < 0.01).sum())

    # C. sample size (Riley 2020): criterion (i) expected shrinkage >= 0.9, criterion (ii) optimism in R2_Nagelkerke <= 0.05
    n, E, P = len(df), int(df.event.sum()), len(CLIN + LAB)
    lr = final.log_likelihood_ratio_test().test_statistic
    r2cs = 1 - np.exp(-lr / n)
    S = 0.9
    n_req_i = P / ((S - 1) * np.log(1 - r2cs / S))
    # max R2_CS for survival models approximated from the event count (Riley 2020, eq. for time-to-event)
    ll_null_approx = E * np.log(E / n) - E  # approximation used by pmsampsize for survival outcomes
    max_r2cs = 1 - np.exp(2 * ll_null_approx / n)
    s_ii = r2cs / (r2cs + 0.05 * max_r2cs)
    n_req_ii = P / ((s_ii - 1) * np.log(1 - r2cs / s_ii))
    # Riley et al. recommend an optimism-adjusted R2: apparent R2 x van Houwelingen shrinkage
    s_vh = (lr - P) / lr
    r2adj = r2cs * s_vh
    n_req_i_adj = P / ((S - 1) * np.log(1 - r2adj / S))
    s_ii_adj = r2adj / (r2adj + 0.05 * max_r2cs)
    n_req_ii_adj = P / ((s_ii_adj - 1) * np.log(1 - r2adj / s_ii_adj))
    out["sample_size"] = dict(n=n, events=E, parameters=P, events_per_parameter=round(E / P, 1),
                              R2_CS_apparent=round(r2cs, 4), required_n_criterion_i_apparentR2=int(np.ceil(n_req_i)),
                              required_n_criterion_ii_apparentR2=int(np.ceil(n_req_ii)),
                              R2_CS_adjusted=round(r2adj, 4), required_n_criterion_i=int(np.ceil(n_req_i_adj)),
                              required_n_criterion_ii=int(np.ceil(n_req_ii_adj)),
                              expected_shrinkage_vanhouwelingen=round(s_vh, 3), LR_chi2=round(lr, 1))

    # D. by setting
    per = pd.read_csv(OUT_TAB / "iecv_per_trial_pfs.csv")
    per = per[per.metric == "harrell_c"]
    setting = {"PFE111": "pretreated", "PFE113": "pretreated", "SAN135": "pretreated", "PFE112": "first-line", "LLY168": "first-line"}
    per["setting"] = per.trial.map(setting)
    rows = []
    for (m, s), g in per.groupby(["model", "setting"]):
        c, lo, hi, i2 = pooled_c(g)
        rows.append(dict(model=m, setting=s, k=len(g), pooled_c=c, lo=lo, hi=hi, i2=i2))
    pd.DataFrame(rows).to_csv(OUT_TAB / "cindex_by_setting.csv", index=False)

    # E. risk groups from held-out M2 predictions (tertiles of the linear predictor within each held-out trial)
    pr = pd.read_csv(OUT_TAB / "iecv_predictions_pfs.csv")
    d = pr[pr.model == "M2"].copy()
    d["group"] = d.groupby("trial")["lp"].transform(lambda s: pd.qcut(s, 3, labels=["Low", "Intermediate", "High"]))
    d = d.assign(high=(d.group == "High").astype(float), inter=(d.group == "Intermediate").astype(float))
    cg = CoxPHFitter().fit(d[["time", "event", "high", "inter", "trial"]], "time", "event", strata=["trial"])
    hr = cg.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    hr.to_csv(OUT_TAB / "risk_groups_hr_M2.csv")
    med = {}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = {"Low": "#86b6ef", "Intermediate": "#2a78d6", "High": "#0d366b"}
    for gname in ["Low", "Intermediate", "High"]:
        g = d[d.group == gname]
        k = KaplanMeierFitter().fit(g.time, g.event)
        med[gname] = dict(n=len(g), events=int(g.event.sum()), median_pfs=float(k.median_survival_time_),
                          pfs6=float(k.survival_function_at_times(6).iloc[0]), pfs12=float(k.survival_function_at_times(12).iloc[0]))
        sf = k.survival_function_
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=colors[gname], lw=2,
                label=f"{gname} risk (n={len(g)}; median {k.median_survival_time_:.1f} mo)")
    ax.set_xlim(0, 24); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months since randomisation"); ax.set_ylabel("Progression-free survival")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.text(0.98, 0.55, f"HR high vs low {hr.loc['high', 'exp(coef)']:.2f}\n(95% CI {hr.loc['high', 'exp(coef) lower 95%']:.2f}–{hr.loc['high', 'exp(coef) upper 95%']:.2f}),\ntrial-stratified",
            transform=ax.transAxes, ha="right", fontsize=7, color="#52514e")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig_riskgroups_M2_heldout.{ext}", dpi=300)
    plt.close(fig)
    out["risk_groups"] = med
    out["risk_groups_hr"] = hr.round(3).to_dict()

    # F. export final model
    bs = final.baseline_survival_
    s0 = {h: float(bs.loc[bs.index <= h].iloc[-1, 0]) for h in (6, 12)}
    export = dict(features=CLIN + LAB, log_vars=sorted(set(CLIN + LAB) & LOG_VARS), binary=sorted(set(CLIN + LAB) & BINARY),
                  lo=pre.lo.to_dict(), hi=pre.hi.to_dict(), mu=pre.mu.to_dict(), sd=pre.sd.to_dict(),
                  coef=final.params_.to_dict(), baseline_survival=s0,
                  medians=df[CLIN + LAB].median().to_dict(),
                  lp_tertiles=np.quantile(final.predict_log_partial_hazard(X), [1 / 3, 2 / 3]).tolist(),
                  note="S(t|x) = S0(t)^exp(lp); lp = sum coef_j * (clip(tx_j) - mu_j)/sd_j; tx = log for log_vars")
    json.dump(export, open(OUT_TAB / "final_model_M2.json", "w"), indent=1)
    # coefficient table per original unit
    tab = []
    for f in CLIN + LAB:
        b = final.params_[f] / pre.sd[f]
        unit = "per 1-unit increase in log(value)" if f in LOG_VARS else ("present vs absent" if f in BINARY else "per 1-unit increase")
        tab.append(dict(variable=PRETTY[f], coef_original_scale=b, HR_original_scale=np.exp(b), scale=unit))
    pd.DataFrame(tab).to_csv(OUT_TAB / "final_model_M2_coefficients_original_scale.csv", index=False)

    json.dump(out, open(OUT_TAB / "extra_analyses_summary.json", "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps(o, indent=1, default=float))
