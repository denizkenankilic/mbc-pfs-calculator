"""Table 1, endpoint validation against published trial reports, missingness summary, KM figure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from common import OUT_FIG, OUT_TAB, PROCESSED

ORDER = ["PFE111", "PFE113", "SAN135", "PFE112", "LLY168"]
LABEL = {"PFE111": "Pfizer A6181107 (cape, pretreated)", "PFE113": "Pfizer A6181099 (cape, pretreated)",
         "SAN135": "Sanofi EFC6089 (cape, pretreated)", "PFE112": "Pfizer A6181094 (bev+pac, 1st line)",
         "LLY168": "Lilly ROSE/TRIO-12 (doc, 1st line)"}
COLOR = {"PFE111": "#2a78d6", "PFE113": "#eb6834", "SAN135": "#1baf7a", "PFE112": "#eda100", "LLY168": "#e87ba4"}

# Published comparator-arm medians (months). Sources listed in outputs/tables/endpoint_validation.csv
PUBLISHED = {
    "PFE111": dict(pfs="4.2", os=24.6, ref="Barrios CH et al. Breast Cancer Res Treat 2010;121:121-131",
                   note="interim analysis; trial stopped for futility"),
    "PFE112": dict(pfs="9.2 (7.7-13.0)", os=np.nan, ref="Robert NJ et al. Clin Breast Cancer 2011;11(2):82-92",
                   note="data cut-off 1 Jun 2009, median follow-up 8.1 mo, 70 PFS events (29%); PDS data have longer follow-up"),
    "PFE113": dict(pfs="5.9 (5.4-7.6)", os=np.nan, ref="Crown JP et al. J Clin Oncol 2013;31(23):2870-2878",
                   note="derived 95% CI overlaps published CI"),
    "LLY168": dict(pfs="8.2", os=27.2, ref="Mackey JR et al. J Clin Oncol 2015 (doi:10.1200/JCO.2014.57.1513)",
                   note="investigator-assessed PFS"),
    "SAN135": dict(pfs=np.nan, os=np.nan, ref="No peer-reviewed publication identified", note="unpublished trial"),
}

CONT = [("age", "Age, years"), ("bmi", "BMI, kg/m²"), ("n_met_sites", "Organ sites involved, n"),
        ("hgb", "Haemoglobin, g/dL"), ("wbc", "WBC, ×10⁹/L"), ("anc", "Neutrophils, ×10⁹/L"),
        ("alc", "Lymphocytes, ×10⁹/L"), ("plt", "Platelets, ×10⁹/L"), ("alb", "Albumin, g/dL"),
        ("alp_uln", "ALP, ×ULN"), ("ast_uln", "AST, ×ULN"), ("bili_uln", "Bilirubin, ×ULN"),
        ("creat", "Creatinine, mg/dL"), ("ca", "Calcium, mg/dL"), ("nlr", "NLR")]
BIN = [("ecog", "ECOG ≥1"), ("visceral_mets", "Visceral metastases"), ("liver_mets", "Liver metastases"),
       ("bone_mets", "Bone metastases"), ("measurable", "Measurable disease"), ("prior_taxane", "Prior taxane"),
       ("prior_anthracycline", "Prior anthracycline"), ("prior_endocrine", "Prior endocrine therapy"),
       ("prior_adj_chemo", "Prior (neo)adjuvant chemotherapy"), ("er_pos", "ER positive"), ("tnbc", "Triple-negative")]


def fmt_cont(s, dec=None):
    s = s.dropna()
    if s.empty:
        return "NA"
    q = s.quantile([.25, .5, .75])
    if dec is None:
        dec = 0 if s.median() >= 100 else (1 if s.median() >= 10 else 2)
    return f"{q[.5]:.{dec}f} ({q[.25]:.{dec}f}–{q[.75]:.{dec}f})"


def fmt_bin(s, thr=None):
    s = s.dropna()
    if s.empty:
        return "NA"
    x = (s >= 1) if thr else (s == 1)
    return f"{int(x.sum())} ({100 * x.mean():.0f}%)"


def km_row(t, e):
    k = KaplanMeierFitter().fit(t, e)
    ci = median_survival_times(k.confidence_interval_)
    lo, hi = ci.iloc[0, 0], ci.iloc[0, 1]
    f = lambda v: "NR" if not np.isfinite(v) else f"{v:.1f}"
    return k.median_survival_time_, f"{f(k.median_survival_time_)} ({f(lo)}–{f(hi)})"


def table1(df):
    groups = [(tr, df[df.trial == tr]) for tr in ORDER] + [("All", df)]
    rows = [("N", [str(len(g)) for _, g in groups])]
    for c, lab in CONT:
        rows.append((f"{lab}, median (IQR)", [fmt_cont(g[c], 0 if c == "n_met_sites" else None) for _, g in groups]))
    for c, lab in BIN:
        rows.append((f"{lab}, n (%)", [fmt_bin(g[c], thr=(c == "ecog")) for _, g in groups]))
    rows.append(("PFS events, n", [str(int(g.pfs_event.sum())) for _, g in groups]))
    rows.append(("Median PFS, months (95% CI)", [km_row(g.pfs_months, g.pfs_event)[1] for _, g in groups]))
    rows.append(("Deaths (OS available), n", [str(int(g.os_event.sum())) if g.os_event.notna().any() else "NA" for _, g in groups]))
    t1 = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows], columns=[g for g, _ in groups])
    t1.index.name = "Characteristic"
    # denominators note: percentages among patients with non-missing values
    return t1


def missingness(df):
    vars_ = [c for c, _ in CONT] + [c for c, _ in BIN] + ["ast", "alt", "tprot", "na", "glu", "amc"]
    m = df.groupby("trial")[vars_].apply(lambda g: (100 * g.isna().mean()).round(1)).T[ORDER]
    m["All"] = (100 * df[vars_].isna().mean()).round(1)
    m.index.name = "variable (% missing)"
    return m


def validation(df):
    rows = []
    for tr in ORDER:
        g = df[df.trial == tr]
        pfs_med, pfs_txt = km_row(g.pfs_months, g.pfs_event)
        os_txt = km_row(g.os_months, g.os_event)[1] if g.os_event.notna().any() else "not available"
        fu = km_row(g.pfs_months, 1 - g.pfs_event)[0]
        pub = PUBLISHED[tr]
        rows.append(dict(trial=tr, n=len(g), pfs_events=int(g.pfs_event.sum()),
                         derived_median_pfs=pfs_txt, published_median_pfs=pub["pfs"],
                         derived_median_os=os_txt, published_median_os=pub["os"],
                         pfs_followup_reverse_km=round(fu, 1) if np.isfinite(fu) else np.nan,
                         reference=pub["ref"], note=pub["note"]))
    v = pd.DataFrame(rows)
    lly = df[df.trial == "LLY168"]
    if "pfs_event_alg" in lly and lly.pfs_event_alg.notna().any():
        agree = (lly.pfs_event == lly.pfs_event_alg).mean()
        within = ((lly.pfs_months - lly.pfs_months_alg).abs() * 30.4375 <= 7).mean()
        alg_med = km_row(lly.pfs_months_alg, lly.pfs_event_alg)[1]
        v.attrs["algorithm_check"] = (f"LLY168: harmonised algorithm vs sponsor ADTTE PFS — event agreement "
                                      f"{100 * agree:.1f}%, time within ±7 days {100 * within:.1f}%, "
                                      f"median {alg_med} vs sponsor {km_row(lly.pfs_months, lly.pfs_event)[1]}")
    return v


def km_figure(df):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (ep, title, trials) in zip(axes, [("pfs", "A  Progression-free survival", ORDER),
                                              ("os", "B  Overall survival", ["PFE111", "SAN135", "LLY168"])]):
        for tr in trials:
            g = df[df.trial == tr]
            k = KaplanMeierFitter().fit(g[f"{ep}_months"], g[f"{ep}_event"])
            sf = k.survival_function_
            ls = "--" if tr in ("PFE112", "LLY168") else "-"
            ax.step(sf.index, sf.iloc[:, 0], where="post", color=COLOR[tr], lw=2, ls=ls,
                    label=f"{LABEL[tr]} (n={len(g)})")
        ax.set_xlim(0, 36 if ep == "os" else 24)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Months since randomisation")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#0b0b0b")
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Probability")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Solid: pretreated (capecitabine) trials; dashed: first-line trials.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig_km_by_trial.{ext}", dpi=300)
    plt.close(fig)


def run():
    df = pd.read_csv(PROCESSED / "mbc_pooled.csv")
    t1 = table1(df)
    mi = missingness(df)
    va = validation(df)
    t1.to_csv(OUT_TAB / "table1_baseline.csv")
    mi.to_csv(OUT_TAB / "missingness_by_trial.csv")
    va.to_csv(OUT_TAB / "endpoint_validation.csv", index=False)
    with pd.ExcelWriter(OUT_TAB / "descriptive_tables.xlsx") as xw:
        t1.to_excel(xw, sheet_name="Table1")
        mi.to_excel(xw, sheet_name="Missingness")
        va.to_excel(xw, sheet_name="Endpoint_validation", index=False)
    km_figure(df)
    return t1, mi, va


if __name__ == "__main__":
    t1, mi, va = run()
    pd.set_option("display.width", 250)
    print(t1.to_string())
    print(va.drop(columns=["reference"]).to_string(index=False))
    print(va.attrs.get("algorithm_check"))
    print(mi.to_string())
