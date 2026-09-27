"""Main-text figures at 600 dpi (MDPI requirement) -> outputs/manuscript/figures/Figure1..7.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib.patches import FancyBboxPatch

import descriptives as D
import figures as F
from common import OUT_TAB, ROOT

OUT = ROOT / "outputs" / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 600
INK, INK2 = "#0b0b0b", "#52514e"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)


def fig1_design():
    flow = pd.read_csv(OUT_TAB / "patient_flow.csv").set_index("trial")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f6f8", ec="#52514e", fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", wrap=True)

    box(18, 53, 64, 7, "Project Data Sphere: comparator arms of 5 phase III MBC trials (3 sponsors)\n1,299 patients shared; 1 male excluded → 1,298 analysed", bold=True)
    labels = [("PFE111", "Pfizer A6181107\ncapecitabine\npretreated"), ("PFE113", "Pfizer A6181099\ncapecitabine\npretreated"),
              ("SAN135", "Sanofi EFC6089\ncapecitabine\npretreated"), ("PFE112", "Pfizer A6181094\nbev + paclitaxel\nfirst-line"),
              ("LLY168", "Lilly ROSE/TRIO-12\nplacebo + docetaxel\nfirst-line")]
    for i, (k, lab) in enumerate(labels):
        x = 1 + i * 19.8
        box(x, 36, 18, 13, f"{lab}\nn = {flow.loc[k, 'analysed']}", fc="#e8f1fb")
        ax.annotate("", xy=(x + 9, 49.5), xytext=(50, 52.6), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    box(1, 20, 98, 11, "Harmonisation: baseline window −35 to +1 days; 16 analytes (300,574 records) in common units\n"
        "plausibility limits with documented rescaling; liver enzymes and bilirubin as ×ULN\n"
        "harmonised investigator PFS algorithm (91% event agreement with sponsor PFS in LLY168)", fs=6.5)
    box(1, 3, 47, 13, "Internal–external cross-validation\ndevelop on 4 trials, validate on the\nheld-out trial (×5); preprocessing,\nimputation and tuning in development data", fc="#fdf1e8", fs=6.5)
    box(52, 3, 47, 13, "Models: points score; Cox (clinical);\nCox (clinical + blood); LASSO-Cox; RSF;\ngradient-boosted Cox; DeepSurv\nMetrics: C-index, AUC(t), calibration, IBS, DCA, SHAP", fc="#fdf1e8", fs=6.5)
    for x in (25, 75):
        ax.annotate("", xy=(x, 16.5), xytext=(x, 19.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.annotate("", xy=(50, 31.5), xytext=(50, 35.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    fig.tight_layout()
    save(fig, "Figure1")


def fig3_forest_compact():
    per = pd.read_csv(OUT_TAB / "iecv_per_trial_pfs.csv").query("metric=='harrell_c'")
    pooled = pd.read_csv(OUT_TAB / "iecv_pooled_pfs.csv").query("metric=='harrell_c'")
    models = list(F.MODEL_LABEL)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    marker = {"PFE111": "o", "PFE113": "s", "SAN135": "^", "PFE112": "D", "LLY168": "v"}
    for i, m in enumerate(models[::-1]):
        y = i
        p = pooled[pooled.model == m].iloc[0]
        ax.plot([p.lo, p.hi], [y, y], color=F.MODEL_COLOR[m], lw=3, solid_capstyle="round")
        ax.plot(p.pooled, y, "D", color=F.MODEL_COLOR[m], ms=8, mec="white", mew=1)
        if np.isfinite(p.pi_lo):
            ax.plot([p.pi_lo, p.pi_hi], [y, y], color=F.MODEL_COLOR[m], lw=0.8, ls=":")
        for _, r in per[per.model == m].iterrows():
            ax.plot(r.estimate, y + 0.28, marker[r.trial], color=INK2, ms=3.5, mfc="white", mew=0.8)
        ax.text(0.712, y, f"{p.pooled:.3f} ({p.lo:.3f}–{p.hi:.3f})", va="center", fontsize=7)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([F.MODEL_LABEL[m] for m in models[::-1]], fontsize=8)
    ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xlim(0.48, 0.80)
    ax.set_xlabel("Harrell's C-index (held-out trials)")
    hs = [plt.Line2D([], [], marker=marker[t], color=INK2, mfc="white", ls="", ms=4, label=t) for t in marker]
    hs += [plt.Line2D([], [], marker="D", color=INK2, ls="-", lw=3, ms=6, label="Pooled (95% CI)"),
           plt.Line2D([], [], color=INK2, ls=":", lw=0.8, label="95% prediction interval")]
    ax.legend(handles=hs, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=7, handletextpad=0.3, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6e5e1", lw=0.8)
    fig.tight_layout()
    save(fig, "Figure3")


def fig7_riskgroups():
    pr = pd.read_csv(OUT_TAB / "iecv_predictions_pfs.csv")
    d = pr[pr.model == "M2"].copy()
    d["group"] = d.groupby("trial")["lp"].transform(lambda s: pd.qcut(s, 3, labels=["Low", "Intermediate", "High"]))
    hr = pd.read_csv(OUT_TAB / "risk_groups_hr_M2.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = {"Low": "#86b6ef", "Intermediate": "#2a78d6", "High": "#0d366b"}
    for g in ["Low", "Intermediate", "High"]:
        s = d[d.group == g]
        k = KaplanMeierFitter().fit(s.time, s.event)
        sf = k.survival_function_
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=colors[g], lw=2,
                label=f"{g} risk (n={len(s)}; median {k.median_survival_time_:.1f} mo)")
    ax.set_xlim(0, 24); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months since randomisation"); ax.set_ylabel("Progression-free survival")
    ax.legend(frameon=False, fontsize=7)
    ax.text(0.98, 0.55, f"HR high vs low {hr.loc['high', 'exp(coef)']:.2f} (95% CI {hr.loc['high', 'exp(coef) lower 95%']:.2f}–"
            f"{hr.loc['high', 'exp(coef) upper 95%']:.2f})\nHR intermediate vs low {hr.loc['inter', 'exp(coef)']:.2f} "
            f"({hr.loc['inter', 'exp(coef) lower 95%']:.2f}–{hr.loc['inter', 'exp(coef) upper 95%']:.2f})\ntrial-stratified Cox",
            transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    fig.tight_layout()
    save(fig, "Figure7")


def rerender_existing():
    """Re-save existing figure functions at 600 dpi into the manuscript folder."""
    import shutil
    orig = plt.Figure.savefig

    def hi(self, fname, *a, **k):
        k["dpi"] = DPI
        return orig(self, fname, *a, **k)
    plt.Figure.savefig = hi
    D.km_figure(pd.read_csv(ROOT / "data" / "processed" / "mbc_pooled.csv"))
    F.calibration(); F.dca(); F.shap_plot(); F.forest("os"); F.calibration("os", horizon=12.0); F.forest("pfs")
    plt.Figure.savefig = orig
    src = ROOT / "outputs" / "figures"
    for a, b in [("fig_km_by_trial.png", "Figure2.png"), ("fig_calibration_pfs_6m.png", "Figure4.png"),
                 ("fig_dca_pfs_6m.png", "Figure5.png"), ("fig_shap_xgb.png", "Figure6.png"),
                 ("fig_forest_cindex_pfs.png", "FigureS1.png"), ("fig_forest_cindex_os.png", "FigureS2.png"),
                 ("fig_calibration_os_12m.png", "FigureS3.png")]:
        shutil.copy(src / a, OUT / b)


if __name__ == "__main__":
    fig1_design(); fig3_forest_compact(); fig7_riskgroups(); rerender_existing()
    print(sorted(p.name for p in OUT.iterdir()))
