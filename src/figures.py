"""Manuscript figures from IECV outputs: forest plot of C-index, calibration, decision curves, SHAP."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from common import OUT_FIG, OUT_TAB
from evaluate import net_benefit
from model_prep import PRETTY

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MODEL_COLOR = {"M0": "#52514e", "M1": "#86b6ef", "M2": "#2a78d6", "M3": "#1c5cab",
               "M4": "#1baf7a", "M5": "#eb6834", "M6": "#e87ba4"}
MODEL_LABEL = {"M0": "Points score", "M1": "Cox clinical", "M2": "Cox clinical+blood", "M3": "LASSO-Cox",
               "M4": "Random survival forest", "M5": "Gradient-boosted Cox", "M6": "DeepSurv"}
TRIAL_ORDER = ["PFE111", "PFE113", "SAN135", "PFE112", "LLY168"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)


def forest(outcome="pfs"):
    per = pd.read_csv(OUT_TAB / f"iecv_per_trial_{outcome}.csv")
    pooled = pd.read_csv(OUT_TAB / f"iecv_pooled_{outcome}.csv")
    per = per[per.metric == "harrell_c"]
    pooled = pooled[pooled.metric == "harrell_c"]
    models = list(MODEL_LABEL)
    trials = [t for t in TRIAL_ORDER if t in per.trial.unique()]
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(models) * (len(trials) + 2) + 0.8))
    y, yt, yl = 0, [], []
    for m in models:
        for t in trials:
            r = per[(per.model == m) & (per.trial == t)].iloc[0]
            ax.plot([r.boot_lo, r.boot_hi], [y, y], color=MODEL_COLOR[m], lw=1.5)
            ax.plot(r.estimate, y, "s", color=MODEL_COLOR[m], ms=4)
            yt.append(y); yl.append(f"   {t}")
            y -= 1
        p = pooled[pooled.model == m].iloc[0]
        ax.fill([p.lo, p.pooled, p.hi, p.pooled], [y, y + 0.35, y, y - 0.35], color=MODEL_COLOR[m])
        if np.isfinite(p.pi_lo) and p.k >= 4:
            ax.plot([p.pi_lo, p.pi_hi], [y, y], color=MODEL_COLOR[m], lw=0.8, ls=":")
        ax.text(0.745, y, f"{p.pooled:.3f} ({p.lo:.3f}–{p.hi:.3f}); I²={100 * p.i2:.0f}%", va="center", fontsize=7, color=INK)
        yt.append(y); yl.append(f"{MODEL_LABEL[m]} — pooled")
        y -= 1.6
    ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7)
    ax.set_xlim(0.45, 0.74)
    ax.set_xlabel("Harrell's C-index in held-out trial (95% bootstrap CI); diamond = random-effects pooled")
    _style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig_forest_cindex_{outcome}.{ext}", dpi=300)
    plt.close(fig)


def calibration(outcome="pfs", models=("M0", "M2", "M4", "M5"), horizon=6.0):
    pr = pd.read_csv(OUT_TAB / f"iecv_predictions_{outcome}.csv")
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 2.8), sharey=True)
    col = f"risk{int(horizon)}"
    for ax, m in zip(axes, models):
        d = pr[pr.model == m].copy()
        d["bin"] = d.groupby("trial")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        pts = []
        for (t, b), g in d.groupby(["trial", "bin"]):
            km = KaplanMeierFitter().fit(g.time, g.event)
            obs = 1 - float(km.survival_function_at_times(horizon).iloc[0])
            pts.append((t, g[col].mean(), obs))
        pts = pd.DataFrame(pts, columns=["trial", "pred", "obs"])
        for t, g in pts.groupby("trial"):
            ax.plot(g.pred, g.obs, "o-", ms=3, lw=1, color=INK2, alpha=0.35)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        ax.plot(pts.groupby(pts.pred.rank(pct=True).mul(5).clip(upper=4.999).astype(int)).pred.mean(),
                pts.groupby(pts.pred.rank(pct=True).mul(5).clip(upper=4.999).astype(int)).obs.mean(),
                "s-", color=MODEL_COLOR[m], lw=2, ms=5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(MODEL_LABEL[m], fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"Predicted {int(horizon)}-mo risk")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"Observed {int(horizon)}-mo risk (KM)")
    fig.text(0.01, 0.005, "Grey: quintiles within each held-out trial; coloured: averaged across trials.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig_calibration_{outcome}_{int(horizon)}m.{ext}", dpi=300)
    plt.close(fig)


def dca(outcome="pfs", models=("M0", "M2", "M4", "M5"), horizon=6.0):
    pr = pd.read_csv(OUT_TAB / f"iecv_predictions_{outcome}.csv")
    th = np.arange(0.10, 0.71, 0.02)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    rows = []
    for m in models:
        d = pr[pr.model == m]
        nb = net_benefit(d.time.values, d.event.values, d[f"risk{int(horizon)}"].values, th, horizon)
        nb["model"] = m
        rows.append(nb)
        ax.plot(nb.threshold, nb.net_benefit, color=MODEL_COLOR[m], lw=2, label=MODEL_LABEL[m])
    ax.plot(th, rows[0].treat_all, color=INK2, lw=1, ls="--", label="Treat all as high risk")
    ax.axhline(0, color=INK, lw=0.8, label="Treat none")
    ax.set_ylim(-0.05, max(r.net_benefit.max() for r in rows) + 0.05)
    ax.set_xlabel(f"Threshold probability of progression by {int(horizon)} months")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig_dca_{outcome}_{int(horizon)}m.{ext}", dpi=300)
    plt.close(fig)
    pd.concat(rows).to_csv(OUT_TAB / f"dca_{outcome}_{int(horizon)}m.csv", index=False)


def shap_plot():
    sv = pd.read_csv(OUT_TAB / "final_xgb_shap_values.csv")
    X = pd.read_csv(OUT_TAB / "final_xgb_shap_X.csv")
    order = sv.abs().mean().sort_values().index
    fig, ax = plt.subplots(figsize=(6, 5.2))
    rng = np.random.default_rng(0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#2a78d6", "#0d366b"])
    for i, c in enumerate(order):
        x = X[c].values
        z = (x - np.nanpercentile(x, 5)) / (np.nanpercentile(x, 95) - np.nanpercentile(x, 5) + 1e-9)
        ax.scatter(sv[c], i + rng.uniform(-0.25, 0.25, len(sv)), c=np.clip(z, 0, 1), cmap=cmap, s=4, lw=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([PRETTY[c] for c in order], fontsize=8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("SHAP value (impact on log relative hazard of progression)")
    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1]); cb.ax.set_yticklabels(["low", "high"], fontsize=7)
    cb.set_label("Feature value", fontsize=7)
    _style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig_shap_xgb.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["forest", "calibration", "dca", "shap"]
    if "forest" in what: forest()
    if "calibration" in what: calibration()
    if "dca" in what: dca()
    if "shap" in what: shap_plot()
