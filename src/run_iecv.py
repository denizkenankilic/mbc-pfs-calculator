"""Internal–external cross-validation (leave-one-trial-out) for all prespecified models.

Usage: python run_iecv.py [--outcome pfs|os] [--boot 200] [--no-tune] [--trials PFE111,...]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import OUT_TAB
from evaluate import Calibrator, boot_indices, expit, logit, metrics, random_effects
from model_prep import load
from models import make_models, oof_scores

METRICS = ["harrell_c", "uno_c", "auc_6", "auc_12", "ibs_12", "cal_slope", "oe_6"]


def main(outcome="pfs", B=200, tune=True, trials=None, tag=""):
    df = load(outcome)
    trials = trials or sorted(df.trial.unique())
    per, preds, tuning_log, diffs = [], [], {}, []
    for tr in trials:
        t0 = time.time()
        dev, test = df[df.trial != tr].reset_index(drop=True), df[df.trial == tr].reset_index(drop=True)
        fitted, tuning, factories = make_models(dev, tune=tune)
        tuning_log[tr] = tuning
        if "M0" in fitted:
            tuning_log[tr]["M0_points"] = fitted["M0"].points.to_dict()
        scores, cals = {}, {}
        for key, m in fitted.items():
            s_te = m.risk(test)
            cal = Calibrator().fit(oof_scores(factories[key], dev), dev)  # map fitted on out-of-fold dev scores
            scores[key], cals[key] = s_te, cal
            est = metrics(dev, test, s_te, cal)
            preds.append(pd.DataFrame({"trial": tr, "patient_id": test.patient_id, "model": key, "score": s_te,
                                       "lp": cal.lp(s_te), "risk6": 1 - cal.surv(s_te, [6.0])[:, 0],
                                       "risk12": 1 - cal.surv(s_te, [12.0])[:, 0],
                                       "time": test.time, "event": test.event.astype(int)}))
            for k, v in est.items():
                per.append(dict(trial=tr, model=key, name=m.name, metric=k, estimate=v))
        # shared bootstrap resamples -> SEs and paired differences
        idxs = boot_indices(len(test), B)
        boot = {k: [] for k in fitted}
        for i in idxs:
            tb = test.iloc[i].reset_index(drop=True)
            if tb.event.sum() < 5:
                continue
            for key in fitted:
                boot[key].append(metrics(dev, tb, scores[key][i], cals[key]))
        for key in fitted:
            bd = pd.DataFrame(boot[key])
            for k in METRICS:
                for r in per:
                    if r["trial"] == tr and r["model"] == key and r["metric"] == k:
                        r["se"] = float(bd[k].std())
                        r["boot_lo"], r["boot_hi"] = (float(bd[k].quantile(.025)), float(bd[k].quantile(.975)))
        for ref in ("M2", "M0"):
            rb = pd.DataFrame(boot[ref])
            for key in fitted:
                if key == ref:
                    continue
                d = pd.DataFrame(boot[key])["harrell_c"] - rb["harrell_c"]
                point = [r["estimate"] for r in per if r["trial"] == tr and r["model"] == key and r["metric"] == "harrell_c"][0] - \
                        [r["estimate"] for r in per if r["trial"] == tr and r["model"] == ref and r["metric"] == "harrell_c"][0]
                diffs.append(dict(trial=tr, model=key, reference=ref, delta_c=point, se=float(d.std()),
                                  lo=float(d.quantile(.025)), hi=float(d.quantile(.975))))
        print(f"{tr}: done in {time.time() - t0:.0f}s", flush=True)

    per = pd.DataFrame(per)
    preds = pd.concat(preds)
    diffs = pd.DataFrame(diffs)

    # random-effects pooling
    pooled = []
    for (key, k), g in per.groupby(["model", "metric"]):
        est, se = g.estimate.values, g.se.values
        if k in ("harrell_c", "uno_c", "auc_6", "auc_12"):
            ok = (est > 0) & (est < 1)
            r = random_effects(logit(est[ok]), se[ok] / (est[ok] * (1 - est[ok])))
            for f in ("pooled", "lo", "hi", "pi_lo", "pi_hi"):
                r[f] = expit(r[f]) if np.isfinite(r[f]) else r[f]
        elif k == "oe_6":
            ok = est > 0
            r = random_effects(np.log(est[ok]), se[ok] / est[ok])
            for f in ("pooled", "lo", "hi", "pi_lo", "pi_hi"):
                r[f] = np.exp(r[f]) if np.isfinite(r[f]) else r[f]
        else:
            r = random_effects(est, se)
        pooled.append(dict(model=key, name=g.name.iloc[0], metric=k, **r))
    pooled = pd.DataFrame(pooled)
    pdiff = []
    for (key, ref), g in diffs.groupby(["model", "reference"]):
        r = random_effects(g.delta_c.values, g.se.values)
        pdiff.append(dict(model=key, reference=ref, **r))
    pdiff = pd.DataFrame(pdiff)

    sfx = f"_{outcome}{tag}"
    per.to_csv(OUT_TAB / f"iecv_per_trial{sfx}.csv", index=False)
    pooled.to_csv(OUT_TAB / f"iecv_pooled{sfx}.csv", index=False)
    diffs.to_csv(OUT_TAB / f"iecv_cdiff_per_trial{sfx}.csv", index=False)
    pdiff.to_csv(OUT_TAB / f"iecv_cdiff_pooled{sfx}.csv", index=False)
    preds.to_csv(OUT_TAB / f"iecv_predictions{sfx}.csv", index=False)
    json.dump(tuning_log, open(OUT_TAB / f"iecv_tuning{sfx}.json", "w"), indent=1, default=str)
    return per, pooled, pdiff


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="pfs")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--trials", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    per, pooled, pdiff = main(a.outcome, a.boot, not a.no_tune, a.trials.split(",") if a.trials else None, a.tag)
    pd.set_option("display.width", 250)
    print(pooled.pivot(index="model", columns="metric", values="pooled").round(3).to_string())
    print(pdiff.round(3).to_string())
