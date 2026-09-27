"""Sensitivity analysis: ML models re-tuned over a wider (8-combination) grid, same nested IECV design.
Reports Harrell C per held-out trial (bootstrap SE, 100 resamples) and random-effects pooled C."""
import itertools
import json

import numpy as np
import pandas as pd

from common import OUT_TAB
from evaluate import boot_indices, random_effects
from model_prep import load
from models import RSF, DeepSurv, XGBCox, cindex, inner_cv

WIDE = {
    "M4": (RSF, [dict(min_samples_leaf=l, max_features=f) for l, f in itertools.product([5, 15, 30, 60], ["sqrt", 0.5])]),
    "M5": (XGBCox, [dict(max_depth=d, n_estimators=n) for d, n in itertools.product([1, 2, 3, 4], [100, 300])]),
    "M6": (DeepSurv, [dict(hidden=h, dropout=d) for h, d in itertools.product([(8,), (16,), (32, 16), (64, 32)], [0.1, 0.3])]),
}


def main(B=100):
    df = load("pfs")
    rows, tun = [], {}
    for tr in sorted(df.trial.unique()):
        dev, test = df[df.trial != tr].reset_index(drop=True), df[df.trial == tr].reset_index(drop=True)
        idx = boot_indices(len(test), B)
        for key, (cls, grid) in WIDE.items():
            params, inner = inner_cv(cls, dev, grid)
            s = cls(**params).fit(dev).risk(test)
            c = cindex(test, s)
            se = np.std([cindex(test.iloc[i], s[i]) for i in idx])
            rows.append(dict(trial=tr, model=key, harrell_c=c, se=se))
            tun[f"{tr}_{key}"] = dict(params=params, inner_c=inner)
        print(tr, "done", flush=True)
    r = pd.DataFrame(rows)
    lg = lambda p: np.log(p / (1 - p))
    ex = lambda x: 1 / (1 + np.exp(-x))
    pooled = []
    for key, g in r.groupby("model"):
        p = random_effects(lg(g.harrell_c.values), (g.se / (g.harrell_c * (1 - g.harrell_c))).values)
        pooled.append(dict(model=key, grid="wide (8 combos)", pooled_c=ex(p["pooled"]), lo=ex(p["lo"]), hi=ex(p["hi"]), i2=p["i2"]))
    prim = pd.read_csv(OUT_TAB / "iecv_pooled_pfs.csv")
    prim = prim[(prim.metric == "harrell_c") & prim.model.isin(["M2", "M4", "M5", "M6"])]
    for _, x in prim.iterrows():
        pooled.append(dict(model=x.model, grid="primary (4 combos)" if x.model != "M2" else "Cox reference",
                           pooled_c=x.pooled, lo=x.lo, hi=x.hi, i2=x.i2))
    pooled = pd.DataFrame(pooled).sort_values(["model", "grid"])
    r.to_csv(OUT_TAB / "tuning_sensitivity_per_trial.csv", index=False)
    pooled.to_csv(OUT_TAB / "tuning_sensitivity_pooled.csv", index=False)
    json.dump(tun, open(OUT_TAB / "tuning_sensitivity_params.json", "w"), indent=1, default=str)
    print(pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
