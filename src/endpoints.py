"""Harmonised time-to-event derivations (days since randomisation).

PFS (investigator-assessed), applied identically to all trials that provide
visit-level tumour response data:
  * event  = first post-baseline progressive disease (PD) or death from any cause,
             whichever comes first;
  * censor = last adequate post-baseline tumour assessment (CR/PR/SD/non-CR-non-PD);
             patients without any adequate post-baseline assessment and without an
             early event are censored at day 1;
  * new anticancer therapy before PD/death -> censor at last adequate assessment
             before the new therapy started;
  * PD/death occurring after >= 2 missed scheduled assessments (i.e. more than
    `max_gap` days after the last adequate assessment, or after randomisation)
             -> censor at last adequate assessment.
These rules follow the FDA "Clinical Trial Endpoints for the Approval of Cancer
Drugs and Biologics" guidance (2018) and mirror the Lilly ROSE/TRIO-12 SAP
(see ADTTE CNSRDSC "Death or Progression After Two or More Missed Visits").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADEQUATE = {"CR", "PR", "SD", "NN"}


def derive_pfs(ids, resp: pd.DataFrame, deaths: pd.Series, newtx: pd.Series | None,
               max_gap: float) -> pd.DataFrame:
    """
    ids    : iterable of patient ids (defines the output rows)
    resp   : DataFrame[id, day, resp] visit-level overall response (resp in CR/PR/SD/NN/PD/NE)
    deaths : Series id -> death day (may be missing for most)
    newtx  : Series id -> first day of new (post-study) anticancer therapy, or None
    max_gap: days allowed between last adequate assessment and PD/death
    Returns DataFrame indexed by id: pfs_days, pfs_event, pfs_reason
    """
    resp = resp[resp["day"] > 0].sort_values(["id", "day"])
    out = []
    grouped = {k: g for k, g in resp.groupby("id")}
    for pid in ids:
        g = grouped.get(pid, pd.DataFrame(columns=["day", "resp"]))
        pd_days = g.loc[g.resp == "PD", "day"]
        first_pd = pd_days.min() if len(pd_days) else np.nan
        dth = deaths.get(pid, np.nan) if deaths is not None else np.nan
        ntx = newtx.get(pid, np.nan) if newtx is not None else np.nan

        # candidate event
        ev_day, ev_type = np.nan, None
        for d, t in ((first_pd, "PD"), (dth, "death")):
            if not np.isnan(d) and (np.isnan(ev_day) or d < ev_day):
                ev_day, ev_type = d, t

        # adequate assessments before the event (and before new therapy)
        adeq = g.loc[g.resp.isin(ADEQUATE), "day"]
        limit = ev_day if not np.isnan(ev_day) else np.inf
        if not np.isnan(ntx):
            limit = min(limit, ntx)
        adeq_before = adeq[adeq < limit] if np.isfinite(limit) else adeq
        last_adeq = adeq_before.max() if len(adeq_before) else np.nan
        ref = last_adeq if not np.isnan(last_adeq) else 0.0

        if ev_type is not None and not np.isnan(ntx) and ntx < ev_day:
            out.append((pid, max(last_adeq, 1) if not np.isnan(last_adeq) else 1, 0, "censored: new anticancer therapy"))
        elif ev_type is not None and (ev_day - ref) > max_gap:
            out.append((pid, max(last_adeq, 1) if not np.isnan(last_adeq) else 1, 0, f"censored: {ev_type} after >=2 missed assessments"))
        elif ev_type is not None:
            out.append((pid, max(ev_day, 1), 1, f"event: {ev_type}"))
        elif not np.isnan(last_adeq):
            out.append((pid, max(last_adeq, 1), 0, "censored: last adequate assessment"))
        else:
            out.append((pid, 1, 0, "censored: no post-baseline assessment"))
    return pd.DataFrame(out, columns=["id", "pfs_days", "pfs_event", "pfs_reason"]).set_index("id")


def median_gap(resp: pd.DataFrame) -> float:
    """Median interval (days) between consecutive post-baseline assessments."""
    r = resp[resp["day"] > 0].sort_values(["id", "day"]).drop_duplicates(["id", "day"])
    gaps = r.groupby("id")["day"].diff().dropna()
    gaps = gaps[gaps > 7]
    return float(gaps.median())


def km_median(t, e) -> float:
    from lifelines import KaplanMeierFitter
    k = KaplanMeierFitter().fit(t, e)
    return float(k.median_survival_time_)
