# Data sources, harmonisation and cleaning — methods note (draft for manuscript §2.1–2.4)

## Data sources
De-identified individual patient data from the comparator arms of five randomised phase III trials in
HER2-negative / advanced breast cancer were obtained from Project Data Sphere (PDS):

| ID | Trial (NCT) | Sponsor | Setting | Comparator regimen | Format |
|---|---|---|---|---|---|
| PFE111 | A6181107 (NCT00373113) | Pfizer | pretreated | capecitabine | raw CRF (SAS) |
| PFE113 | A6181099 (NCT00435409) | Pfizer | pretreated | capecitabine | raw CRF (SAS) |
| SAN135 | XRP9881B-3001/EFC6089 (NCT00081796) | Sanofi | pretreated (post taxane + anthracycline) | capecitabine | raw CRF (SAS) |
| PFE112 | A6181094 (NCT00373256) | Pfizer | first-line | bevacizumab + paclitaxel | raw CRF (SAS) |
| LLY168 | ROSE/TRIO-12 (NCT00703326) | Eli Lilly | first-line | placebo + docetaxel | CDISC ADaM |

All 1,299 comparator-arm patients shared on PDS were eligible; one male patient (PFE113) was excluded,
leaving **1,298** patients for analysis.

## Time origin and baseline
Time zero = randomisation. All study days were re-expressed relative to randomisation.
Baseline = last non-missing value from day −35 to day +1 (prespecified window).

## Laboratory harmonisation
Sixteen routine analytes (haemoglobin, WBC, absolute neutrophils/lymphocytes/monocytes, platelets, albumin,
ALP, AST, ALT, total bilirubin, creatinine, calcium, total protein, sodium, glucose) were mapped to a
single target unit using an explicit unit-label conversion table (e.g., g/L→g/dL ÷10; µmol/L bilirubin ÷17.104;
µmol/L creatinine ÷88.42; mmol/L calcium ×4.008; cells/µL→10⁹/L ÷1000).
Values whose label conversion was missing or fell outside prespecified physiological plausibility limits
entered a documented magnitude-rescue step: candidate conversion factors were tried and the value was
retained only if exactly one candidate fell in a typical range; otherwise it was set to missing.
Absolute white-cell differentials exceeding same-day WBC were treated as mislabelled percentages and set to
missing; when absolute differentials were missing, they were derived as percentage × same-day WBC.
Liver enzymes, bilirubin and creatinine were additionally expressed as multiples of the record's own upper
limit of normal (×ULN), which is unit- and assay-independent.
Derived indices: NLR, PLR, SII (platelets × neutrophils / lymphocytes), (LMR computed but not analysed) — computed only when all
components were present (no imputation at this stage).
Every action is recorded in `cleaning_log.csv` (Supplementary Table).

## Clinical covariates
Age, BMI, ECOG PS, visceral metastases (Pfizer: liver, lung, pleura/effusion, ascites, brain, ovary; Sanofi: additionally
peritoneum/omentum, adrenal and other visceral organs as coded; Lilly: sponsor VMETA flag), number of involved organ sites, measurable disease, prior taxane,
anthracycline, endocrine and HER2-targeted therapy, prior (neo)adjuvant chemotherapy and number of prior
chemotherapy regimens for advanced disease. Receptor status (ER/PR/HER2) was available only for LLY168 and
SAN135; prior endocrine therapy serves as a pooled proxy for hormone-receptor positivity. Organ-level lesion
data (liver/bone) were not included in the Lilly ADaM release.

## Endpoints
**PFS (primary)**: harmonised investigator-assessed algorithm applied to visit-level responses —
event = first progressive disease or death; censoring at the last adequate tumour assessment; censoring at
start of new anticancer therapy; PD/death occurring after ≥2 missed scheduled assessments
(> 2 × protocol interval + 14 days after the last adequate assessment) censored at the last adequate assessment
(FDA 2018 endpoint guidance; mirrors the ROSE/TRIO-12 SAP). Protocol assessment intervals: 6 weeks
(PFE111, PFE113, SAN135, LLY168) and 8 weeks (PFE112). For LLY168 the sponsor-derived ADTTE PFS was used.
**Algorithm validation**: applied to LLY168 visit-level responses (ADRS), the harmonised algorithm reproduced
the sponsor PFS event status in 91.2% of patients (time within ±7 days in 87.5%; median 8.1 vs 8.2 months).
**OS (secondary)**: available for PFE111, SAN135 and LLY168 (long-term follow-up). SAN135 death dates are
recorded at week precision (mid-week imputation).

## Agreement with published reports (comparator arms)
| Trial | Derived median PFS (95% CI) | Published | Comment |
|---|---|---|---|
| PFE111 | 4.1 (3.4–4.6) | 4.2 | Barrios 2010 |
| LLY168 | 8.2 (7.1–8.5) | 8.2 | Mackey 2015 |
| PFE113 | 5.4 (4.2–6.8) | 5.9 (5.4–7.6) | Crown 2013 |
| PFE112 | 11.2 (9.1–14.5) | 9.2 | early termination, median FU 8.1 mo in publication; PDS cut has longer follow-up |
| SAN135 | 3.5 (3.0–4.2) | unpublished | consistent with pretreated capecitabine arms |
OS: LLY168 28.7 vs 27.2 (later data cut); PFE111 18.0 vs 24.6 (published estimate from an interim analysis
with few events) — to be discussed as a limitation; OS is a secondary endpoint.


Tumour-site baseline window: lesion records from day -42 to +1 (laboratory/clinical baseline window -35 to +1).
Prior HER2-targeted therapy was derived but not used as a predictor.
Calcium plausibility limits 5-16 mg/dL (mislabelled mEq/L values in EFC6089 resolved by magnitude rescue).
