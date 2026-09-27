# MBC routine-blood-test prognostic modelling — analysis code

Code for: "Routine Blood Tests, Cox Regression and Machine Learning for Predicting Progression-Free Survival in
Metastatic Breast Cancer: An Internal–External Validation Study Across Five Phase III Trials".
No patient-level data are included: the source data must be obtained from Project Data Sphere
(https://data.projectdatasphere.org) under its data-use terms, which do not permit redistribution.

## Requirements
Python 3.11; `pip install -r requirements.txt` (pandas, numpy, openpyxl, scikit-learn, lifelines, scikit-survival,
xgboost, shap, torch [CPU build is sufficient], matplotlib). Node.js with the `docx` package only for the Word supplement.

## Folder layout
```
data/raw/<trial>/   PDS files exactly as downloaded
  pfizer_111/ (A6181107)  pfizer_112/ (A6181094)  pfizer_113/ (A6181099)
  lilly_168/ (ROSE/TRIO-12; unzipped ADaM zip)  sanofi_135/ (EFC6089; unzipped datasets zip)
src/
  common.py, labs.py, baseline_labs.py, drugs.py, endpoints.py      helpers: units, plausibility, ×ULN, PFS algorithm
  extract_pfizer.py / extract_lilly.py / extract_sanofi.py         per-sponsor extraction
  build_dataset.py        pooled analysis dataset, dictionary, cleaning log, patient flow
  descriptives.py         Table 1, missingness, endpoint validation, Kaplan–Meier figure
  model_prep.py, models.py, evaluate.py                             preprocessing, models M0–M6, metrics, pooling
  run_iecv.py             internal–external cross-validation (--outcome pfs | os)
  run_subanalyses.py      blood-block LRT, final HR table, points score, SHAP, missing data, NLR, receptor status
  run_tuning_sensitivity.py  wider hyper-parameter grids
  run_extra.py            splines, PH tests, sample size, C by setting, risk groups, final model export (calculator)
  figures.py, manuscript_figures.py, make_results_tables.py         figures (600 dpi) and result workbooks
  make_supplement.py, build_supplement_docx.js                      Supplementary File S1 (xlsx + docx)
```
## Run (order matters)
Either `cd src && ./run_all.sh`, or step by step:
```
cd src
python build_dataset.py
python descriptives.py
python run_iecv.py --outcome pfs
python run_iecv.py --outcome os
python run_subanalyses.py
python run_tuning_sensitivity.py
python run_extra.py            # reads iecv_predictions_pfs.csv; exports final_model_M2.json
python build_calculator.py     # embeds the final model into dst/mbc_pfs_calculator.html
python figures.py
python manuscript_figures.py
python make_results_tables.py
python make_supplement.py
node build_supplement_docx.js ../outputs/manuscript
```
Seed 2026 throughout. The calculator (dst/mbc_pfs_calculator.html) embeds outputs/tables/final_model_M2.json.

## Online calculator
`index.html` is the stand-alone calculator (research use only), served via GitHub Pages.

## Data
No patient-level data are included in this repository (Project Data Sphere terms do not permit redistribution).
