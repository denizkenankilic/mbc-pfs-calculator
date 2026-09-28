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

## Online calculator (research use only)

**Open it:** https://YOUR-USERNAME.github.io/mbc-pfs-calculator/ (the file `index.html` in this repository; it also works offline — download it and open it in any web browser).

### What it does
For a woman with metastatic breast cancer who is starting chemotherapy-based treatment, the calculator estimates the probability of being **progression-free at 6 and 12 months**, using routine clinical information and baseline blood tests. It implements the final clinical-plus-blood Cox model (M2) from the article, developed on 1,298 patients from the comparator arms of five phase III trials and validated by internal–external cross-validation (pooled C-index 0.60, 95% CI 0.57–0.63).

### Inputs
**Patient and disease**
| Field | How to enter |
|---|---|
| Age | years |
| Height, weight | cm, kg (used to compute BMI) |
| ECOG performance status | 0, 1 or 2 |
| Line of therapy | *First-line for metastatic disease* or *Pretreated (after taxane/anthracycline)* |
| Organ sites involved | number of organs with metastases (e.g. liver, bone, nodes = 3) |
| Prior chemo lines for metastatic disease | number of previous chemotherapy regimens for advanced disease |
| Visceral metastases | yes/no (e.g. liver, lung, pleura, peritoneum/ascites, brain) |
| Measurable disease (RECIST) | yes/no |
| Previous treatment (any setting) | tick endocrine therapy, (neo)adjuvant chemotherapy, taxane, anthracycline if received |

**Baseline blood tests** (values before starting treatment; choose the unit from the drop-down where offered)
| Field | Units |
|---|---|
| Haemoglobin | g/dL, g/L or mmol/L |
| White blood cells, neutrophils, platelets | ×10⁹/L |
| Albumin | g/dL or g/L |
| Creatinine | mg/dL or µmol/L |
| Calcium (total, uncorrected) | mg/dL or mmol/L |
| ALP, AST | U/L **and** your laboratory's upper limit of normal (ULN) |
| Bilirubin | value **and** your laboratory's ULN, in the same unit |

Use the buttons *Load example patient* / *Load higher-risk example* to see how it works. Leaving a field empty is allowed: the missing value is replaced by the median of the development cohort (the estimate is then less individual).

### Outputs
- **Progression-free at 6 and 12 months:** estimated probabilities (e.g. 69% means about 69 of 100 similar patients would be expected to be free of progression at that time).
- **Risk group:** low, intermediate or high (tertiles of the model score in the development cohort).
- **What drives this estimate:** bars show how much each input raises (red) or lowers (green) the risk compared with an average trial patient.

### How to interpret it
- Estimates are **approximate**: discrimination is modest (C-index about 0.60) and calibration varied between trials. Use them for research, risk stratification and trial design — **not** for individual treatment decisions.
- The model reflects patients treated with **chemotherapy-based regimens** (capecitabine, docetaxel, paclitaxel ± bevacizumab) in clinical trials; it does not represent endocrine, CDK4/6-inhibitor, antibody–drug conjugate or HER2-targeted settings. Tumour subtype (ER/PR/HER2) is not included.
- *Pretreated* corresponds to the anthracycline- and taxane-pretreated capecitabine trials; its effect combines prior treatment with regimen differences.
- Values outside the development range are truncated to the 1st/99th percentile.

### Model details
Cox proportional-hazards model with 22 predictors (12 clinical, 10 laboratory); WBC, neutrophils, platelets, creatinine and ALP/AST/bilirubin (as multiples of the ULN) are log-transformed; ridge penalty 0.01; baseline survival pooled across trials. Full equation: Supplementary Table S9 of the article.

## Data
No patient-level data are included in this repository (Project Data Sphere terms do not permit redistribution).

## Citation
If you use this code or calculator, please cite the article (details to be added after publication).
