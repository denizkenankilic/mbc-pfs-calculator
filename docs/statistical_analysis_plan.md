# Statistical analysis plan (version 1.0 fixed before modelling, 26 Sep 2026; amendments listed at the end, 27 Sep 2026)

## Objective
Develop and externally validate prognostic models for progression-free survival (PFS) in metastatic breast
cancer (MBC) from routinely available clinical variables and blood tests, comparing a simple points-based
clinical score, classical Cox regression and machine-learning (ML) survival models.

## Population
1,298 female patients from the comparator arms of five phase III trials (see data_cleaning_methods.md).

## Outcome
Primary: PFS (harmonised definition). Prediction horizons: 6 and 12 months.
Secondary: overall survival (OS) in the three trials with long-term follow-up (PFE111, SAN135, LLY168).

## Predictors (core set — available in all five trials)
Clinical: age, BMI, ECOG (0 / ≥1), visceral metastases, number of organ sites, measurable disease,
prior endocrine therapy, prior (neo)adjuvant chemotherapy, prior taxane, prior anthracycline, number of prior
chemotherapy lines for advanced disease, treatment setting (first-line vs pretreated).
Blood: haemoglobin, WBC, absolute neutrophils, platelets, albumin, ALP×ULN, AST×ULN, bilirubin×ULN,
creatinine, calcium.
Transformations: log for right-skewed labs (WBC, neutrophils, platelets, ALP/AST/bilirubin×ULN, creatinine);
winsorisation at the 1st/99th percentile of the development data; standardisation (development-data mean/SD).
Not in core set (sub-analyses): lymphocytes/NLR/PLR/SII (absent in LLY168); ER/PR/HER2 (LLY168, SAN135 only).

## Models
M0 Simple clinical score: dichotomised predictors at standard clinical cut-offs — ECOG ≥1; visceral metastases;
≥3 organ sites; pretreated setting; haemoglobin <12 g/dL; albumin <3.5 g/dL; ALP >1×ULN; neutrophils >7.5×10⁹/L;
points = development-data Cox coefficient / 0.25, rounded (1 point ≈ HR 1.28; items with non-positive
coefficients receive 0 points) — Sullivan-type integer scoring. (Amended before the final run: dividing by the
smallest coefficient produced unusable point ranges.)
M1 Cox, clinical variables only.   M2 Cox, clinical + blood (full).   M3 LASSO-Cox (clinical + blood).
M4 Random survival forest.   M5 Gradient-boosted Cox (XGBoost, objective survival:cox).   M6 DeepSurv (MLP).
Hyper-parameters of M3–M6 tuned by 5-fold cross-validation (Harrell's C) inside the development data only.

## Validation — internal–external cross-validation (IECV)
Each trial is held out once; models are developed on the remaining four trials (all preprocessing, imputation,
tuning and score-point derivation re-done inside each development set) and evaluated on the held-out trial.
Because comparator regimens differ between trials, the baseline hazard is not assumed transportable;
risk scores are mapped to survival probabilities through a Cox calibration model (risk score as single
covariate) fitted in the development data.

## Performance measures (per held-out trial)
Discrimination: Harrell's C; Uno's C (τ = 12 months); time-dependent AUC at 6 and 12 months.
Calibration: calibration slope (Cox regression of held-out outcome on the linear risk score);
observed/expected ratio of 6-month progression risk (Kaplan–Meier observed vs mean predicted).
Overall: integrated Brier score (1–12 months).
Uncertainty: 200 bootstrap resamples of the held-out trial.
Pooling: random-effects meta-analysis (DerSimonian–Laird) of per-trial estimates, reporting pooled estimate,
95% CI, I² and 95% prediction interval; C-index pooled on the logit scale.
Model comparison: paired bootstrap difference in C (each model vs M2 and vs M0) per trial, then pooled.
Clinical utility: decision-curve analysis for 6-month progression risk on pooled held-out predictions.

## Added value of blood tests (RQ1)
M2 vs M1 (Δ C-index, pooled) and likelihood-ratio test of blood block in the full development data.

## Missing data (RQ4)
Primary: single iterative (chained-equation) imputation fitted on development data, applied to validation data.
Robustness: (a) complete-case analysis; (b) native missing-value handling (XGBoost); (c) systematically
missing lymphocytes: NLR-augmented model developed in the four trials with lymphocytes and applied to LLY168
with lymphocytes missing (imputed vs dropped).

## Explainability (RQ5)
SHAP values for the best ML model and standardized hazard ratios for the Cox model fitted on all data;
agreement between rankings checked for clinical plausibility.

## Sub-analyses
Receptor status (LLY168 + SAN135); NLR (four trials; SII not analysed); OS as outcome (three trials); treatment-setting subgroups.

## Reporting
TRIPOD+AI. Software: Python 3.11, scikit-survival, lifelines, XGBoost, PyTorch. Seed 2026.

## Additional analyses added after the primary run (reported as post hoc sensitivity analyses)
- Trial-stratified Cox development (M2s); cubic-spline Cox (non-linearity); Schoenfeld-residual PH tests.
- Wider hyper-parameter grids (8 combinations each) for M4–M6 under the same nested IECV.
- Sample-size adequacy (Riley et al. 2020); discrimination by treatment setting; tertile risk groups from held-out M2 predictions.
- Corrections after independent code review (applied to all models, C-indices unaffected): (1) the score-to-risk
  calibration map is fitted on 5-fold out-of-fold development scores (fixed hyper-parameters) instead of in-sample
  scores; (2) IPCW weights for Uno's C, time-dependent AUC and IBS use the held-out trial's censoring distribution
  (development-data weights had made AUC inestimable for LLY168); (3) the reported final-model hazard-ratio table uses
  the same ridge penalty (0.01) as the exported model/calculator.
- SHAP values were computed for the gradient-boosted Cox model (exact TreeSHAP available); in the final results it had
  the highest pooled C among the non-linear ML models (0.591 vs random survival forest 0.588; DeepSurv 0.572);
  LASSO-Cox (0.592) is a penalised Cox model whose coefficients are directly interpretable.
- Of the lymphocyte-based indices, only NLR was analysed (PLR and SII were computed but not analysed).
- Data corrections before the final run (all analyses re-run): EFC6089 calcium values recorded in mg/dL but labelled
  mEq/L are no longer converted; calcium plausibility limits 5-16 mg/dL; EFC6089 investigator progressions without a
  dated response-page PD are retained (918 PFS events); pleural effusion counted with pleura as one organ site.
- Riley criteria computed with the optimism-adjusted Cox-Snell R2 (apparent R2 x van Houwelingen shrinkage).
