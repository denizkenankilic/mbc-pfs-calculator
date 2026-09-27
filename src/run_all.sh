#!/bin/sh
# Full pipeline (order matters)
set -e
python3 build_dataset.py
python3 descriptives.py
python3 run_iecv.py --outcome pfs
python3 run_iecv.py --outcome os
python3 run_subanalyses.py
python3 run_tuning_sensitivity.py
python3 run_extra.py
python3 build_calculator.py
python3 figures.py
python3 manuscript_figures.py
python3 make_results_tables.py
python3 make_supplement.py
node build_supplement_docx.js ../outputs/manuscript
