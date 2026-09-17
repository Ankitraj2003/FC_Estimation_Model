from pathlib import Path
import sys, runpy
ROOT=Path(__file__).resolve().parents[1]; CODE=ROOT/'Code'
for f in ['01_data_preprocessing.py','02_event_detection.py','03_fc_estimation_physics.py','04b_r_sax_cycle_analysis.py','04_fc_estimation_catboost.py','05_visualizations.py','06_generate_reports.py']:
    print('\n===',f,'==='); runpy.run_path(str(CODE/f),run_name='__main__')
print('\nPIPELINE COMPLETED SUCCESSFULLY')
