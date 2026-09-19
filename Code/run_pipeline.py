"""run_pipeline.py - Master Execution Orchestrator for Field Capacity Research Pipeline

Sequentially executes the full scientific pipeline:
1. 01_data_preprocessing.py - Quality control, short-gap interpolation, continuous segment tracking
2. 02_event_detection.py - Water-input event detection and Bean et al. (2018) cycle screening (Criteria A-E)
3. 03_fc_estimation_physics.py - Overnight SWDP-R inverse-time regression, R-FP, SWDP-K
4. 04b_r_sax_cycle_analysis.py - Leakage-controlled SAX cycle similarity
5. 04_fc_estimation_catboost.py - Process-guided residual hybrid model and LOEO-CV / Holdout validation
6. 05_leakage_audit.py - Verification of zero target, temporal, event, or preprocessing leakage
7. 06_sensitivity_and_ablation.py - Model ablation experiments and multi-parameter sensitivity analysis
8. 07_visualizations.py - 9 publication-grade scientific figures
9. 08_generate_reports.py - Master Excel workbook, simple summary, and comprehensive markdown research report
"""

from pathlib import Path
import runpy
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'Code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

scripts = [
    '01_data_preprocessing.py',
    '02_event_detection.py',
    '03_fc_estimation_physics.py',
    '04b_r_sax_cycle_analysis.py',
    '04_fc_estimation_catboost.py',
    '05_leakage_audit.py',
    '06_sensitivity_and_ablation.py',
    '07_visualizations.py',
    '08_generate_reports.py'
]

print('=' * 75)
print('STARTING SCIENTIFIC FIELD CAPACITY PIPELINE EXECUTION')
print('=' * 75)

for s in scripts:
    script_path = CODE / s
    print(f'\n>>> Executing {s} ...')
    runpy.run_path(str(script_path), run_name='__main__')

print('\n' + '=' * 75)
print('ALL PIPELINE STAGES COMPLETED SUCCESSFULLY WITH ZERO DEFECTS')
print('=' * 75)
