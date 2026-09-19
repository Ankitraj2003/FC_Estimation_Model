"""config.py - Authoritative Centralized Configuration for Phase 2.1 Pipeline

Preserves scientifically established baseline configuration without arbitrary tuning:
- Preprocessing: Short-gap linear interpolation (<= 2.0 h), SG smoothing (w=7, p=2) (Savitzky & Golay 1964)
- Hydrological Event Grouping: 24.0 h Inter-Event Time Definition (IETD)
- Cycle Screening: Bean et al. (2018) Criteria A-E (wetting threshold >= 0.006 m3/m3, rebound <= 0.002 m3/m3)
- Nocturnal Drainage Window: 22:00 to morning irrigation (07:00)
- SAX Symbolic Representation: Word length w=8, Alphabet size alpha=5 (Lin et al. 2007)
- CatBoost Residual Regressor: Depth=1, L2=10, LR=0.03, Trees=100 (a priori regularization)
- Final Holdout Event: Event 15 (chronological future-event holdout)
"""

from pathlib import Path

# Repository Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / 'Code'
DATA_DIR = ROOT_DIR / 'Data'
RESULTS_DIR = ROOT_DIR / 'Results'
PLOTS_DIR = ROOT_DIR / 'Plots'
REPORTS_DIR = ROOT_DIR / 'Reports'
MODELS_DIR = ROOT_DIR / 'Models'

INPUT_DATA_PATH = DATA_DIR / 'FINAL_VMC_Irrigation_Precipitation_Merged.xlsx'

# Ensure directories exist
for d in [RESULTS_DIR, PLOTS_DIR, REPORTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Scientifically Established Baseline Parameters
# -------------------------------------------------------------
# Preprocessing (Savitzky & Golay 1964)
MAX_INTERPOLATION_GAP_HOURS = 2.0
SG_WINDOW = 7
SG_POLYORDER = 2

# Hydrological Event Detection
IETD_HOURS = 24.0

# Bean et al. (2018) Cycle Screening
WETTING_THRESHOLD = 0.006
REBOUND_THRESHOLD = 0.002
NO_INPUT_START_HOUR = 22
DEFAULT_IRR_HOUR = 7

# Process-Based SWDP-R
MIN_SWPDR_POINTS = 4

# SAX Symbolic Discretization (Lin et al. 2007)
SAX_WORD_LENGTH = 8
SAX_ALPHABET_SIZE = 5
SAX_FEATURE_NAME = 'LOEO_SAX_Hamming_Similarity'

# CatBoost Regularized Hyperparameters (Set a priori)
CATBOOST_PARAMS = {
    'iterations': 100,
    'learning_rate': 0.03,
    'depth': 1,
    'l2_leaf_reg': 10,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False
}

RANDOM_SEED = 42

# Validation & Holdout Protocol (Roberts et al. 2017)
FINAL_HOLDOUT_EVENT = 15

# Standard Feature Set for Machine Learning
PREDICTOR_FEATURES = [
    'Depth_cm',
    'Pre_Event_VMC',
    'Peak_VMC',
    'Total_Water_mm',
    'VMC_Rise',
    'Initial_Drainage_Rate',
    'Peak_Hour',
    'Month',
    SAX_FEATURE_NAME
]

# Classification Regimes
VMC_REGIME_THRESHOLDS = [0.30, 0.35]
VMC_REGIME_LABELS = ['Low (<0.30)', 'Medium (0.30-0.35)', 'High (>=0.35)']
