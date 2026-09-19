"""06_sensitivity_and_ablation.py - Production-Based Ablation Study & One-Factor-at-a-Time Sensitivity Analysis

Follows Bean et al. (2018), Roberts et al. (2017), and Lin et al. (2007):
1. Model Ablation Study (LOEO-CV Validated with Nested Out-of-Sample SAX):
   - A1: SWDP-R Physics Baseline Only
   - A2: Direct CatBoost FC (Pure Machine Learning without Physics Baseline)
   - A3: Hybrid CatBoost without SAX Feature
   - A4: Full Hybrid CatBoost with Nested Out-of-Sample SAX
   - A5: Hybrid CatBoost with Minimal Early-Event Features Only
   Zero global SAX features are used in any ablation experiment.

2. One-Factor-at-a-Time (OFAT) Sensitivity Analysis:
   - Calls the exact production pipeline functions:
     Raw VMC -> Preprocessing -> Event Detection -> Valid-Cycle Screening -> Physics FC -> Nested Out-of-Sample SAX -> CatBoost -> LOEO-CV -> Holdout
   - Modifies strictly ONE parameter under investigation while keeping all baseline parameters fixed.
   - Experiments:
     * S1: Max Interpolation Gap: 0 h, 1 h, 2 h, 3 h (baseline: 2 h)
     * S2: Savitzky-Golay: (5,2), (7,2), (9,2) (baseline: 7, 2)
     * S3: IETD: 12 h, 24 h, 36 h, 48 h (baseline: 24 h)
     * S4: SAX Word Length: 6, 8, 10 (baseline: 8)
     * S5: SAX Alphabet Size: 4, 5, 6 (baseline: 5)
     * S6: Rebound Threshold: 0.001, 0.002, 0.003, 0.004 m3/m3 (baseline: 0.002)
   - Holdout Protection: Holdout metrics are recorded descriptively; no parameter selection is made via holdout.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from config import (
    RESULTS_DIR, MAX_INTERPOLATION_GAP_HOURS, SG_WINDOW, SG_POLYORDER,
    IETD_HOURS, WETTING_THRESHOLD, REBOUND_THRESHOLD,
    SAX_WORD_LENGTH, SAX_ALPHABET_SIZE, SAX_FEATURE_NAME,
    CATBOOST_PARAMS, FINAL_HOLDOUT_EVENT, PREDICTOR_FEATURES
)

# Import exact production functions
from importlib import import_module
prep_mod = import_module('01_data_preprocessing')
event_mod = import_module('02_event_detection')
phys_mod = import_module('03_fc_estimation_physics')
sax_mod = import_module('04b_r_sax_cycle_analysis')
cb_mod = import_module('04_fc_estimation_catboost')

compute_metrics = cb_mod.compute_metrics
build_base_feature_dataset = cb_mod.build_base_feature_dataset
run_catboost_loeo_and_test = cb_mod.run_catboost_loeo_and_test
construct_nested_fold_sax = cb_mod.construct_nested_fold_sax
extract_cycle_sax_words = cb_mod.extract_cycle_sax_words


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Ablation Study (LOEO-CV Validated with Nested Out-of-Sample SAX)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study():
    df_clean = pd.read_csv(RESULTS_DIR / 'cleaned_vmc_data.csv')
    df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'], format='mixed')
    fc = pd.read_csv(RESULTS_DIR / 'physics_fc_event_estimates.csv')
    cy = pd.read_csv(RESULTS_DIR / 'valid_recession_cycles.csv')
    cy_valid = cy[cy['Is_Valid'].astype(bool)].copy()

    ml, base_feature_cols = build_base_feature_dataset(df_clean, fc, cy)

    train_mask = ml['Event_ID'] != FINAL_HOLDOUT_EVENT
    test_mask = ml['Event_ID'] == FINAL_HOLDOUT_EVENT
    y_target = ml['FC_Direct_SensorReference']
    swdp_r = ml['FC_SWDP_R']

    early_feature_cols = ['Depth_cm', 'Pre_Event_VMC', 'Peak_VMC', 'Total_Water_mm', 'VMC_Rise']

    # Pre-extract cycle SAX words
    cycle_words = extract_cycle_sax_words(df_clean, cy_valid, word_length=SAX_WORD_LENGTH, alphabet_size=SAX_ALPHABET_SIZE)

    ablation_configs = [
        ('A1_SWDP_R_Only', 'baseline', None, False),
        ('A2_Direct_CatBoost', 'direct_ml', base_feature_cols, True),
        ('A3_Hybrid_No_SAX', 'residual', base_feature_cols, False),
        ('A4_Full_Hybrid_With_SAX', 'residual', base_feature_cols, True),
        ('A5_Hybrid_Early_Features', 'residual', early_feature_cols, False)
    ]

    records = []
    baseline_test_rmse = None

    for code, mtype, fcols, inc_sax in ablation_configs:
        if mtype == 'baseline':
            pred_test = swdp_r.loc[test_mask].to_numpy()
            pred_train = swdp_r.loc[train_mask].to_numpy()

            event_order = ml.groupby('Event_ID')['Peak_Time'].min().sort_values().index.tolist()
            train_events = [e for e in event_order if e != FINAL_HOLDOUT_EVENT]
            loeo_rmses_s = []
            for val_eid in train_events:
                fold_val = ml['Event_ID'] == val_eid
                y_val = y_target.loc[fold_val].to_numpy()
                s_val = swdp_r.loc[fold_val].to_numpy()
                _, rmse_s, _, _ = compute_metrics(y_val, s_val)
                loeo_rmses_s.append(rmse_s)
            loeo_med_h = float(np.median(loeo_rmses_s)) if loeo_rmses_s else np.nan
            loeo_med_s = loeo_med_h
            f_count = 0
            f_names = 'None (Physics Only)'

        else:
            _, comp_df, loeo_df, _, final_model, _, used_fcols = run_catboost_loeo_and_test(
                ml=ml.copy(),
                df_clean=df_clean,
                cy_valid=cy_valid,
                base_feature_cols=fcols,
                include_sax=inc_sax,
                holdout_event_id=FINAL_HOLDOUT_EVENT
            )
            f_count = len(used_fcols)
            f_names = ', '.join(used_fcols)

            if mtype == 'direct_ml':
                target_row = comp_df[comp_df['Model'].str.contains('Direct CatBoost')].iloc[0]
                loeo_med_h = float(loeo_df['Direct_CatBoost_RMSE'].median())
                loeo_med_s = float(loeo_df['SWDP_R_RMSE'].median())
            else:
                target_row = comp_df[comp_df['Model'].str.contains('Hybrid')].iloc[0]
                loeo_med_h = float(loeo_df['Hybrid_RMSE'].median())
                loeo_med_s = float(loeo_df['SWDP_R_RMSE'].median())

            r2_te = float(target_row['R2'])
            rmse_te = float(target_row['RMSE'])
            mae_te = float(target_row['MAE'])
            bias_te = float(target_row['Bias'])

        if mtype == 'baseline':
            r2_te, rmse_te, mae_te, bias_te = compute_metrics(y_target.loc[test_mask], pred_test)
            baseline_test_rmse = rmse_te

        if baseline_test_rmse is None:
            baseline_test_rmse = rmse_te

        records.append({
            'Ablation_Code': code,
            'Model_Architecture': mtype,
            'Feature_Count': f_count,
            'Features_Used': f_names,
            'Test_R2': r2_te,
            'Test_RMSE': rmse_te,
            'Test_MAE': mae_te,
            'Test_Bias': bias_te,
            'LOEO_CV_Median_Hybrid_RMSE': loeo_med_h,
            'LOEO_CV_Median_SWDP_R_RMSE': loeo_med_s,
            'Delta_RMSE_vs_SWDP_R': rmse_te - baseline_test_rmse
        })

    abl_df = pd.DataFrame(records)
    abl_df.to_csv(RESULTS_DIR / 'ablation_study_results.csv', index=False)
    print("Ablation study complete (LOEO-CV validated with Nested Out-of-Sample SAX). Saved to Results/ablation_study_results.csv")
    print(abl_df[['Ablation_Code', 'Test_R2', 'Test_RMSE', 'LOEO_CV_Median_Hybrid_RMSE', 'Delta_RMSE_vs_SWDP_R']].to_string(index=False))
    return abl_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Production One-Factor-at-a-Time (OFAT) Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _run_full_production_pipeline(max_gap_hours: float = MAX_INTERPOLATION_GAP_HOURS,
                                  sg_window: int = SG_WINDOW,
                                  sg_poly: int = SG_POLYORDER,
                                  ietd_hours: float = IETD_HOURS,
                                  wetting_thresh: float = WETTING_THRESHOLD,
                                  rebound_thresh: float = REBOUND_THRESHOLD,
                                  sax_w: int = SAX_WORD_LENGTH,
                                  sax_a: int = SAX_ALPHABET_SIZE):
    """Executes the exact production pipeline functions under one OFAT configuration."""
    # Step 1: Preprocessing
    df_clean = prep_mod.run_preprocessing(
        max_gap_hours=max_gap_hours,
        sg_window=sg_window,
        sg_poly=sg_poly,
        save_outputs=False
    )

    # Step 2: Event detection & screening
    events_df, cycles_df = event_mod.run_event_detection(
        ietd_hours=ietd_hours,
        wetting_thresh=wetting_thresh,
        rebound_thresh=rebound_thresh,
        df_clean=df_clean,
        save_outputs=False
    )

    cand_events = len(events_df)
    valid_cycles = cycles_df[cycles_df['Is_Valid'].astype(bool)] if not cycles_df.empty else pd.DataFrame()
    valid_events = len(valid_cycles['Event_ID'].unique()) if not valid_cycles.empty else 0
    rejected_events = cand_events - valid_events

    if len(valid_cycles) < 3 or FINAL_HOLDOUT_EVENT not in valid_cycles['Event_ID'].values:
        return {
            'candidate_events': cand_events,
            'valid_events': valid_events,
            'rejected_events': rejected_events,
            'fc_estimates': len(valid_cycles),
            'n_depth_event_observations': len(valid_cycles),
            'loeo_rmse': np.nan,
            'loeo_mae': np.nan,
            'loeo_r2': np.nan,
            'holdout_rmse': np.nan,
            'holdout_mae': np.nan,
            'holdout_r2': np.nan,
            'bias': np.nan,
            'n_predictions': 0
        }

    # Step 3: Physics FC estimation
    fc_df, _ = phys_mod.run_physics_fc_estimation(
        df_clean=df_clean,
        cycles_df=cycles_df,
        save_outputs=False
    )

    # Step 4: SAX descriptive generation (retained for diagnostics)
    sax_mod.run_sax_cycle_analysis(
        df_clean=df_clean,
        cycles_df=cycles_df,
        word_size=sax_w,
        alphabet_size=sax_a,
        holdout_event_id=FINAL_HOLDOUT_EVENT,
        save_outputs=False
    )

    # Step 5: Base feature dataset and Nested Out-of-Sample SAX CatBoost evaluation
    ml, base_feature_cols = build_base_feature_dataset(df_clean, fc_df, cycles_df)

    if FINAL_HOLDOUT_EVENT not in ml['Event_ID'].values or len(ml['Event_ID'].unique()) < 2:
        return {
            'candidate_events': cand_events,
            'valid_events': valid_events,
            'rejected_events': rejected_events,
            'fc_estimates': len(fc_df),
            'n_depth_event_observations': len(ml),
            'loeo_rmse': np.nan,
            'loeo_mae': np.nan,
            'loeo_r2': np.nan,
            'holdout_rmse': np.nan,
            'holdout_mae': np.nan,
            'holdout_r2': np.nan,
            'bias': np.nan,
            'n_predictions': 0
        }

    ml, comp_df, loeo_df, _, _, _, _ = run_catboost_loeo_and_test(
        ml=ml, df_clean=df_clean, cy_valid=valid_cycles,
        base_feature_cols=base_feature_cols,
        include_sax=True,
        holdout_event_id=FINAL_HOLDOUT_EVENT,
        word_length=sax_w,
        alphabet_size=sax_a
    )

    hy_row = comp_df[comp_df['Model'].str.contains('Hybrid')].iloc[0]

    return {
        'candidate_events': cand_events,
        'valid_events': valid_events,
        'rejected_events': rejected_events,
        'fc_estimates': len(fc_df),
        'n_depth_event_observations': len(ml),
        'loeo_rmse': float(loeo_df['Hybrid_RMSE'].median()) if not loeo_df.empty else np.nan,
        'loeo_mae': float(loeo_df['Hybrid_MAE'].median()) if not loeo_df.empty else np.nan,
        'loeo_r2': float(loeo_df['Hybrid_R2'].median()) if not loeo_df.empty else np.nan,
        'holdout_rmse': float(hy_row['RMSE']),
        'holdout_mae': float(hy_row['MAE']),
        'holdout_r2': float(hy_row['R2']),
        'bias': float(hy_row['Bias']),
        'n_predictions': len(ml)
    }


def run_sensitivity_analysis():
    """Runs One-Factor-at-a-Time (OFAT) sensitivity analysis using production functions with Nested SAX."""
    print("Starting One-Factor-at-a-Time (OFAT) Sensitivity Analysis using exact production pipeline...")
    records = []

    # 1. Max Interpolation Gap: 0 h, 1 h, 2 h, 3 h (baseline: 2 h)
    for gap in [0.0, 1.0, 2.0, 3.0]:
        print(f"  Testing Max Interpolation Gap: {gap} h ...")
        res = _run_full_production_pipeline(max_gap_hours=gap)
        records.append({
            'parameter': 'max_interpolation_gap',
            'parameter_value': f"{gap} h",
            **res
        })

    # 2. Savitzky-Golay: (5,2), (7,2), (9,2) (baseline: 7, 2)
    for sg_w, sg_p in [(5, 2), (7, 2), (9, 2)]:
        print(f"  Testing Savitzky-Golay Smoothing: window={sg_w}, poly={sg_p} ...")
        res = _run_full_production_pipeline(sg_window=sg_w, sg_poly=sg_p)
        records.append({
            'parameter': 'savitzky_golay_smoothing',
            'parameter_value': f"w={sg_w}, p={sg_p}",
            **res
        })

    # 3. IETD: 12 h, 24 h, 36 h, 48 h (baseline: 24 h)
    for ietd in [12.0, 24.0, 36.0, 48.0]:
        print(f"  Testing IETD: {ietd} h ...")
        res = _run_full_production_pipeline(ietd_hours=ietd)
        records.append({
            'parameter': 'ietd_hours',
            'parameter_value': f"{ietd} h",
            **res
        })

    # 4. SAX Word Length: 6, 8, 10 (baseline: 8)
    for wl in [6, 8, 10]:
        print(f"  Testing SAX Word Length: w={wl} ...")
        res = _run_full_production_pipeline(sax_w=wl)
        records.append({
            'parameter': 'sax_word_length',
            'parameter_value': str(wl),
            **res
        })

    # 5. SAX Alphabet Size: 4, 5, 6 (baseline: 5)
    for az in [4, 5, 6]:
        print(f"  Testing SAX Alphabet Size: alpha={az} ...")
        res = _run_full_production_pipeline(sax_a=az)
        records.append({
            'parameter': 'sax_alphabet_size',
            'parameter_value': str(az),
            **res
        })

    # 6. Nocturnal Rebound Threshold: 0.001, 0.002, 0.003, 0.004 (baseline: 0.002)
    for reb in [0.001, 0.002, 0.003, 0.004]:
        print(f"  Testing Rebound Tolerance: {reb} m3/m3 ...")
        res = _run_full_production_pipeline(rebound_thresh=reb)
        records.append({
            'parameter': 'rebound_threshold',
            'parameter_value': f"{reb} m3/m3",
            **res
        })

    sens_df = pd.DataFrame(records)
    out_path = RESULTS_DIR / 'sensitivity_analysis_results.csv'
    sens_df.to_csv(out_path, index=False)
    print(f"OFAT Sensitivity analysis complete. Saved to {out_path}")
    print(sens_df[['parameter', 'parameter_value', 'valid_events', 'fc_estimates', 'loeo_rmse', 'holdout_rmse']].to_string(index=False))
    return sens_df


if __name__ == '__main__':
    run_ablation_study()
    run_sensitivity_analysis()
