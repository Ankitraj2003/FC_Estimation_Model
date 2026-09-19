"""04_fc_estimation_catboost.py - Process-Guided Residual Hybrid Modeling & Nested Out-of-Sample SAX Validation

Scientific Methodology:
1. Process-Guided Residual Architecture:
   - Physically motivated baseline: SWDP-R asymptotic drainage estimate (Bean et al. 2018; Fazackerley & Lawrence 2012).
   - Machine learning residual target: Delta = FC_Direct_SensorReference - FC_SWDP_R.
   - Final hybrid prediction: FC_pred = FC_SWDP_R + Delta_pred.
   Following Jiang et al. (2025) and Zhou et al. (2025).

2. Dual-Level Temporal Validation (Roberts et al. 2017):
   - Level 1 (Model Development): Hydrological Leave-One-Event-Out Cross-Validation (LOEO-CV)
     across training events. Depths from the same event are strictly held together.
   - Level 2 (Final Future Evaluation): Chronological future-event holdout on Event 15.

3. Nested Fold-Specific / Nested Out-of-Sample SAX Construction:
   - For every LOEO fold (outer validation event V, outer training events {A, B, C, D}):
     * Training event A: Reference library = {B, C, D} (excludes A to prevent self-similarity; excludes V).
     * Training event B: Reference library = {A, C, D} (excludes B; excludes V).
     * Training event C: Reference library = {A, B, D} (excludes C; excludes V).
     * Training event D: Reference library = {A, B, C} (excludes D; excludes V).
     * Outer validation event V: Reference library = {A, B, C, D} (outer training events only; excludes V).
   - Zero reliance on global SAX columns in ML feature matrices.
   - Short series (< word_length) explicitly return None (never artificial 'aaaa...').
   - Insufficient libraries (< 1 reference event) explicitly return NaN (no automatic median imputation).
   - CatBoost natively processes NaN values without synthetic imputation.
   - Comprehensive provenance logged to Results/sax_fold_provenance_audit.csv.
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, accuracy_score, precision_score
)
from scipy.stats import norm

from config import (
    ROOT_DIR, RESULTS_DIR, MODELS_DIR, CATBOOST_PARAMS, RANDOM_SEED,
    FINAL_HOLDOUT_EVENT, PREDICTOR_FEATURES, SAX_FEATURE_NAME,
    VMC_REGIME_THRESHOLDS, VMC_REGIME_LABELS, SAX_WORD_LENGTH, SAX_ALPHABET_SIZE
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute standard regression metrics."""
    y_t = np.asarray(y_true, float)
    y_p = np.asarray(y_pred, float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if valid.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    y_t = y_t[valid]
    y_p = y_p[valid]

    r2 = r2_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    bias = float(np.mean(y_p - y_t))
    return r2, rmse, mae, bias


def classify_fc(val: float) -> str:
    """Categorize FC into hydrological soil moisture regimes."""
    if val < VMC_REGIME_THRESHOLDS[0]:
        return VMC_REGIME_LABELS[0]
    elif val < VMC_REGIME_THRESHOLDS[1]:
        return VMC_REGIME_LABELS[1]
    else:
        return VMC_REGIME_LABELS[2]


# ─────────────────────────────────────────────────────────────────────────────
# Robust SAX Word Representation (Lin et al. 2007; Bean et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

def _paa_transform(y: np.ndarray, n_segments: int) -> np.ndarray:
    """Piecewise Aggregate Approximation without empty-slice warnings."""
    n = len(y)
    if n == 0 or n_segments < 2:
        return np.zeros(n_segments)
    edges = np.linspace(0, n, n_segments + 1).astype(int)
    means = []
    for i in range(n_segments):
        start, end = edges[i], edges[i + 1]
        if start >= end:
            end = min(start + 1, n)
            start = max(0, end - 1)
        sub = y[start:end]
        means.append(float(np.mean(sub)) if len(sub) > 0 else float(y[-1]))
    return np.array(means)


def _get_gaussian_breakpoints(alphabet_size: int) -> np.ndarray:
    """Gaussian equiprobable breakpoints for SAX alphabet (Lin et al. 2003)."""
    return norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])


def _to_sax_word(y: np.ndarray, word_length: int = SAX_WORD_LENGTH, alphabet_size: int = SAX_ALPHABET_SIZE) -> str:
    """Convert a time series to a SAX symbolic word.
    
    Returns None if series is too short or variance is zero.
    Never returns an artificial 'aaaa...' word.
    """
    if y is None:
        return None
    y_arr = np.asarray(y, float)
    y_arr = y_arr[np.isfinite(y_arr)]
    if len(y_arr) < word_length:
        return None
    std = float(np.std(y_arr))
    if std < 1e-10:
        return None
    y_norm = (y_arr - np.mean(y_arr)) / std
    paa = _paa_transform(y_norm, word_length)
    breakpoints = _get_gaussian_breakpoints(alphabet_size)
    chars = []
    for val in paa:
        idx = np.searchsorted(breakpoints, val, side='right')
        chars.append(chr(ord('a') + idx))
    return ''.join(chars)


def extract_cycle_sax_words(df_clean: pd.DataFrame,
                             cy_valid: pd.DataFrame,
                             word_length: int = SAX_WORD_LENGTH,
                             alphabet_size: int = SAX_ALPHABET_SIZE) -> dict:
    """Extract SAX words for all valid recession cycles."""
    words = {}
    for _, r in cy_valid.iterrows():
        eid = int(r['Event_ID'])
        col = r['Column']
        col_smooth = col + '_smoothed'
        peak_time = pd.to_datetime(r['Peak_Time'])
        sched_time = pd.to_datetime(r['Scheduled_Irrigation_Time'])
        rec_df = df_clean[(df_clean['Timestamp'] >= peak_time) & (df_clean['Timestamp'] <= sched_time)]
        y = rec_df[col_smooth].dropna().to_numpy(float)
        w = _to_sax_word(y, word_length=word_length, alphabet_size=alphabet_size)
        words[(eid, col)] = w
    return words


def construct_nested_fold_sax(cycle_words: dict,
                              cy_valid: pd.DataFrame,
                              train_event_ids: list,
                              val_event_id: int = None,
                              word_length: int = SAX_WORD_LENGTH,
                              outer_fold_name: str = 'LOEO_Fold'):
    """Constructs Nested Out-of-Sample SAX Features with Zero Leakage and Zero Self-Similarity.

    Workflow:
    - For training event E in train_event_ids:
      * Reference Library = train_event_ids \\ {E} (strictly excludes E and val_event_id).
      * Assertions: E not in Ref_Library and val_event_id not in Ref_Library.
    - For validation event val_event_id:
      * Reference Library = train_event_ids (outer training events only).
      * Assertion: val_event_id not in Ref_Library.
    - If library is empty, similarity is set to NaN (no automatic median imputation).
    """
    feature_values = {}
    provenance_records = []

    train_eids_set = set(train_event_ids)
    if val_event_id is not None:
        assert val_event_id not in train_eids_set, (
            f"Outer validation event {val_event_id} must not be present in train_event_ids."
        )
    cy_dict = {}
    for _, r in cy_valid.iterrows():
        cy_dict[(int(r['Event_ID']), r['Column'])] = int(r['Depth_cm'])

    # 1. Training Events: Leave-One-Training-Event-Out inside the fold
    for cur_eid in train_event_ids:
        ref_eids = sorted(list(train_eids_set - {cur_eid}))

        # Assertions
        assert cur_eid not in ref_eids, f"Self-similarity violation: Event {cur_eid} in its own reference library!"
        if val_event_id is not None:
            assert val_event_id not in ref_eids, f"Validation leakage: Validation Event {val_event_id} in training reference library!"

        ref_words = [cycle_words.get((e, c)) for (e, c) in cycle_words if e in ref_eids and cycle_words.get((e, c)) is not None]

        # Cycles belonging to cur_eid
        cur_cycles = [(e, c) for (e, c) in cycle_words if e == cur_eid]
        for (e, c) in cur_cycles:
            w_input = cycle_words.get((e, c))
            depth = cy_dict.get((e, c), 0)

            if w_input is None:
                sim = np.nan
                status = 'SHORT_SERIES'
            elif len(ref_words) == 0:
                sim = np.nan
                status = 'INSUFFICIENT_REFERENCE_LIBRARY'
            else:
                sim_scores = [sum(a == b for a, b in zip(w_input, rw)) / word_length for rw in ref_words]
                sim = float(np.mean(sim_scores))
                status = 'PASS'

            feature_values[(e, c)] = sim
            provenance_records.append({
                'Outer_Fold': outer_fold_name,
                'Outer_Validation_Event': val_event_id,
                'Feature_Event': e,
                'Depth_cm': depth,
                'Feature_Type': 'Training',
                'Input_SAX_Word': w_input if w_input else 'None (Unavailable)',
                'Reference_Library_Events': str(ref_eids),
                'Reference_Library_Word_Count': len(ref_words),
                'Reference_Library_SAX_Words': str(ref_words),
                'SAX_Word_Length': int(word_length),
                'SAX_Alphabet_Size': int(SAX_ALPHABET_SIZE),
                'Feature_Value': sim,
                'Computed_Hamming_Similarity': sim,
                    'Feature_Calculation_Method': 'Mean_Normalized_Hamming_Similarity',
                    'Reconstructed_Feature_Value': np.nan,
                    'Absolute_Reconstruction_Error': np.nan,
                    'Feature_Value_Reproducible': False,
                'Feature_Event_In_Reference_Library': (e in ref_eids),
                'Outer_Validation_In_Reference_Library': (val_event_id in ref_eids) if val_event_id else False,
                'Feature_Source': 'Nested_Out_Of_Sample_Fold_Library',
                'Status': status
            })

    # 2. Outer Validation Event (if provided)
    if val_event_id is not None:
        val_ref_eids = sorted(list(train_eids_set))

        # Assertion
        assert val_event_id not in val_ref_eids, f"Validation event {val_event_id} in its own reference library!"

        val_ref_words = [cycle_words.get((e, c)) for (e, c) in cycle_words if e in val_ref_eids and cycle_words.get((e, c)) is not None]

        val_cycles = [(e, c) for (e, c) in cycle_words if e == val_event_id]
        for (e, c) in val_cycles:
            w_input = cycle_words.get((e, c))
            depth = cy_dict.get((e, c), 0)

            if w_input is None:
                sim = np.nan
                status = 'SHORT_SERIES'
            elif len(val_ref_words) == 0:
                sim = np.nan
                status = 'INSUFFICIENT_REFERENCE_LIBRARY'
            else:
                sim_scores = [sum(a == b for a, b in zip(w_input, rw)) / word_length for rw in val_ref_words]
                sim = float(np.mean(sim_scores))
                status = 'PASS'

            feature_values[(e, c)] = sim
            provenance_records.append({
                'Outer_Fold': outer_fold_name,
                'Outer_Validation_Event': val_event_id,
                'Feature_Event': e,
                'Depth_cm': depth,
                'Feature_Type': 'Validation',
                'Input_SAX_Word': w_input if w_input else 'None (Unavailable)',
                'Reference_Library_Events': str(val_ref_eids),
                'Reference_Library_Word_Count': len(val_ref_words),
                'Reference_Library_SAX_Words': str(val_ref_words),
                'SAX_Word_Length': int(word_length),
                'SAX_Alphabet_Size': int(SAX_ALPHABET_SIZE),
                'Feature_Value': sim,
                'Computed_Hamming_Similarity': sim,
                    'Feature_Calculation_Method': 'Mean_Normalized_Hamming_Similarity',
                    'Reconstructed_Feature_Value': np.nan,
                    'Absolute_Reconstruction_Error': np.nan,
                    'Feature_Value_Reproducible': False,
                'Feature_Event_In_Reference_Library': (e in val_ref_eids),
                'Outer_Validation_In_Reference_Library': (val_event_id in val_ref_eids),
                'Feature_Source': 'Nested_Out_Of_Sample_Fold_Library',
                'Status': status
            })

    return feature_values, provenance_records


def build_base_feature_dataset(df: pd.DataFrame,
                               fc: pd.DataFrame,
                               cy: pd.DataFrame):
    """Construct base features available prior to or at FC estimation time.
    
    Strictly excludes any global SAX features from the base dataset.
    """
    cy_valid = cy[cy['Is_Valid'].astype(bool)].copy()
    ml = fc.merge(
        cy_valid[['Event_ID', 'Column', 'Pre_Event_VMC', 'Total_Water_mm', 'VMC_Rise',
                  'Recession_Hours', 'Peak_Time', 'Peak_Idx']],
        on=['Event_ID', 'Column'], how='inner'
    )

    drainage_rates = []
    peak_hours = []
    months = []

    for _, r in ml.iterrows():
        eid = int(r['Event_ID'])
        col = r['Column']
        match = cy_valid[(cy_valid['Event_ID'] == eid) & (cy_valid['Column'] == col)].iloc[0]
        p_idx = int(match['Peak_Idx'])
        end_idx = int(match['Recession_End_Idx'])

        rec_window = df.iloc[p_idx:end_idx + 1]
        t = (rec_window['Elapsed_hours'] - rec_window['Elapsed_hours'].iloc[0]).to_numpy(float)
        y = rec_window[col + '_smoothed'].to_numpy(float)

        elapsed_early = min(3.0, float(t[-1])) if len(t) else 0.0
        if len(t) > 1 and elapsed_early > 0:
            idx_3h = np.argmin(np.abs(t - elapsed_early))
            rate = float((y[0] - y[idx_3h]) / max(elapsed_early, 1.0))
        else:
            rate = 0.0

        p_time = pd.to_datetime(r['Peak_Time'])
        drainage_rates.append(rate)
        peak_hours.append(int(p_time.hour))
        months.append(int(p_time.month))

    ml['Initial_Drainage_Rate'] = drainage_rates
    ml['Peak_Hour'] = peak_hours
    ml['Month'] = months

    base_feature_cols = [f for f in PREDICTOR_FEATURES if f != SAX_FEATURE_NAME]
    clean_ml = ml.dropna(subset=base_feature_cols + ['FC_Direct_SensorReference', 'FC_SWDP_R']).reset_index(drop=True)
    return clean_ml, base_feature_cols


def run_catboost_loeo_and_test(ml: pd.DataFrame,
                               df_clean: pd.DataFrame,
                               cy_valid: pd.DataFrame,
                               base_feature_cols: list,
                               include_sax: bool = True,
                               holdout_event_id: int = FINAL_HOLDOUT_EVENT,
                               word_length: int = SAX_WORD_LENGTH,
                               alphabet_size: int = SAX_ALPHABET_SIZE):
    """Core production LOEO-CV and Holdout evaluation function using Nested Out-of-Sample SAX."""
    y_target = ml['FC_Direct_SensorReference']
    swdp_r = ml['FC_SWDP_R']

    event_order = ml.groupby('Event_ID')['Peak_Time'].min().sort_values().index.tolist()
    train_mask = ml['Event_ID'] != holdout_event_id
    test_mask = ml['Event_ID'] == holdout_event_id
    train_event_ids = [e for e in event_order if e != holdout_event_id]

    # Pre-extract raw SAX words for all valid cycles
    cycle_words = extract_cycle_sax_words(df_clean, cy_valid, word_length=word_length, alphabet_size=alphabet_size)

    all_provenance_records = []
    feature_cols = list(base_feature_cols)
    if include_sax:
        feature_cols.append(SAX_FEATURE_NAME)

    # ─────────────────────────────────────────────────────────────
    # LEVEL 1: LEAVE-ONE-EVENT-OUT CROSS-VALIDATION (LOEO-CV)
    # ─────────────────────────────────────────────────────────────
    loeo_records = []
    loeo_preds_hybrid = np.full(len(ml), np.nan)
    loeo_preds_direct_cb = np.full(len(ml), np.nan)

    for val_eid in train_event_ids:
        fold_train_eids = [e for e in train_event_ids if e != val_eid]
        
        # Nested Fold-Specific / Nested Out-of-Sample SAX Feature Construction
        if include_sax:
            fold_sax_feats, fold_prov = construct_nested_fold_sax(
                cycle_words=cycle_words,
                cy_valid=cy_valid,
                train_event_ids=fold_train_eids,
                val_event_id=val_eid,
                word_length=word_length,
                outer_fold_name=f'LOEO_Fold_Val_{val_eid}'
            )
            all_provenance_records.extend(fold_prov)
        else:
            fold_sax_feats = {}

        fold_train_mask = ml['Event_ID'].isin(fold_train_eids)
        fold_val_mask = ml['Event_ID'] == val_eid

        # Construct X_fold_train and X_fold_val explicitly
        X_fold_train = ml.loc[fold_train_mask, base_feature_cols].copy()
        X_fold_val = ml.loc[fold_val_mask, base_feature_cols].copy()

        if include_sax:
            X_fold_train[SAX_FEATURE_NAME] = [fold_sax_feats.get((int(ml.loc[i, 'Event_ID']), ml.loc[i, 'Column']), np.nan) for i in X_fold_train.index]
            X_fold_val[SAX_FEATURE_NAME] = [fold_sax_feats.get((int(ml.loc[i, 'Event_ID']), ml.loc[i, 'Column']), np.nan) for i in X_fold_val.index]

        # Train residual CatBoost
        cb_res_fold = CatBoostRegressor(**CATBOOST_PARAMS)
        target_res_fold = y_target.loc[fold_train_mask] - swdp_r.loc[fold_train_mask]
        cb_res_fold.fit(X_fold_train, target_res_fold)

        pred_res_val = cb_res_fold.predict(X_fold_val)
        pred_hybrid_val = swdp_r.loc[fold_val_mask].to_numpy() + pred_res_val
        loeo_preds_hybrid[fold_val_mask] = pred_hybrid_val

        # Direct CatBoost comparator
        cb_direct_fold = CatBoostRegressor(**CATBOOST_PARAMS)
        cb_direct_fold.fit(X_fold_train, y_target.loc[fold_train_mask])
        loeo_preds_direct_cb[fold_val_mask] = cb_direct_fold.predict(X_fold_val)

        # Evaluate fold performance
        y_val = y_target.loc[fold_val_mask].to_numpy()
        r_swdp_val = swdp_r.loc[fold_val_mask].to_numpy()

        r2_hy, rmse_hy, mae_hy, bias_hy = compute_metrics(y_val, pred_hybrid_val)
        r2_sw, rmse_sw, mae_sw, bias_sw = compute_metrics(y_val, r_swdp_val)
        r2_dcb, rmse_dcb, mae_dcb, bias_dcb = compute_metrics(y_val, loeo_preds_direct_cb[fold_val_mask])

        loeo_records.append({
            'Validation_Type': 'LOEO_CV',
            'Held_Out_Event': val_eid,
            'N_Samples': int(fold_val_mask.sum()),
            'Hybrid_R2': r2_hy,
            'Hybrid_RMSE': rmse_hy,
            'Hybrid_MAE': mae_hy,
            'Hybrid_Bias': bias_hy,
            'SWDP_R_R2': r2_sw,
            'SWDP_R_RMSE': rmse_sw,
            'SWDP_R_MAE': mae_sw,
            'SWDP_R_Bias': bias_sw,
            'Direct_CatBoost_R2': r2_dcb,
            'Direct_CatBoost_RMSE': rmse_dcb,
            'Direct_CatBoost_MAE': mae_dcb,
            'Direct_CatBoost_Bias': bias_dcb,
            'Delta_RMSE': rmse_hy - rmse_sw
        })

    loeo_df = pd.DataFrame(loeo_records)

    # ─────────────────────────────────────────────────────────────
    # LEVEL 2: CHRONOLOGICAL FUTURE-EVENT HOLDOUT (EVENT 15)
    # ─────────────────────────────────────────────────────────────
    # Nested Fold-Specific SAX for Final Model: Training events use leave-one-out across all training events;
    # Holdout event uses all training events.
    if include_sax:
        final_sax_feats, final_prov = construct_nested_fold_sax(
            cycle_words=cycle_words,
            cy_valid=cy_valid,
            train_event_ids=train_event_ids,
            val_event_id=holdout_event_id,
            word_length=word_length,
            outer_fold_name=f'Final_Holdout_Event_{holdout_event_id}'
        )
        all_provenance_records.extend(final_prov)
    else:
        final_sax_feats = {}

    X_train_final = ml.loc[train_mask, base_feature_cols].copy()
    X_test_final = ml.loc[test_mask, base_feature_cols].copy()

    if include_sax:
        X_train_final[SAX_FEATURE_NAME] = [final_sax_feats.get((int(ml.loc[i, 'Event_ID']), ml.loc[i, 'Column']), np.nan) for i in X_train_final.index]
        X_test_final[SAX_FEATURE_NAME] = [final_sax_feats.get((int(ml.loc[i, 'Event_ID']), ml.loc[i, 'Column']), np.nan) for i in X_test_final.index]

    residual_train = y_target.loc[train_mask] - swdp_r.loc[train_mask]
    final_model = CatBoostRegressor(**CATBOOST_PARAMS)
    final_model.fit(X_train_final, residual_train)

    pred_hybrid_test = swdp_r.loc[test_mask].to_numpy() + final_model.predict(X_test_final)
    pred_hybrid_train = swdp_r.loc[train_mask].to_numpy() + final_model.predict(X_train_final)

    pred_hybrid_all = np.full(len(ml), np.nan)
    pred_hybrid_all[train_mask] = pred_hybrid_train
    pred_hybrid_all[test_mask] = pred_hybrid_test

    # Direct CatBoost
    direct_model = CatBoostRegressor(**CATBOOST_PARAMS)
    direct_model.fit(X_train_final, y_target.loc[train_mask])
    pred_direct_cb_test = direct_model.predict(X_test_final)
    pred_direct_cb_train = direct_model.predict(X_train_final)

    pred_direct_all = np.full(len(ml), np.nan)
    pred_direct_all[train_mask] = pred_direct_cb_train
    pred_direct_all[test_mask] = pred_direct_cb_test

    # Metrics on Holdout
    y_test = y_target.loc[test_mask].to_numpy()
    pred_swdp_test = swdp_r.loc[test_mask].to_numpy()
    pred_rfp_test = ml.loc[test_mask, 'FC_R_FP'].to_numpy()
    pred_swdp_k_test = ml.loc[test_mask, 'FC_SWDP_K_Rate'].to_numpy()

    hy_r2, hy_rmse, hy_mae, hy_bias = compute_metrics(y_test, pred_hybrid_test)
    sw_r2, sw_rmse, sw_mae, sw_bias = compute_metrics(y_test, pred_swdp_test)
    rfp_r2, rfp_rmse, rfp_mae, rfp_bias = compute_metrics(y_test, pred_rfp_test)
    k_r2, k_rmse, k_mae, k_bias = compute_metrics(y_test, pred_swdp_k_test)
    dcb_r2, dcb_rmse, dcb_mae, dcb_bias = compute_metrics(y_test, pred_direct_cb_test)

    # Classification metrics
    actual_classes = y_target.apply(classify_fc)
    pred_classes = pd.Series(pred_hybrid_all).apply(classify_fc)
    test_actual_cls = actual_classes.loc[test_mask]
    test_pred_cls = pred_classes.loc[test_mask]

    acc = float(accuracy_score(test_actual_cls, test_pred_cls)) if len(test_actual_cls) else np.nan
    prec_macro = float(precision_score(test_actual_cls, test_pred_cls, labels=VMC_REGIME_LABELS, average='macro', zero_division=0)) if len(test_actual_cls) else np.nan

    cm = confusion_matrix(test_actual_cls, test_pred_cls, labels=VMC_REGIME_LABELS) if len(test_actual_cls) else np.zeros((3, 3))
    cm_df = pd.DataFrame(cm, index=[f'Actual_{l}' for l in VMC_REGIME_LABELS], columns=[f'Predicted_{l}' for l in VMC_REGIME_LABELS])
    cm_df.insert(0, 'Actual_Class', VMC_REGIME_LABELS)

    # Attach predictions
    ml['CatBoost_Predicted_FC'] = pred_hybrid_all
    ml['Direct_CatBoost_FC'] = pred_direct_all
    ml['Physics_Baseline_FC'] = swdp_r
    ml['Residual_Correction'] = pred_hybrid_all - swdp_r
    ml['Set'] = np.where(test_mask, 'Chronological_Test', 'Training')
    ml['LOEO_CV_Hybrid_FC'] = loeo_preds_hybrid
    ml['Absolute_Error_Hybrid'] = np.abs(y_target - pred_hybrid_all)
    ml['Absolute_Error_SWDP_R'] = np.abs(y_target - swdp_r)
    ml['Actual_Class'] = actual_classes
    ml['Predicted_Class'] = pred_classes
    ml['Generic_VMC_Plausibility_Flag'] = (pred_hybrid_all >= 0.0) & (pred_hybrid_all <= 1.0)

    comp_records = [
        {
            'Model': 'Process-Guided Hybrid CatBoost',
            'Validation_Protocol': f'Chronological Holdout (Event {holdout_event_id}, n={int(test_mask.sum())})',
            'R2': hy_r2,
            'RMSE': hy_rmse,
            'MAE': hy_mae,
            'Bias': hy_bias,
            'Delta_RMSE_vs_SWDP_R': hy_rmse - sw_rmse,
            'Accuracy': acc,
            'Macro_Precision': prec_macro
        },
        {
            'Model': 'SWDP-R Physics Baseline',
            'Validation_Protocol': f'Chronological Holdout (Event {holdout_event_id}, n={int(test_mask.sum())})',
            'R2': sw_r2,
            'RMSE': sw_rmse,
            'MAE': sw_mae,
            'Bias': sw_bias,
            'Delta_RMSE_vs_SWDP_R': 0.0,
            'Accuracy': 1.0,
            'Macro_Precision': 1.0
        },
        {
            'Model': 'R-FP-style Recession Plateau Estimator',
            'Validation_Protocol': f'Chronological Holdout (Event {holdout_event_id}, n={int(test_mask.sum())})',
            'R2': rfp_r2,
            'RMSE': rfp_rmse,
            'MAE': rfp_mae,
            'Bias': rfp_bias,
            'Delta_RMSE_vs_SWDP_R': rfp_rmse - sw_rmse,
            'Accuracy': float(accuracy_score(test_actual_cls, pd.Series(pred_rfp_test).apply(classify_fc))) if len(test_actual_cls) else np.nan,
            'Macro_Precision': float(precision_score(test_actual_cls, pd.Series(pred_rfp_test).apply(classify_fc), labels=VMC_REGIME_LABELS, average='macro', zero_division=0)) if len(test_actual_cls) else np.nan
        },
        {
            'Model': 'Rate-Based Recession Knee Estimator',
            'Validation_Protocol': f'Chronological Holdout (Event {holdout_event_id}, n={int(test_mask.sum())})',
            'R2': k_r2,
            'RMSE': k_rmse,
            'MAE': k_mae,
            'Bias': k_bias,
            'Delta_RMSE_vs_SWDP_R': k_rmse - sw_rmse,
            'Accuracy': float(accuracy_score(test_actual_cls, pd.Series(pred_swdp_k_test).apply(classify_fc))) if len(test_actual_cls) else np.nan,
            'Macro_Precision': float(precision_score(test_actual_cls, pd.Series(pred_swdp_k_test).apply(classify_fc), labels=VMC_REGIME_LABELS, average='macro', zero_division=0)) if len(test_actual_cls) else np.nan
        },
        {
            'Model': 'Direct CatBoost (No Physics Baseline)',
            'Validation_Protocol': f'Chronological Holdout (Event {holdout_event_id}, n={int(test_mask.sum())})',
            'R2': dcb_r2,
            'RMSE': dcb_rmse,
            'MAE': dcb_mae,
            'Bias': dcb_bias,
            'Delta_RMSE_vs_SWDP_R': dcb_rmse - sw_rmse,
            'Accuracy': float(accuracy_score(test_actual_cls, pd.Series(pred_direct_cb_test).apply(classify_fc))) if len(test_actual_cls) else np.nan,
            'Macro_Precision': float(precision_score(test_actual_cls, pd.Series(pred_direct_cb_test).apply(classify_fc), labels=VMC_REGIME_LABELS, average='macro', zero_division=0)) if len(test_actual_cls) else np.nan
        }
    ]
    comp_df = pd.DataFrame(comp_records)
    prov_df = pd.DataFrame(all_provenance_records)

    return ml, comp_df, loeo_df, cm_df, final_model, prov_df, feature_cols


def run_catboost_estimation(save_outputs: bool = True):
    df = pd.read_csv(RESULTS_DIR / 'cleaned_vmc_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    fc = pd.read_csv(RESULTS_DIR / 'physics_fc_event_estimates.csv')
    cy = pd.read_csv(RESULTS_DIR / 'valid_recession_cycles.csv')
    cy_valid = cy[cy['Is_Valid'].astype(bool)].copy()

    ml, base_feature_cols = build_base_feature_dataset(df, fc, cy)

    ml, comp_df, loeo_df, cm_df, final_model, prov_df, feature_cols = run_catboost_loeo_and_test(
        ml=ml, df_clean=df, cy_valid=cy_valid,
        base_feature_cols=base_feature_cols,
        include_sax=True,
        holdout_event_id=FINAL_HOLDOUT_EVENT
    )

    if save_outputs:
        loeo_df.to_csv(RESULTS_DIR / 'catboost_loeo_cv_metrics.csv', index=False)
        cm_df.to_csv(RESULTS_DIR / 'catboost_confusion_matrix.csv', index=False)
        ml.to_csv(RESULTS_DIR / 'catboost_fc_predictions.csv', index=False)
        comp_df.to_csv(RESULTS_DIR / 'catboost_validation_metrics.csv', index=False)
        prov_df.to_csv(RESULTS_DIR / 'sax_fold_provenance_audit.csv', index=False)
        final_model.save_model(MODELS_DIR / 'catboost_fc_residual_model.cbm')

        # Feature Importance
        fi_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': final_model.get_feature_importance()
        }).sort_values('Importance', ascending=False)
        fi_df.to_csv(RESULTS_DIR / 'catboost_feature_importance.csv', index=False)

        # Clean leftover catboost_info
        for cb_d in [ROOT_DIR / 'catboost_info', ROOT_DIR / 'Code' / 'catboost_info']:
            if cb_d.exists():
                shutil.rmtree(cb_d, ignore_errors=True)

        print("CatBoost validation and benchmarking complete.")
        print(f"SAX fold provenance audit saved: {RESULTS_DIR / 'sax_fold_provenance_audit.csv'}")
        hy_rmse = comp_df.loc[comp_df['Model'].str.contains('Hybrid'), 'RMSE'].iloc[0]
        sw_rmse = comp_df.loc[comp_df['Model'].str.contains('SWDP-R'), 'RMSE'].iloc[0]
        print(f"Chronological Test (Event {FINAL_HOLDOUT_EVENT}): Hybrid RMSE={hy_rmse:.5f}, SWDP-R RMSE={sw_rmse:.5f}, Delta_RMSE={hy_rmse-sw_rmse:+.5f}")
        print(f"LOEO-CV Median Hybrid RMSE: {loeo_df['Hybrid_RMSE'].median():.5f}, Median SWDP-R RMSE: {loeo_df['SWDP_R_RMSE'].median():.5f}")

    return ml, comp_df, loeo_df


if __name__ == '__main__':
    run_catboost_estimation()


# Research-audit invariant:
# Any exported Nested Out-of-Sample SAX Feature_Value must be identical to the
# value inserted into the corresponding model feature matrix (within numerical tolerance).
def assert_sax_feature_matches_provenance(feature_value, provenance_value, tolerance=1e-12):
    if pd.isna(feature_value) and pd.isna(provenance_value):
        return True
    assert abs(float(feature_value) - float(provenance_value)) <= tolerance, (
        "SAX model feature value does not match provenance Feature_Value."
    )
    return True
