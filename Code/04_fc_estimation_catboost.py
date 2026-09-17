from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'Results'
M = ROOT / 'Models'
M.mkdir(exist_ok=True)
TEST_EVENTS = 2


def build_features(df, fc, cy, sax):
    cy = cy[cy.Is_Valid.astype(bool)].copy()
    ml = fc.merge(
        cy[['Event_ID','Column','Pre_Event_VMC','Total_Water_mm','VMC_Rise',
            'Recession_Hours','Peak_Time','Peak_Idx']],
        on=['Event_ID','Column'], how='inner'
    )
    ml = ml.merge(
        sax[['Event_ID','Column','LeaveOneEventOut_SAX_Similarity']],
        on=['Event_ID','Column'], how='left'
    )

    rows = []
    for _, r in ml.iterrows():
        a = int(r.Event_ID)
        c = r.Column
        match = cy[(cy.Event_ID == a) & (cy.Column == c)].iloc[0]
        p = int(match.Peak_Idx)
        e = int(match.Recession_End_Idx)
        rec = df.iloc[p:e+1]
        t = (rec.Elapsed_hours - rec.Elapsed_hours.iloc[0]).to_numpy(float)
        y = rec[c + '_smoothed'].to_numpy(float)
        d = np.gradient(y, t)

        def val_at(hours):
            if len(t) == 0:
                return np.nan
            i = np.argmin(np.abs(t - hours))
            return float(y[i])

        elapsed3 = min(3.0, float(t[-1])) if len(t) else 0.0
        rows.append({
            'Initial_Drainage_Rate': float((y[0] - val_at(elapsed3)) / max(elapsed3, 1.0)),
            'Peak_Hour': int(r.Peak_Time.hour),
            'Month': int(r.Peak_Time.month),
        })

    ml = pd.concat([ml.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    ml['SAX_Similarity_Missing'] = ml['LeaveOneEventOut_SAX_Similarity'].isna().astype(int)
    ml['LeaveOneEventOut_SAX_Similarity'] = ml['LeaveOneEventOut_SAX_Similarity'].fillna(0.0)

    feature_cols = [
        'Depth_cm', 'Pre_Event_VMC', 'Peak_VMC', 'Total_Water_mm', 'VMC_Rise',
        'Initial_Drainage_Rate', 'Peak_Hour', 'Month',
        'LeaveOneEventOut_SAX_Similarity'
    ]
    return ml.dropna(subset=feature_cols + ['FC_ExpertStyle_Direct']).reset_index(drop=True), feature_cols


def run_catboost_estimation():
    df = pd.read_csv(R / 'cleaned_vmc_data.csv')
    fc = pd.read_csv(R / 'physics_fc_event_estimates.csv')
    cy = pd.read_csv(R / 'valid_recession_cycles.csv', parse_dates=['Peak_Time'])
    sax = pd.read_csv(R / 'r_sax_cycle_similarity.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')

    ml, feature_cols = build_features(df, fc, cy, sax)
    X = ml[feature_cols]
    y = ml.FC_ExpertStyle_Direct

    event_order = ml.groupby('Event_ID')['Peak_Time'].min().sort_values().index.tolist()
    n_test = min(TEST_EVENTS, max(1, len(event_order) // 3))
    test_events = set(event_order[-n_test:])
    train_mask = ~ml.Event_ID.isin(test_events)
    test_mask = ml.Event_ID.isin(test_events)

    # Physics-first hybrid model:
    # predict only the residual between the direct FC reference and SWDP-R.
    # This is a residual-correction formulation, not a replacement of the
    # physically derived estimator. It is used because SWDP-R is already a
    # strong drainage-based estimator and the ML model should learn only
    # systematic residual structure not explained by that estimator.
    residual_train = y.loc[train_mask] - ml.loc[train_mask, 'FC_SWDP_R']
    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.03,
        depth=2,
        l2_leaf_reg=5,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    model.fit(X.loc[train_mask], residual_train)
    pred = ml.FC_SWDP_R.to_numpy() + model.predict(X)

    y_test = y.loc[test_mask].to_numpy()
    pred_test = pred[test_mask]
    y_train = y.loc[train_mask].to_numpy()
    pred_train = pred[train_mask]

    def metrics(yv, pv):
        r2 = r2_score(yv, pv) if len(np.unique(yv)) > 1 else np.nan
        rmse = mean_squared_error(yv, pv) ** 0.5
        mae = mean_absolute_error(yv, pv)
        bias = float(np.mean(pv - yv))
        return r2, rmse, mae, bias

    test_r2, test_rmse, test_mae, test_bias = metrics(y_test, pred_test)
    train_r2, train_rmse, train_mae, train_bias = metrics(y_train, pred_train)
    base_test_r2, base_test_rmse, base_test_mae, base_test_bias = metrics(y_test, ml.loc[test_mask, 'FC_SWDP_R'].to_numpy())

    ml['CatBoost_Predicted_FC'] = pred
    ml['Physics_Baseline_FC'] = ml['FC_SWDP_R']
    ml['Residual_Correction'] = pred - ml['FC_SWDP_R']
    ml['Set'] = np.where(test_mask, 'Chronological_Test', 'Training')
    ml['Absolute_Error'] = np.abs(y - pred)
    ml['Prediction_Target'] = 'FC_ExpertStyle_Direct'
    ml.to_csv(R / 'catboost_fc_predictions.csv', index=False)
    model.save_model(M / 'catboost_fc_residual_model.cbm')

    import shutil
    cb_info = ROOT / 'catboost_info'
    if cb_info.exists():
        shutil.rmtree(cb_info)

    pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.get_feature_importance()
    }).sort_values('Importance', ascending=False).to_csv(R / 'catboost_feature_importance.csv', index=False)

    ml.groupby('Depth_cm').CatBoost_Predicted_FC.agg(['mean','median','std','min','max']).reset_index().to_csv(
        R / 'catboost_fc_summary_by_depth.csv', index=False
    )

    metrics_row = {
        'Validation': 'Chronological hydrological-event holdout; no random K-fold CV',
        'Model': 'Physics-first residual-correction CatBoost hybrid',
        'Target': 'Expert-style direct FC point immediately before scheduled irrigation',
        'Base_Physics_Estimator': 'SWDP-R',
        'Physics_Features': 'SWDP-R baseline plus R-FP, recession dynamics, wetting response, depth and event descriptors',
        'Train_Events': len(event_order) - n_test,
        'Test_Events': n_test,
        'Train_Samples': int(train_mask.sum()),
        'Test_Samples': int(test_mask.sum()),
        'Test_Event_IDs': ','.join(map(str, event_order[-n_test:])),
        'Train_R2': train_r2,
        'Train_RMSE': train_rmse,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae,
        'Test_Bias': test_bias,
        'SWDP_R_Baseline_Test_R2': base_test_r2,
        'SWDP_R_Baseline_Test_RMSE': base_test_rmse,
        'SWDP_R_Baseline_Test_MAE': base_test_mae,
        'SWDP_R_Baseline_Test_Bias': base_test_bias,
        'Scientific_Interpretation': 'The hybrid learns residual structure around a drainage-based SWDP-R estimate; it does not replace the physics estimator with a black-box prediction.'
    }
    pd.DataFrame([metrics_row]).to_csv(R / 'catboost_validation_metrics.csv', index=False)

    print(
        f'Chronological holdout: train events={len(event_order)-n_test}, test events={n_test}; '
        f'Hybrid Test R2={test_r2:.4f}, RMSE={test_rmse:.5f}, MAE={test_mae:.5f}, Bias={test_bias:.5f}; '
        f'SWDP-R baseline R2={base_test_r2:.4f}, RMSE={base_test_rmse:.5f}'
    )
    return ml, test_r2, test_rmse, test_mae


if __name__ == '__main__':
    run_catboost_estimation()
