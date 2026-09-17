from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'Results'; OUT=ROOT/'Reports'; OUT.mkdir(exist_ok=True)
def run():
    files=['data_quality_report.csv','detected_events_summary.csv','valid_recession_cycles.csv','physics_fc_event_estimates.csv','physics_fc_summary_by_depth.csv','catboost_fc_predictions.csv','catboost_fc_summary_by_depth.csv','catboost_feature_importance.csv','catboost_validation_metrics.csv','r_sax_cycle_similarity.csv']
    data={f[:-4]:pd.read_csv(R/f) for f in files}
    x=OUT/'Final_FC_Estimation_Master_Results.xlsx'
    with pd.ExcelWriter(x,engine='openpyxl') as w:
        for name,d in data.items(): d.to_excel(w,sheet_name=name[:31],index=False)
    metrics=data['catboost_validation_metrics'].iloc[0]
    report=OUT/'fc_estimation_research_report.md'
    text=f"""# Field Capacity Estimation – Phase 2.2 Final Hybrid Model

## Scientific basis
This implementation follows Bean et al. (2018) for soil-water time-series cycle validation and recession-based FC estimation, supported by Fazackerley & Lawrence (2012) and USDA-NRCS drainage-based FC guidance.

## Final model
The final model is a physics-informed CatBoost hybrid. The target is the expert-style direct FC point immediately before scheduled irrigation. Predictors include depth, rainfall/irrigation amount, VMC wetting response, recession duration and slopes, overnight depletion, early recession VMC, and R-FP/SWDP-R physics estimates. The direct FC target is excluded from predictors.

## Time-series validation
Random K-fold CV is not used. The final complete hydrological events are held out chronologically as future events.

- Training events: {int(metrics.Train_Events)}
- Test events: {int(metrics.Test_Events)}
- Training samples: {int(metrics.Train_Samples)}
- Test samples: {int(metrics.Test_Samples)}
- Test event IDs: {metrics.Test_Event_IDs}
- **Chronological test R²: {metrics.Test_R2:.4f}**
- Test RMSE: {metrics.Test_RMSE:.4f} m³/m³
- Test MAE: {metrics.Test_MAE:.4f} m³/m³
- Test Bias: {metrics.Test_Bias:.4f} m³/m³
- Training R² (fit diagnostic only): {metrics.Train_R2:.4f}
- SWDP-R baseline test R²: {metrics.SWDP_R_Baseline_Test_R2:.4f}
- SWDP-R baseline test RMSE: {metrics.SWDP_R_Baseline_Test_RMSE:.5f} m³/m³

## Interpretation
R² = {metrics.Test_R2:.4f} is the chronological future-event holdout result for the residual-correction hybrid. The SWDP-R baseline is also reported because the hybrid must demonstrate value beyond the physics estimator. It is not laboratory accuracy because only two independent future events are available and the FC target is an expert-style sensor-derived reference. Independent expert or laboratory FC measurements are still required for external validation.

## References
- Bean, G., Huffaker, R.G., & Migliaccio, K.W. (2018). Vadose Zone Journal. DOI 10.2136/vzj2018.04.0073.
- Fazackerley, S. & Lawrence, R. (2012). Irrigation and Drainage 61:416–424. DOI 10.1002/ird.646.
- USDA-NRCS soil-water/available-water guidance.
- Wang et al. (2026). PSO-CatBoost sensor-based FC estimation. DOI 10.1016/j.agrcom.2026.100156.
- Bhasme, Vagadiya & Bhatia (2022). Physics-informed machine learning for hydrological processes. DOI 10.1016/j.jhydrol.2022.128145.
- Jiang et al. (2025). Hybrid process-based model + ML error correction. DOI 10.1016/j.advwatres.2025.105177.
- (2025). Hybrid time-series and physics-informed ML for soil water content. DOI 10.1016/j.engappai.2025.110105.
- Zhou et al. (2025). Physical, ML and hybrid methods for soil water movement. DOI 10.1016/j.jhydrol.2024.132639.
"""
    # Create concise, simple Excel report containing only FC values, models, and R2 performance metrics
    cb_df = data['catboost_fc_predictions']
    metrics_df = data['catboost_validation_metrics']

    # Sheet 1: Model Performance R2
    perf_summary = pd.DataFrame([
        {
            'Model / Estimator': 'CatBoost Hybrid Model (Leak-Free Residual)',
            'Test R²': metrics.Test_R2,
            'Test RMSE (m³/m³)': metrics.Test_RMSE,
            'Test MAE (m³/m³)': metrics.Test_MAE,
            'Test Bias (m³/m³)': metrics.Test_Bias
        },
        {
            'Model / Estimator': 'SWDP-R Physical Baseline Estimator',
            'Test R²': metrics.SWDP_R_Baseline_Test_R2,
            'Test RMSE (m³/m³)': metrics.SWDP_R_Baseline_Test_RMSE,
            'Test MAE (m³/m³)': metrics.SWDP_R_Baseline_Test_MAE,
            'Test Bias (m³/m³)': metrics.SWDP_R_Baseline_Test_Bias
        }
    ])

    # Sheet 2: FC Values Aggregated by Depth
    fc_by_depth = cb_df.groupby('Depth_cm').agg(
        Actual_FC_Direct=('FC_ExpertStyle_Direct', 'mean'),
        SWDP_R_Physics_FC=('FC_SWDP_R', 'mean'),
        CatBoost_Hybrid_FC=('CatBoost_Predicted_FC', 'mean')
    ).reset_index().rename(columns={
        'Depth_cm': 'Depth (cm)',
        'Actual_FC_Direct': 'Actual FC Direct (m³/m³)',
        'SWDP_R_Physics_FC': 'SWDP-R Physics FC (m³/m³)',
        'CatBoost_Hybrid_FC': 'CatBoost Hybrid FC (m³/m³)'
    })

    # Sheet 3: FC Event Predictions
    fc_events = cb_df[['Depth_cm', 'Event_ID', 'Set', 'FC_ExpertStyle_Direct', 'FC_SWDP_R', 'CatBoost_Predicted_FC']].rename(columns={
        'Depth_cm': 'Depth (cm)',
        'Event_ID': 'Event ID',
        'Set': 'Dataset Split',
        'FC_ExpertStyle_Direct': 'Actual FC Direct (m³/m³)',
        'FC_SWDP_R': 'SWDP-R Physics FC (m³/m³)',
        'CatBoost_Predicted_FC': 'CatBoost Hybrid FC (m³/m³)'
    })

    simple_excel_results = R / 'FC_Summary_Simple.xlsx'
    simple_excel_reports = OUT / 'FC_Summary_Simple.xlsx'

    for dest in [simple_excel_results, simple_excel_reports]:
        with pd.ExcelWriter(dest, engine='openpyxl') as w:
            perf_summary.to_excel(w, sheet_name='Model_Performance_R2', index=False)
            fc_by_depth.to_excel(w, sheet_name='FC_Summary_by_Depth', index=False)
            fc_events.to_excel(w, sheet_name='FC_Event_Predictions', index=False)

    report.write_text(text, encoding='utf-8')
    print('Master report written:', x)
    print('Simple summary Excel written:', simple_excel_results)

if __name__ == '__main__':
    run()
