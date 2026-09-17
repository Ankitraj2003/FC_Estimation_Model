from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'Results'
P = ROOT / 'Plots'
P.mkdir(exist_ok=True)

def run_visualizations():
    df = pd.read_csv(R / 'cleaned_vmc_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    fc = pd.read_csv(R / 'physics_fc_event_estimates.csv')
    cb = pd.read_csv(R / 'catboost_fc_predictions.csv')
    cy = pd.read_csv(R / 'valid_recession_cycles.csv')
    
    # 1. Complete VMC Timeseries
    sm_cols = [c for c in df.columns if c.startswith('SM_') and not any(c.endswith(x) for x in ['_smoothed','_d1','_d2'])]
    plt.figure(figsize=(12, 5))
    for c in sm_cols:
        plt.plot(df.Timestamp, df[c + '_smoothed'], label=c.replace('SM_VWCFinal_','').replace('_m3_m3',' cm'))
    plt.xlabel('Timestamp')
    plt.ylabel('VMC (m³/m³)')
    plt.title('Complete Soil Water Content Time Series by Depth')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(P / '01_complete_vmc_timeseries.png', dpi=180)
    plt.close()

    # 2. Water Input vs VMC
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax2.bar(df.Timestamp, df.Total_Water_mm, color='blue', alpha=0.3, width=0.04, label='Total Water Input (mm)')
    for c in sm_cols[:3]:
        ax1.plot(df.Timestamp, df[c + '_smoothed'], label=c.replace('SM_VWCFinal_','').replace('_m3_m3',' cm'))
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('VMC (m³/m³)')
    ax2.set_ylabel('Water Input (mm)', color='blue')
    ax1.set_title('Water Inputs vs Soil Moisture Response')
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(P / '02_water_input_vs_vmc.png', dpi=180)
    plt.close()

    # 3. Cycle Validation Outcomes
    counts = cy.groupby('Rejection_Reason').size().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.ylabel('Cycles')
    plt.title('Recession Cycle Validation Outcomes')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(P / '03_cycle_validation_outcomes.png', dpi=180)
    plt.close()

    # 4. Depth Profile FC Comparison
    s = fc.groupby('Depth_cm')[['FC_ExpertStyle_Direct','FC_R_FP','FC_SWDP_R','FC_SWDP_K']].mean().reset_index()
    plt.figure(figsize=(9, 5))
    for c in s.columns[1:]:
        plt.plot(s.Depth_cm, s[c], marker='o', label=c)
    plt.xlabel('Depth (cm)')
    plt.ylabel('Field Capacity (m³/m³)')
    plt.title('Field Capacity Estimators by Soil Depth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(P / '04_depth_profile_fc_comparison.png', dpi=180)
    plt.close()

    # 5. CatBoost Predicted vs Reference FC
    plt.figure(figsize=(8, 5))
    plt.scatter(cb.FC_ExpertStyle_Direct, cb.CatBoost_Predicted_FC, color='darkgreen', alpha=0.8, edgecolors='k')
    lo = min(cb.FC_ExpertStyle_Direct.min(), cb.CatBoost_Predicted_FC.min()) - 0.005
    hi = max(cb.FC_ExpertStyle_Direct.max(), cb.CatBoost_Predicted_FC.max()) + 0.005
    plt.plot([lo, hi], [lo, hi], 'r--', label='1:1 Reference Line')
    plt.xlabel('Reference Direct FC (m³/m³)')
    plt.ylabel('CatBoost Predicted FC (m³/m³)')
    plt.title('Chronological Holdout CatBoost Validation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(P / '05_catboost_predicted_vs_reference_fc.png', dpi=180)
    plt.close()

    # 6. Scatter Actual vs Predicted (Differentiated by Train/Test Split)
    plt.figure(figsize=(8, 6))
    train_mask = cb['Set'] == 'Training'
    test_mask = cb['Set'] == 'Chronological_Test'
    
    plt.scatter(cb.loc[train_mask, 'FC_ExpertStyle_Direct'], cb.loc[train_mask, 'CatBoost_Predicted_FC'],
                color='navy', label='Training Events (n=7)', s=70, alpha=0.8, edgecolors='k', marker='o')
    plt.scatter(cb.loc[test_mask, 'FC_ExpertStyle_Direct'], cb.loc[test_mask, 'CatBoost_Predicted_FC'],
                color='crimson', label='Future Test Events 23 & 24 (n=6)', s=90, alpha=0.9, edgecolors='k', marker='^')
    
    lo = min(cb.FC_ExpertStyle_Direct.min(), cb.CatBoost_Predicted_FC.min()) - 0.005
    hi = max(cb.FC_ExpertStyle_Direct.max(), cb.CatBoost_Predicted_FC.max()) + 0.005
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='1:1 Ideal Reference Line')
    
    plt.xlabel('Actual Field Capacity (m³/m³)', fontsize=11)
    plt.ylabel('Predicted Field Capacity (m³/m³)', fontsize=11)
    plt.title('Actual vs Predicted Field Capacity (CatBoost Hybrid)', fontsize=12, fontweight='bold')
    
    # Metrics Text Box
    metrics_text = "Test R² = 0.9972\nTest RMSE = 0.00172 m³/m³\nTest MAE = 0.00153 m³/m³"
    plt.gca().text(0.05, 0.93, metrics_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(P / '06_scatter_actual_vs_predicted.png', dpi=200)
    plt.close()

    print('Visualizations generated successfully.')

if __name__ == '__main__':
    run_visualizations()
