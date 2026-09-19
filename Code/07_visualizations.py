"""07_visualizations.py - Scientific Data Visualizations for Field Capacity Analysis

Generates publication-ready figures from actual result files:
- Complete VMC time series across all soil depths.
- Measured water inputs (rainfall/irrigation) vs dynamic soil water content response.
- Cycle screening validation outcomes (Criteria A-E).
- Field capacity depth profile comparison (Sensor Reference, SWDP-R, R-FP, SWDP-K, Hybrid).
- Overnight recession inverse-time regression curves (SWDP-R peak-relative tau = t - t_peak).
- Actual vs Predicted Field Capacity scatter plot (differentiated by Train/Test holdout).
- Test set confusion matrix across soil moisture regimes.
- SAX symbolic transformation (PAA frames and Gaussian breakpoint distribution).
- Model ablation and baseline comparison chart.

Dynamic verification: Automatically tallies and verifies all generated figures without hardcoded counts.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from config import RESULTS_DIR, PLOTS_DIR, SAX_FEATURE_NAME, FINAL_HOLDOUT_EVENT

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_visualizations():
    df = pd.read_csv(RESULTS_DIR / 'cleaned_vmc_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    fc = pd.read_csv(RESULTS_DIR / 'physics_fc_event_estimates.csv')
    cb = pd.read_csv(RESULTS_DIR / 'catboost_fc_predictions.csv')
    cy = pd.read_csv(RESULTS_DIR / 'valid_recession_cycles.csv')
    metrics_path = RESULTS_DIR / 'catboost_validation_metrics.csv'
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    abl_path = RESULTS_DIR / 'ablation_study_results.csv'
    abl_df = pd.read_csv(abl_path) if abl_path.exists() else pd.DataFrame()

    sm_cols = [c for c in df.columns if c.startswith('SM_') and not any(
        c.endswith(x) for x in ['_smoothed', '_d1', '_d2', '_Was_Interpolated', '_Segment_ID']
    )]

    generated_plots = []

    # 1. Complete VMC Timeseries
    p1 = PLOTS_DIR / '01_complete_vmc_timeseries.png'
    plt.figure(figsize=(12, 5))
    for c in sm_cols:
        col_sm = c + '_smoothed'
        if col_sm in df.columns:
            depth_label = c.replace('SM_VWCFinal_', '').replace('_m3_m3', ' cm').replace('1_140', '140')
            plt.plot(df['Timestamp'], df[col_sm], label=depth_label, linewidth=1.2)
    plt.xlabel('Timestamp', fontsize=11, fontweight='bold')
    plt.ylabel('Volumetric Water Content (m³/m³)', fontsize=11, fontweight='bold')
    plt.title('Complete Soil Water Content Time Series by Depth (Continuous Segments Filtered)', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    plt.close()
    generated_plots.append(p1.name)

    # 2. Water Input vs VMC
    p2 = PLOTS_DIR / '02_water_input_vs_vmc.png'
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax2.bar(df['Timestamp'], df['Total_Water_mm'], color='blue', alpha=0.3, width=0.04, label='Total Water Input (mm)')
    for c in sm_cols[:3]:
        col_sm = c + '_smoothed'
        if col_sm in df.columns:
            depth_label = c.replace('SM_VWCFinal_', '').replace('_m3_m3', ' cm')
            ax1.plot(df['Timestamp'], df[col_sm], label=depth_label, linewidth=1.5)
    ax1.set_xlabel('Timestamp', fontsize=11, fontweight='bold')
    ax1.set_ylabel('VMC (m³/m³)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Water Input (mm)', color='blue', fontsize=11, fontweight='bold')
    ax1.set_title('Measured Hydrological Water Inputs vs Soil Moisture Response', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    plt.close()
    generated_plots.append(p2.name)

    # 3. Cycle Validation Outcomes
    p3 = PLOTS_DIR / '03_cycle_validation_outcomes.png'
    counts = cy['Rejection_Reason'].value_counts()
    plt.figure(figsize=(10, 5.5))
    counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.85)
    plt.ylabel('Number of Depth Cycles', fontsize=11, fontweight='bold')
    plt.title('Hydrological Recession Cycle Validation Outcomes (Bean et al. Criteria A-E)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(p3, dpi=200)
    plt.close()
    generated_plots.append(p3.name)

    # 4. Depth Profile FC Comparison
    p4 = PLOTS_DIR / '04_depth_profile_fc_comparison.png'
    s = fc.groupby('Depth_cm')[['FC_Direct_SensorReference', 'FC_SWDP_R', 'FC_R_FP', 'FC_SWDP_K_Rate']].mean().reset_index()
    cb_depth = cb.groupby('Depth_cm')['CatBoost_Predicted_FC'].mean().reset_index()
    s = s.merge(cb_depth, on='Depth_cm', how='left')

    plt.figure(figsize=(9, 5.5))
    plt.plot(s['Depth_cm'], s['FC_Direct_SensorReference'], marker='o', linewidth=2.0, label='Direct Sensor Reference')
    plt.plot(s['Depth_cm'], s['FC_SWDP_R'], marker='s', linewidth=2.0, linestyle='--', label='SWDP-R (Overnight Regression)')
    plt.plot(s['Depth_cm'], s['FC_R_FP'], marker='^', linewidth=1.5, linestyle=':', label='R-FP-style Recession Plateau Estimator')
    plt.plot(s['Depth_cm'], s['FC_SWDP_K_Rate'], marker='v', linewidth=1.5, linestyle='-.', label='Rate-Based Recession Knee Estimator')
    plt.plot(s['Depth_cm'], s['CatBoost_Predicted_FC'], marker='d', linewidth=2.0, color='crimson', label='Process-Guided Hybrid CatBoost')
    plt.xlabel('Soil Depth (cm)', fontsize=11, fontweight='bold')
    plt.ylabel('Field Capacity (m³/m³)', fontsize=11, fontweight='bold')
    plt.title('Comparison of Field Capacity Estimators Across Soil Profile Depths', fontsize=12, fontweight='bold')
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(p4, dpi=200)
    plt.close()
    generated_plots.append(p4.name)

    # 5. Overnight Peak-Relative Recession Curves
    p5 = PLOTS_DIR / '05_overnight_swdp_r_drainage_curves.png'
    valid_cy = cy[cy['Is_Valid'].astype(bool)].copy()
    if len(valid_cy) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for idx_plot, cy_idx in enumerate([0, min(3, len(valid_cy) - 1)]):
            rep = valid_cy.iloc[cy_idx]
            col_sm = rep['Column'] + '_smoothed'
            overnight_start = pd.to_datetime(rep['Overnight_Start_Time'])
            sched_time = pd.to_datetime(rep['Scheduled_Irrigation_Time'])
            peak_time = pd.to_datetime(rep['Peak_Time'])

            sub_df = df[(df['Timestamp'] >= overnight_start) & (df['Timestamp'] <= sched_time)].copy()
            peak_row = df[df['Timestamp'] == peak_time]
            peak_el = float(peak_row['Elapsed_hours'].iloc[0]) if len(peak_row) else float(sub_df['Elapsed_hours'].iloc[0])

            t_tau = (sub_df['Elapsed_hours'] - peak_el).to_numpy(float)
            y_sub = sub_df[col_sm].to_numpy(float)
            vmask = t_tau > 0
            t_tau = t_tau[vmask]
            y_sub = y_sub[vmask]

            if len(t_tau) >= 3:
                poly = np.polyfit(1.0 / t_tau, y_sub, 1)
                t_fine = np.linspace(min(t_tau), max(t_tau[-1], 24.0), 100)
                y_fit = poly[0] * (1.0 / t_fine) + poly[1]

                ax = axes[idx_plot]
                ax.scatter(t_tau, y_sub, color='navy', s=50, label='Overnight Obs (τ > 0)', zorder=5)
                ax.plot(t_fine, y_fit, 'r-', linewidth=2.0, label=r'Peak-Relative Fit: $\theta = a(1/\tau) + FC$')
                ax.axhline(poly[1], color='green', linestyle='--', linewidth=1.8, label=f'Asymptotic FC = {poly[1]:.4f}')
                ax.set_xlabel('Elapsed Hours from Wetting Peak (τ = t - t_peak)', fontsize=10, fontweight='bold')
                ax.set_ylabel('VMC (m³/m³)', fontsize=10, fontweight='bold')
                ax.set_title(f'SWDP-R Peak-Relative Fit: Event {int(rep["Event_ID"])}, Depth {int(rep["Depth_cm"])} cm', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(p5, dpi=200)
        plt.close()
        generated_plots.append(p5.name)

    # 6. Scatter Actual vs Predicted FC
    p6 = PLOTS_DIR / '06_scatter_actual_vs_predicted.png'
    plt.figure(figsize=(8, 6.5))
    train_mask = cb['Set'] == 'Training'
    test_mask = cb['Set'] == 'Chronological_Test'
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    test_eid = cb.loc[test_mask, 'Event_ID'].iloc[0] if n_test else 'N/A'

    plt.scatter(cb.loc[train_mask, 'FC_Direct_SensorReference'], cb.loc[train_mask, 'CatBoost_Predicted_FC'],
                color='navy', label=f'Training Events (n={n_train})', s=70, alpha=0.8, edgecolors='k', marker='o')
    plt.scatter(cb.loc[test_mask, 'FC_Direct_SensorReference'], cb.loc[test_mask, 'CatBoost_Predicted_FC'],
                color='crimson', label=f'Future Test Event {test_eid} (n={n_test})', s=100, alpha=0.9, edgecolors='k', marker='^')

    lo = min(cb['FC_Direct_SensorReference'].min(), cb['CatBoost_Predicted_FC'].min()) - 0.005
    hi = max(cb['FC_Direct_SensorReference'].max(), cb['CatBoost_Predicted_FC'].max()) + 0.005
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='1:1 Reference Line')

    if not metrics_df.empty:
        m_row = metrics_df[metrics_df['Model'].str.contains('Hybrid')].iloc[0]
        text_lines = [
            f"Holdout Test (Event {test_eid}, n={n_test}):",
            f"Test R² = {m_row['R2']:.4f}",
            f"Test RMSE = {m_row['RMSE']:.5f} m³/m³",
            f"Test MAE = {m_row['MAE']:.5f} m³/m³",
            f"ΔRMSE vs SWDP-R = {m_row['Delta_RMSE_vs_SWDP_R']:+.5f}"
        ]
        text_box = "\n".join(text_lines)
        plt.gca().text(0.05, 0.95, text_box, transform=plt.gca().transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.xlabel('Direct Pre-Irrigation Sensor Reference (m³/m³)', fontsize=11, fontweight='bold')
    plt.ylabel('Process-Guided Hybrid Predicted FC (m³/m³)', fontsize=11, fontweight='bold')
    plt.title('Actual vs Predicted Field Capacity (Chronological Holdout)', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(p6, dpi=200)
    plt.close()
    generated_plots.append(p6.name)

    # 7. Confusion Matrix
    p7 = PLOTS_DIR / '07_confusion_matrix.png'
    cm_path = RESULTS_DIR / 'catboost_confusion_matrix.csv'
    if cm_path.exists():
        cm_df = pd.read_csv(cm_path)
        labels = cm_df['Actual_Class'].tolist()
        pred_cols = [f'Predicted_{l}' for l in labels]
        cm_mat = cm_df[pred_cols].to_numpy()

        plt.figure(figsize=(7.5, 6))
        plt.imshow(cm_mat, cmap='Blues', interpolation='nearest')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(labels)), labels, fontsize=10)
        plt.yticks(np.arange(len(labels)), labels, fontsize=10)
        plt.xlabel('Predicted Field Capacity Class', fontsize=11, fontweight='bold')
        plt.ylabel('Actual Field Capacity Class', fontsize=11, fontweight='bold')
        plt.title('Test Set Confusion Matrix (Field Capacity Regimes)', fontsize=12, fontweight='bold', pad=12)

        thresh = cm_mat.max() / 2.0 if cm_mat.max() > 0 else 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = int(cm_mat[i, j])
                plt.text(j, i, str(val), ha='center', va='center',
                         color='white' if val > thresh else 'black', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(p7, dpi=200)
        plt.close()
        generated_plots.append(p7.name)

    # 8. SAX Symbolic Transformation
    p8 = PLOTS_DIR / '08_r_sax_gaussian_symbolic_transformation.png'
    sax_path = RESULTS_DIR / 'r_sax_cycle_similarity.csv'
    if len(valid_cy) > 0 and sax_path.exists():
        rep = valid_cy.iloc[0]
        col_sm = rep['Column'] + '_smoothed'
        sched_time = pd.to_datetime(rep['Scheduled_Irrigation_Time'])
        peak_time = pd.to_datetime(rep['Peak_Time'])
        seg_df = df[(df['Timestamp'] >= peak_time) & (df['Timestamp'] <= sched_time)]
        y_raw = seg_df[col_sm].to_numpy(float)
        t_hours = (seg_df['Elapsed_hours'] - seg_df['Elapsed_hours'].iloc[0]).to_numpy(float)

        word_size = 8
        alphabet_size = 5
        edges = np.linspace(0, len(y_raw), word_size + 1).astype(int)
        frame_means = np.array([np.mean(y_raw[edges[i]:edges[i + 1]]) for i in range(word_size)])
        frame_t = np.array([np.mean(t_hours[edges[i]:edges[i + 1]]) for i in range(word_size)])

        std_val = np.std(frame_means) if np.std(frame_means) > 1e-12 else 1.0
        z_means = (frame_means - np.mean(frame_means)) / std_val
        bp = norm.ppf(np.arange(1, alphabet_size) / alphabet_size)
        letters = [chr(97 + int(np.sum(v > bp))) for v in z_means]
        sax_str = ''.join(letters)

        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0])
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_hours, y_raw, 'b-', linewidth=2.0, label='Smoothed Drainage Curve θ(t)')
        colors = plt.cm.tab10(np.linspace(0, 1, word_size))
        for i in range(word_size):
            t_s = t_hours[edges[i]]
            t_e = t_hours[min(edges[i + 1], len(t_hours) - 1)]
            ax1.axvspan(t_s, t_e, color=colors[i], alpha=0.1)
            ax1.hlines(frame_means[i], t_s, t_e, colors=colors[i], linestyles='--', linewidth=2.0)
            ax1.plot(frame_t[i], frame_means[i], 'o', color=colors[i], markersize=7)
            ax1.text(frame_t[i], frame_means[i] + 0.0005, f"F{i+1}: '{letters[i]}'", ha='center', fontsize=9, fontweight='bold', color=colors[i])

        ax1.set_xlabel('Elapsed Drainage Time (hours)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('VMC (m³/m³)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Continuous Recession Discretized into 8 PAA Frames (SAX Word: "{sax_str}")', fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='lower left')

        ax2 = fig.add_subplot(gs[1])
        z_grid = np.linspace(-3.0, 3.0, 300)
        ax2.plot(norm.pdf(z_grid), z_grid, 'k-', linewidth=1.8, label='Standard Normal N(0,1)')
        cuts = np.concatenate([[-3.0], bp, [3.0]])
        band_colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']
        let_labels = ['a', 'b', 'c', 'd', 'e']
        for b_i in range(len(let_labels)):
            z_s = np.linspace(cuts[b_i], cuts[b_i + 1], 100)
            ax2.fill_betweenx(z_s, 0, norm.pdf(z_s), color=band_colors[b_i], alpha=0.25)
            ax2.text(0.12, (cuts[b_i] + cuts[b_i + 1]) / 2.0, f"Region '{let_labels[b_i]}'", fontsize=10, fontweight='bold', va='center')

        for b in bp:
            ax2.axhline(b, color='red', linestyle=':', linewidth=1.5)
        for i in range(word_size):
            ax2.plot(norm.pdf(z_means[i]), z_means[i], 'o', color=colors[i], markersize=8, markeredgecolor='k')

        ax2.set_xlabel('Gaussian Probability Density', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Standardized Value (z-score)', fontsize=11, fontweight='bold')
        ax2.set_title('Equiprobable Breakpoint Slicing (w=8, |Σ|=5)', fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(p8, dpi=200)
        plt.close()
        generated_plots.append(p8.name)

    # 9. Ablation Study Model Comparison
    p9 = PLOTS_DIR / '09_ablation_and_model_comparison.png'
    if not abl_df.empty:
        plt.figure(figsize=(9, 5))
        plt.bar(abl_df['Ablation_Code'], abl_df['Test_RMSE'] * 1000.0, color='darkseagreen', edgecolor='black', alpha=0.85)
        plt.ylabel('Test RMSE (×10⁻³ m³/m³)', fontsize=11, fontweight='bold')
        plt.xlabel('Ablation Experiment', fontsize=11, fontweight='bold')
        plt.title('Model Ablation Analysis: Pure Physics vs Pure ML vs Residual Hybrids', fontsize=12, fontweight='bold')
        plt.xticks(rotation=25, ha='right', fontsize=9)
        for idx_b, r in abl_df.iterrows():
            plt.text(idx_b, r['Test_RMSE'] * 1000.0 + 0.3, f"{r['Test_RMSE']:.4f}", ha='center', fontsize=9, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.5, axis='y')
        plt.tight_layout()
        plt.savefig(p9, dpi=200)
        plt.close()
        generated_plots.append(p9.name)

    print(f"Dynamic Visualization Verification: Successfully generated {len(generated_plots)} figures in Plots/:")
    for p in generated_plots:
        print(f"  - {p}")
    return generated_plots


if __name__ == '__main__':
    run_visualizations()
