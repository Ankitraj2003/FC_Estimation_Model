"""03_fc_estimation_physics.py - Process-Based Field Capacity Estimators

Implements process-based field capacity estimation methods following Bean et al. (2018)
and Fazackerley & Lawrence (2012):

1. SWDP-R (Soil Water Dynamics Processing - Regression):
   - Isolates the overnight recession window (22:00 to morning irrigation) after wetting peak,
     minimizing ET interference (ET is sufficiently reduced at night, not zero).
   - Time axis is PEAK-RELATIVE (Bean et al. 2018, Eq. 1):
       tau_i = t_i - t_peak,  tau_i > 0
     Inverse-time transform: x_i = 1 / tau_i
     Linear model: theta_i = a * x_i + FC_SWDP_R
     The y-intercept (FC_SWDP_R) is the asymptotic FC as tau -> infinity.
   - Computes comprehensive regression diagnostics.

2. R-FP-style Recession Plateau Estimator (project-specific adaptation):
   - Locates the first sustained low-change drainage plateau following the wetting peak.

3. Rate-Based Recession Knee Estimator (project-specific adaptation of SWDP-K):
   - Identifies rapid-to-slow drainage transition via chronological forward-search
     for the first persistent low-rate region.

4. Target / Reference:
   - FC_Direct_SensorReference: Operational sensor reading immediately prior to
     scheduled irrigation. Explicitly distinguished from an independent ground truth.
   - FC_Consensus_Index: Multi-estimator robust consensus (median).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

from config import RESULTS_DIR, MIN_SWPDR_POINTS


def parse_depth(c: str) -> int:
    m = re.search(r'_(\d+)_m3', c)
    return int(m.group(1)) if m else int(re.search(r'(\d+)', c).group(1))


def estimate_rfp_plateau(y: np.ndarray, tol: float = 0.0005) -> tuple:
    """R-FP-style recession plateau estimator: first sustained low-rate drainage segment."""
    if len(y) < 5:
        return np.nan, None
    d = np.gradient(y)
    peak = int(np.nanargmax(y))
    for i in range(peak + 2, len(y) - 2):
        window_deriv = np.abs(d[i:i + 3])
        if np.nanmedian(window_deriv) < tol:
            return float(np.nanmedian(y[i:i + 3])), i
    return float(y[-1]), len(y) - 1


def estimate_swdp_k_rate(y: np.ndarray, t: np.ndarray,
                         smooth_window: int = 5,
                         slow_threshold_pct: float = 0.20,
                         min_slow_consecutive: int = 3) -> float:
    """Rate-Based Recession Knee Estimator (chronological forward-search)."""
    if len(y) < 4:
        return float(y[-1]) if len(y) else np.nan

    d1 = np.gradient(y, t)

    if len(d1) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        d1_smooth = np.convolve(d1, kernel, mode='same')
    else:
        d1_smooth = d1.copy()

    neg_rates = d1_smooth[d1_smooth < 0]
    if len(neg_rates) == 0:
        return float(y[-1])

    max_drainage_rate = float(np.min(neg_rates))
    slow_threshold = slow_threshold_pct * abs(max_drainage_rate)

    consecutive = 0
    for i in range(len(d1_smooth)):
        if abs(d1_smooth[i]) <= slow_threshold:
            consecutive += 1
            if consecutive >= min_slow_consecutive:
                knee_idx = max(0, i - min_slow_consecutive + 1)
                return float(y[knee_idx])
        else:
            consecutive = 0

    return float(y[-1])


def estimate_swdp_k_geometric(y: np.ndarray) -> float:
    """Geometric chord-distance knee detector (project-specific comparator)."""
    if len(y) < 3:
        return float(y[-1]) if len(y) else np.nan
    n = len(y) - 1
    xx = np.arange(n + 1)
    denom = np.hypot(n, y[-1] - y[0])
    if denom == 0:
        return float(y[0])
    dist = np.abs((y[-1] - y[0]) * xx - n * (y - y[0])) / denom
    return float(y[int(np.nanargmax(dist))])


def run_physics_fc_estimation(df_clean: pd.DataFrame = None,
                              cycles_df: pd.DataFrame = None,
                              save_outputs: bool = True):
    if df_clean is None:
        df = pd.read_csv(RESULTS_DIR / 'cleaned_vmc_data.csv')
    else:
        df = df_clean.copy()

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')

    if cycles_df is None:
        cy = pd.read_csv(RESULTS_DIR / 'valid_recession_cycles.csv')
    else:
        cy = cycles_df.copy()

    valid_cycles = cy[cy['Is_Valid'].astype(bool)].copy()

    rows = []
    for _, c in valid_cycles.iterrows():
        col = c['Column']
        col_smooth = col + '_smoothed'
        sched_time = pd.to_datetime(c['Scheduled_Irrigation_Time'])
        peak_time = pd.to_datetime(c['Peak_Time'])
        overnight_start = pd.to_datetime(c['Overnight_Start_Time'])

        # 1. SWDP-R: Peak-Relative Inverse-Time Regression
        night_df = df[(df['Timestamp'] >= overnight_start) & (df['Timestamp'] <= sched_time)].copy()

        fc_r = np.nan
        swdp_r_r2 = np.nan
        swdp_r_rmse = np.nan
        swdp_r_mae = np.nan
        swdp_r_bias = np.nan
        swdp_r_slope = np.nan
        swdp_r_intercept = np.nan
        swdp_r_n = len(night_df)
        swdp_r_duration = float((night_df['Elapsed_hours'].iloc[-1] - night_df['Elapsed_hours'].iloc[0])) if len(night_df) > 1 else 0.0
        swdp_r_valid = True
        swdp_r_failure_reason = "Valid"

        if swdp_r_n < MIN_SWPDR_POINTS:
            swdp_r_valid = False
            swdp_r_failure_reason = f"Insufficient overnight points (n={swdp_r_n} < {MIN_SWPDR_POINTS})"
        else:
            y_r = night_df[col_smooth].to_numpy(float)

            peak_row = df[df['Timestamp'] == peak_time]
            if len(peak_row) == 0:
                peak_row = df.iloc[[np.argmin(np.abs((df['Timestamp'] - peak_time).dt.total_seconds()))]]
            peak_elapsed_hours = float(peak_row['Elapsed_hours'].iloc[0])

            # tau_i = t_i - t_peak > 0
            tau_r = (night_df['Elapsed_hours'] - peak_elapsed_hours).to_numpy(float)
            valid_mask = tau_r > 0
            tau_r = tau_r[valid_mask]
            y_r = y_r[valid_mask]
            swdp_r_n = len(tau_r)

            if swdp_r_n < MIN_SWPDR_POINTS:
                swdp_r_valid = False
                swdp_r_failure_reason = f"Insufficient post-peak overnight points (n={swdp_r_n} < {MIN_SWPDR_POINTS})"
            else:
                # Automated Assertion: tau > 0 and 1/tau is finite
                assert all(tau_r > 0), "SWDP-R contains non-positive tau values."
                inv_tau = 1.0 / tau_r
                assert all(np.isfinite(inv_tau)), "SWDP-R contains non-finite 1/tau values."

                try:
                    poly = np.polyfit(inv_tau, y_r, 1)
                    swdp_r_slope = float(poly[0])
                    swdp_r_intercept = float(poly[1])
                    fc_r = swdp_r_intercept

                    pred_r = np.polyval(poly, inv_tau)
                    res = y_r - pred_r
                    ss_res = float(np.sum(res ** 2))
                    ss_tot = float(np.sum((y_r - np.mean(y_r)) ** 2))
                    swdp_r_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
                    swdp_r_rmse = float(np.sqrt(np.mean(res ** 2)))
                    swdp_r_mae = float(np.mean(np.abs(res)))
                    swdp_r_bias = float(np.mean(res))

                    if fc_r <= 0 or fc_r > c['Peak_VMC']:
                        swdp_r_valid = False
                        swdp_r_failure_reason = (
                            f"Asymptotic FC ({fc_r:.4f}) outside plausible range "
                            f"[0, {c['Peak_VMC']:.4f}]"
                        )
                except Exception as e:
                    swdp_r_valid = False
                    swdp_r_failure_reason = str(e)

        # 2. R-FP-style & Rate-Based SWDP-K: Full post-peak recession
        full_rec_df = df[(df['Timestamp'] >= peak_time) & (df['Timestamp'] <= sched_time)].copy()
        y_full = full_rec_df[col_smooth].to_numpy(float)
        t_full = full_rec_df['Elapsed_hours'].to_numpy(float)

        fc_rfp, _ = estimate_rfp_plateau(y_full)
        fc_swdp_k_rate = estimate_swdp_k_rate(y_full, t_full)
        fc_swdp_k_geom = estimate_swdp_k_geometric(y_full)
        fc_direct = float(c['FC_Pre_Irrigation_VMC'])

        # 3. Method spread and consensus index
        candidate_vals = [v for v in [fc_direct, fc_rfp, fc_r, fc_swdp_k_rate] if np.isfinite(v)]
        method_spread = (max(candidate_vals) - min(candidate_vals)) if candidate_vals else np.nan
        consensus_index = float(np.median(candidate_vals)) if candidate_vals else fc_direct

        rows.append({
            'Event_ID': int(c['Event_ID']),
            'Column': col,
            'Depth_cm': int(c['Depth_cm']),
            'Peak_VMC': float(c['Peak_VMC']),
            'FC_Direct_SensorReference': fc_direct,
            'FC_SWDP_R': fc_r,
            'SWDP_R_R2': swdp_r_r2,
            'SWDP_R_RMSE': swdp_r_rmse,
            'SWDP_R_MAE': swdp_r_mae,
            'SWDP_R_Bias': swdp_r_bias,
            'SWDP_R_n': swdp_r_n,
            'SWDP_R_duration_hours': swdp_r_duration,
            'SWDP_R_slope': swdp_r_slope,
            'SWDP_R_intercept': swdp_r_intercept,
            'SWDP_R_valid': swdp_r_valid,
            'SWDP_R_failure_reason': swdp_r_failure_reason,
            'FC_R_FP': fc_rfp,
            'FC_SWDP_K_Rate': fc_swdp_k_rate,
            'FC_SWDP_K_Geometric': fc_swdp_k_geom,
            'Method_Spread': method_spread,
            'FC_Consensus_Index': consensus_index
        })

    out = pd.DataFrame(rows)

    if save_outputs:
        out.to_csv(RESULTS_DIR / 'physics_fc_event_estimates.csv', index=False)

        summary = out.groupby('Depth_cm').agg(
            Valid_Cycles=('Event_ID', 'count'),
            FC_Direct_Mean=('FC_Direct_SensorReference', 'mean'),
            FC_Direct_Std=('FC_Direct_SensorReference', 'std'),
            FC_SWDP_R_Mean=('FC_SWDP_R', 'mean'),
            FC_SWDP_R_Std=('FC_SWDP_R', 'std'),
            SWDP_R_R2_Median=('SWDP_R_R2', 'median'),
            FC_R_FP_Mean=('FC_R_FP', 'mean'),
            FC_SWDP_K_Rate_Mean=('FC_SWDP_K_Rate', 'mean'),
            Method_Spread_Mean=('Method_Spread', 'mean'),
            FC_Consensus_Mean=('FC_Consensus_Index', 'mean')
        ).reset_index()

        summary['Direct_CV_pct'] = 100.0 * summary['FC_Direct_Std'] / summary['FC_Direct_Mean']
        summary['SWDP_R_CV_pct'] = 100.0 * summary['FC_SWDP_R_Std'] / summary['FC_SWDP_R_Mean']
        summary.to_csv(RESULTS_DIR / 'physics_fc_summary_by_depth.csv', index=False)

        print("Physics-based FC estimation complete. Saved to Results/physics_fc_event_estimates.csv")
        print(summary[['Depth_cm', 'Valid_Cycles', 'FC_Direct_Mean', 'FC_SWDP_R_Mean', 'SWDP_R_R2_Median', 'Method_Spread_Mean']].to_string(index=False))
        return out, summary

    return out, None


if __name__ == '__main__':
    run_physics_fc_estimation()
