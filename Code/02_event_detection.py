"""02_event_detection.py - Hydrological Event Detection and Cycle Screening

Follows the cycle-screening framework of Bean et al. (2018) for volumetric soil water content
time series, with explicit quality-control criteria:
- Criterion A (Wetting Magnitude): Sensor rise >= 0.006 m3/m3 (Bean et al. 2018 screening criterion).
- Criterion B (Recession Peak): Clear peak detected following wetting event before recession onset.
- Criterion C (Input Isolation): Zero additional rainfall or irrigation during the selected overnight
  drainage interval.
- Criterion D (Nocturnal Drainage Dominance): Selected overnight period exhibits monotonic declining
  moisture (negative slope and negative median step change). Nocturnal evapotranspiration and root water
  uptake are assumed to be sufficiently reduced that this period approximates drainage dominance.
- Criterion E (Rebound Tolerance): Maximum single-step rebound <= 0.002 m3/m3 (project-specific QC parameter).

Event Grouping: 24-hour Inter-Event Time Definition (IETD) baseline.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config import (
    RESULTS_DIR, IETD_HOURS, WETTING_THRESHOLD, REBOUND_THRESHOLD,
    NO_INPUT_START_HOUR, DEFAULT_IRR_HOUR
)


def parse_depth(col: str) -> int:
    """Extract sensor depth in cm from column name."""
    m = re.search(r'_(\d+)_m3', col)
    return int(m.group(1)) if m else int(re.search(r'(\d+)', col).group(1))


def detect_events(df: pd.DataFrame, ietd_hours: float = IETD_HOURS) -> pd.DataFrame:
    """Group measured water inputs into distinct hydrological events using an IETD rule.
    
    A wetting event must be supported by an observed precipitation or irrigation input.
    Synthetic events are not created from VMC rise alone.
    """
    mask = df['Total_Water_mm'] > 0
    idx = np.flatnonzero(mask.to_numpy())
    groups = []
    if len(idx):
        s = p = idx[0]
        for i in idx[1:]:
            time_gap = (df.loc[i, 'Timestamp'] - df.loc[p, 'Timestamp']).total_seconds() / 3600.0
            if time_gap <= ietd_hours:
                p = i
            else:
                groups.append((s, p))
                s = p = i
        groups.append((s, p))

    rows = []
    for eid, (s, e) in enumerate(groups, 1):
        g = df.iloc[s:e + 1]
        rows.append({
            'Event_ID': eid,
            'Start_Idx': s,
            'End_Idx': e,
            'Start_Time': g['Timestamp'].iloc[0],
            'End_Time': g['Timestamp'].iloc[-1],
            'Precipitation_mm': float(g['Precipitation_mm'].sum()),
            'Irrigation_mm': float(g['Irrigation_mm'].sum()),
            'Total_Water_mm': float(g['Total_Water_mm'].sum()),
            'Duration_Hours': float((g['Timestamp'].iloc[-1] - g['Timestamp'].iloc[0]).total_seconds() / 3600.0)
        })
    return pd.DataFrame(rows)


def get_scheduled_recession_window(peak_time: pd.Timestamp, scheduled_hour: int = DEFAULT_IRR_HOUR):
    """Identify the overnight drainage window following a wetting peak.
    
    The overnight window runs from 22:00 on the day of the wetting event to the scheduled
    morning irrigation (typically 07:00 next day).
    """
    base = peak_time.normalize() + pd.Timedelta(hours=NO_INPUT_START_HOUR)
    sched_end = peak_time.normalize() + pd.Timedelta(days=1, hours=scheduled_hour)
    if peak_time >= sched_end:
        base += pd.Timedelta(days=1)
        sched_end += pd.Timedelta(days=1)
    return base, sched_end


def run_event_detection(ietd_hours: float = IETD_HOURS,
                         wetting_thresh: float = WETTING_THRESHOLD,
                         rebound_thresh: float = REBOUND_THRESHOLD,
                         df_clean: pd.DataFrame = None,
                         save_outputs: bool = True):
    if df_clean is None:
        df = pd.read_csv(RESULTS_DIR / 'cleaned_vmc_data.csv')
    else:
        df = df_clean.copy()

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')

    sm_cols = sorted([c for c in df.columns if c.startswith('SM_') and not any(
        c.endswith(x) for x in ['_smoothed', '_d1', '_d2', '_Was_Interpolated', '_Segment_ID'])],
        key=parse_depth
    )

    events = detect_events(df, ietd_hours=ietd_hours)
    if events.empty:
        if save_outputs:
            raise RuntimeError('No measured water-input events found.')
        return pd.DataFrame(), pd.DataFrame()

    irr_hours = df.loc[df['Irrigation_mm'] > 0, 'Timestamp'].dt.hour
    scheduled_hour = int(irr_hours.mode().iloc[0]) if len(irr_hours) else DEFAULT_IRR_HOUR
    events['Scheduled_Irrigation_Hour'] = scheduled_hour

    cycles = []
    for _, ev in events.iterrows():
        s, e = int(ev['Start_Idx']), int(ev['End_Idx'])
        search_end = min(e + 36, len(df) - 1)

        for c in sm_cols:
            col_smooth = c + '_smoothed'
            x = df[col_smooth].to_numpy(float)

            # Pre-event baseline VMC
            baseline_window = x[max(0, s - 6):s + 1]
            baseline = float(np.nanmedian(baseline_window)) if len(baseline_window) else np.nan

            # Criterion B: Peak search following water input
            sub_x = x[s:search_end + 1]
            valid_sub = np.isfinite(sub_x)
            if not valid_sub.any():
                continue

            peaks, _ = find_peaks(sub_x, prominence=0.001, distance=3)
            cand = [s + int(p_i) for p_i in peaks if (x[s + int(p_i)] - baseline) >= wetting_thresh]
            if not cand:
                p_max = s + int(np.nanargmax(sub_x))
                cand = [p_max] if (x[p_max] - baseline) >= wetting_thresh else []

            if not cand:
                # Did not meet Criterion A / B
                continue

            peak_idx = cand[0]
            peak = float(x[peak_idx])
            peak_time = df.loc[peak_idx, 'Timestamp']
            rise = peak - baseline

            overnight_start, overnight_end = get_scheduled_recession_window(peak_time, scheduled_hour)
            night_mask = (df['Timestamp'] >= overnight_start) & (df['Timestamp'] <= overnight_end)
            night_df = df.loc[night_mask].copy()

            # Criterion C: Check for any water input during overnight recession
            night_input = float(night_df['Total_Water_mm'].sum()) if len(night_df) else np.nan

            # Full recession from peak to scheduled morning irrigation
            rec_mask = (df['Timestamp'] >= peak_time) & (df['Timestamp'] <= overnight_end)
            rec_df = df.loc[rec_mask].copy()
            rec_hours = float((rec_df['Elapsed_hours'].iloc[-1] - rec_df['Elapsed_hours'].iloc[0])) if len(rec_df) > 1 else 0.0

            # Screening evaluation
            is_valid = True
            rejection_reasons = []

            # Criterion A
            if rise < wetting_thresh:
                is_valid = False
                rejection_reasons.append(f"Wetting rise ({rise:.4f}) < threshold ({wetting_thresh})")

            # Criterion C
            if pd.isna(night_input) or night_input > 0.0:
                is_valid = False
                rejection_reasons.append(f"Rainfall/irrigation occurred during overnight recession ({night_input:.1f} mm)")

            # Criterion D & E: Monotonic decline and rebound tolerance
            slope_night = np.nan
            if len(night_df) >= 3:
                y_n = night_df[col_smooth].to_numpy(float)
                t_n = night_df['Elapsed_hours'].to_numpy(float)
                if len(y_n) > 1:
                    slope_night = float(np.polyfit(t_n, y_n, 1)[0])
                    dy = np.diff(y_n)
                    max_rebound = float(np.max(dy)) if len(dy) else 0.0
                    median_dy = float(np.median(dy)) if len(dy) else 0.0

                    if slope_night >= 0.0 or median_dy > 0.0:
                        is_valid = False
                        rejection_reasons.append(f"Overnight drying was not declining (slope={slope_night:.5f})")

                    if max_rebound > rebound_thresh:
                        is_valid = False
                        rejection_reasons.append(f"Nocturnal rebound ({max_rebound:.4f}) > tolerance ({rebound_thresh})")
            else:
                is_valid = False
                rejection_reasons.append(f"Insufficient overnight observations (n={len(night_df)})")

            fc_direct = float(rec_df[col_smooth].iloc[-1]) if len(rec_df) > 0 else np.nan

            cycles.append({
                'Event_ID': int(ev['Event_ID']),
                'Column': c,
                'Depth_cm': parse_depth(c),
                'Peak_Time': peak_time,
                'Peak_Idx': peak_idx,
                'Overnight_Start_Time': overnight_start,
                'Scheduled_Irrigation_Time': overnight_end,
                'Recession_End_Idx': rec_df.index[-1] if len(rec_df) else peak_idx,
                'Pre_Event_VMC': baseline,
                'Peak_VMC': peak,
                'VMC_Rise': rise,
                'Total_Water_mm': float(ev['Total_Water_mm']),
                'Recession_Hours': rec_hours,
                'Overnight_Input_mm': night_input,
                'Overnight_Slope': slope_night,
                'FC_Pre_Irrigation_VMC': fc_direct,
                'Is_Valid': is_valid,
                'Rejection_Reason': "; ".join(rejection_reasons) if not is_valid else "Valid"
            })

    cycles_df = pd.DataFrame(cycles)

    # Automated Assertions on Valid Cycles
    if not cycles_df.empty:
        valid_subset = cycles_df[cycles_df['Is_Valid']]
        for _, vc in valid_subset.iterrows():
            assert vc['VMC_Rise'] >= wetting_thresh, f"Valid cycle failed Criterion A: rise {vc['VMC_Rise']} < {wetting_thresh}"
            assert vc['Overnight_Input_mm'] == 0.0, f"Valid cycle failed Criterion C: overnight input {vc['Overnight_Input_mm']}"

    if save_outputs:
        events.to_csv(RESULTS_DIR / 'detected_events_summary.csv', index=False)
        cycles_df.to_csv(RESULTS_DIR / 'valid_recession_cycles.csv', index=False)
        valid_count = cycles_df['Is_Valid'].sum() if not cycles_df.empty else 0
        print(f"Event detection complete. Measured water events: {len(events)}; Depth cycles: {len(cycles_df)}; Valid: {valid_count}")
        if not cycles_df.empty:
            print("Cycle status breakdown:")
            print(cycles_df['Rejection_Reason'].value_counts().to_string())

    return events, cycles_df


if __name__ == '__main__':
    run_event_detection()
