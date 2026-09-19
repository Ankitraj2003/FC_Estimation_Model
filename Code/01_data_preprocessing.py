"""01_data_preprocessing.py - Scientifically Defensible Preprocessing and Quality Control

Follows Savitzky & Golay (1964) for polynomial smoothing and numerical differentiation,
with explicit quality-control guardrails:
1. Short-gap-only interpolation (threshold <= 2.0 hours). Gaps exceeding the threshold
   remain NaN to prevent the creation of artificial drainage curves.
2. Interpolation never bridges across rainfall/irrigation events.
3. Continuous segments are identified and tracked via Continuous_Segment_ID.
4. Savitzky-Golay filtering and numerical derivatives are calculated strictly within
   continuous segments of valid data, never across missing gaps.
5. Preserves explicit audit columns: Was_Interpolated, Gap_Duration_hours, Continuous_Segment_ID.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import (
    INPUT_DATA_PATH, RESULTS_DIR,
    MAX_INTERPOLATION_GAP_HOURS, SG_WINDOW, SG_POLYORDER
)


def parse_depth(col: str) -> int:
    """Extract sensor depth in cm from column name."""
    m = re.search(r'_(\d+)_m3', col)
    return int(m.group(1)) if m else int(re.search(r'(\d+)', col).group(1))


def run_preprocessing(max_gap_hours: float = MAX_INTERPOLATION_GAP_HOURS,
                      sg_window: int = SG_WINDOW,
                      sg_poly: int = SG_POLYORDER,
                      input_file: Path = INPUT_DATA_PATH,
                      save_outputs: bool = True) -> pd.DataFrame:
    print(f"Loading raw data from: {input_file}")
    df = pd.read_excel(input_file)
    df.columns = [str(c).replace("'", '').strip() for c in df.columns]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').drop_duplicates('Timestamp', keep='first').reset_index(drop=True)

    # Automated Assertion: Timestamps must be strictly monotonic
    assert df['Timestamp'].is_monotonic_increasing, "Timestamps are not monotonically increasing."

    sm_cols = sorted([c for c in df.columns if c.startswith('SM_')], key=parse_depth)

    # Clean water input variables
    for c in ['Irrigation_mm', 'Precipitation_mm']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['Total_Water_mm'] = df['Irrigation_mm'] + df['Precipitation_mm']

    # Time tracking
    dt_sec = df['Timestamp'].diff().dt.total_seconds().to_numpy(copy=True)
    dt_sec[0] = np.nanmedian(dt_sec[1:]) if len(dt_sec) > 1 else 3600.0
    df['Gap_hours'] = dt_sec / 3600.0
    df['Elapsed_hours'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600.0

    # Flag wetting events to prevent interpolating across water inputs
    is_water_event = df['Total_Water_mm'] > 0

    qc_records = []

    # Process each soil moisture column
    for c in sm_cols:
        depth = parse_depth(c)
        raw_series = df[c].copy()

        # Step 1: Identify invalid physical range [0, 1]
        invalid_mask = (~raw_series.between(0.0, 1.0)) | raw_series.isna()
        raw_series.loc[invalid_mask] = np.nan
        n_invalid = int(invalid_mask.sum())

        # Step 2: Short-gap-only interpolation
        is_nan = raw_series.isna()
        was_interp = pd.Series(False, index=df.index)
        cleaned_series = raw_series.copy()

        # Find blocks of NaNs
        nan_blocks = (~is_nan).cumsum()
        for _, group in is_nan.groupby(nan_blocks):
            if group.all():
                start_idx = group.index[0]
                end_idx = group.index[-1]
                prev_idx = start_idx - 1
                next_idx = end_idx + 1
                if prev_idx >= 0 and next_idx < len(df):
                    t_gap = (df.loc[next_idx, 'Timestamp'] - df.loc[prev_idx, 'Timestamp']).total_seconds() / 3600.0
                    # Check if water input occurred in this gap
                    water_in_gap = is_water_event.iloc[prev_idx:next_idx + 1].any()
                    # Interpolate ONLY if gap <= max_gap_hours and no water input occurred
                    if (t_gap <= max_gap_hours) and (not water_in_gap) and (max_gap_hours > 0):
                        sub_t = df.loc[prev_idx:next_idx, 'Timestamp']
                        sub_v = df.loc[prev_idx:next_idx, c]
                        interp_vals = pd.Series(sub_v.to_numpy(), index=sub_t).interpolate(method='time')
                        cleaned_series.loc[start_idx:end_idx] = interp_vals.iloc[1:-1].to_numpy()
                        was_interp.loc[start_idx:end_idx] = True

        df[c] = cleaned_series
        df[f'{c}_Was_Interpolated'] = was_interp

        # Step 3: Segment identification for derivative and smoothing calculation
        valid_obs = df[c].notna()
        time_diff = df['Timestamp'].diff().dt.total_seconds() / 3600.0
        gap_limit = max_gap_hours if max_gap_hours > 0 else 1.0
        new_segment = (~valid_obs) | (time_diff > gap_limit)
        segment_id = new_segment.cumsum()
        df[f'{c}_Segment_ID'] = np.where(valid_obs, segment_id, -1)

        # Step 4: Savitzky-Golay smoothing and derivatives restricted to continuous segments
        smoothed = np.full(len(df), np.nan)
        d1 = np.full(len(df), np.nan)
        d2 = np.full(len(df), np.nan)

        for seg in df[f'{c}_Segment_ID'].unique():
            if seg == -1:
                continue
            idx = df.index[df[f'{c}_Segment_ID'] == seg]
            if len(idx) >= sg_window:
                y = df.loc[idx, c].to_numpy(float)
                t = df.loc[idx, 'Elapsed_hours'].to_numpy(float)
                sm = savgol_filter(y, window_length=sg_window, polyorder=sg_poly)
                grad1 = np.gradient(sm, t)
                grad2 = np.gradient(grad1, t)
                smoothed[idx] = sm
                d1[idx] = grad1
                d2[idx] = grad2
            elif len(idx) > 0:
                smoothed[idx] = df.loc[idx, c].to_numpy(float)
                if len(idx) >= 2:
                    t = df.loc[idx, 'Elapsed_hours'].to_numpy(float)
                    grad1 = np.gradient(smoothed[idx], t)
                    d1[idx] = grad1

        df[f'{c}_smoothed'] = smoothed
        df[f'{c}_d1'] = d1
        df[f'{c}_d2'] = d2

        qc_records.append({
            'Column': c,
            'Depth_cm': depth,
            'Total_Observations': len(df),
            'Invalid_or_Missing_Count': n_invalid,
            'Short_Gaps_Interpolated': int(was_interp.sum()),
            'Remaining_NaN_Count': int(df[c].isna().sum()),
            'Valid_Segments_Count': len([s for s in df[f'{c}_Segment_ID'].unique() if s != -1]),
            'Min_VMC': float(np.nanmin(df[c])),
            'Max_VMC': float(np.nanmax(df[c])),
            'Mean_VMC': float(np.nanmean(df[c])),
            'Std_VMC': float(np.nanstd(df[c]))
        })

    if save_outputs:
        out_csv = RESULTS_DIR / 'cleaned_vmc_data.csv'
        df.to_csv(out_csv, index=False)
        qc_df = pd.DataFrame(qc_records)
        qc_df.to_csv(RESULTS_DIR / 'data_quality_report.csv', index=False)
        print(f"Data preprocessing complete. Saved to {out_csv}")
        print(qc_df[['Column', 'Depth_cm', 'Invalid_or_Missing_Count', 'Short_Gaps_Interpolated', 'Remaining_NaN_Count']].to_string(index=False))

    return df


if __name__ == '__main__':
    run_preprocessing()
