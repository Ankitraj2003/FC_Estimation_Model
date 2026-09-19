"""04b_r_sax_cycle_analysis.py - Leakage-Controlled SAX Cycle Similarity

Following Lin et al. (2007) and Bean et al. (2018):
- Discretizes standardized soil moisture recession segments into symbolic SAX representations
  via Piecewise Aggregate Approximation (PAA) and Gaussian equiprobable breakpoints.
- Parameters: Word length w = 8, Alphabet size alpha = 5 (study-specific parameters).
- Leakage Guardrail:
  1. Training cycles query only other training cycles via Leave-One-Event-Out (LOEO).
  2. Future holdout test cycles query only the historical training library.
  Test cycles never contaminate the reference library or feature construction.
- Standardized Feature Nomenclature: LOEO_SAX_Hamming_Similarity.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import RESULTS_DIR, SAX_WORD_LENGTH, SAX_ALPHABET_SIZE, SAX_FEATURE_NAME, FINAL_HOLDOUT_EVENT


def compute_sax_word(series: np.ndarray,
                     word_size: int = SAX_WORD_LENGTH,
                     alphabet_size: int = SAX_ALPHABET_SIZE) -> str:
    """Transform a numerical time series into a SAX word.
    
    Robust validation:
    - Checks finite values, minimum series length >= word_size, word_size >= 2, alphabet_size >= 2.
    - Eliminates empty-slice warnings in PAA calculation.
    """
    if series is None or word_size < 2 or alphabet_size < 2:
        return None

    y = np.asarray(series, float)
    y = y[np.isfinite(y)]
    if len(y) < word_size:
        return None

    # Step 1: PAA (Piecewise Aggregate Approximation) with guaranteed non-empty slices
    edges = np.linspace(0, len(y), word_size + 1).astype(int)
    paa_means = []
    for i in range(word_size):
        start, end = edges[i], edges[i + 1]
        if start >= end:
            end = min(start + 1, len(y))
            start = max(0, end - 1)
        slice_vals = y[start:end]
        paa_means.append(float(np.mean(slice_vals)) if len(slice_vals) > 0 else float(y[-1]))
    paa_means = np.array(paa_means)

    # Step 2: Z-normalization
    std_val = float(np.std(paa_means))
    if std_val < 1e-12:
        z = np.zeros(word_size)
    else:
        z = (paa_means - np.mean(paa_means)) / std_val

    # Step 3: Equiprobable Gaussian breakpoints
    bp = norm.ppf(np.arange(1, alphabet_size) / alphabet_size)
    word = ''.join(chr(97 + int(np.sum(v > bp))) for v in z)
    return word


def compute_hamming_similarity(word_a: str, word_b: str) -> float:
    """Normalized character match similarity in [0, 1]."""
    if word_a is None or word_b is None:
        return np.nan
    n = min(len(word_a), len(word_b))
    if n == 0:
        return np.nan
    return float(np.mean([word_a[i] == word_b[i] for i in range(n)]))


def run_sax_cycle_analysis(df_clean: pd.DataFrame = None,
                           cycles_df: pd.DataFrame = None,
                           word_size: int = SAX_WORD_LENGTH,
                           alphabet_size: int = SAX_ALPHABET_SIZE,
                           holdout_event_id: int = FINAL_HOLDOUT_EVENT,
                           save_outputs: bool = True) -> pd.DataFrame:
    if df_clean is None:
        df = pd.read_csv(RESULTS_DIR / 'cleaned_vmc_data.csv')
    else:
        df = df_clean.copy()

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')

    if cycles_df is None:
        cy = pd.read_csv(RESULTS_DIR / 'valid_recession_cycles.csv')
    else:
        cy = cycles_df.copy()

    valid_cy = cy[cy['Is_Valid'].astype(bool)].copy()

    # Extract recession series and generate SAX words
    words = []
    for _, r in valid_cy.iterrows():
        col_sm = r['Column'] + '_smoothed'
        sched_time = pd.to_datetime(r['Scheduled_Irrigation_Time'])
        peak_time = pd.to_datetime(r['Peak_Time'])
        seg_df = df[(df['Timestamp'] >= peak_time) & (df['Timestamp'] <= sched_time)]
        y = seg_df[col_sm].to_numpy(float)

        w = compute_sax_word(y, word_size=word_size, alphabet_size=alphabet_size)
        words.append(w)

    valid_cy['SAX_Word'] = words

    # Build Training Reference Library
    train_cy = valid_cy[valid_cy['Event_ID'] != holdout_event_id].copy()
    test_cy = valid_cy[valid_cy['Event_ID'] == holdout_event_id].copy()

    training_library = {}
    for _, r in train_cy.iterrows():
        if r['SAX_Word'] is not None:
            training_library[(int(r['Event_ID']), r['Column'])] = r['SAX_Word']

    # Compute LOEO-isolated similarities
    sim_scores = []
    for _, r in valid_cy.iterrows():
        eid = int(r['Event_ID'])
        w = r['SAX_Word']
        if w is None:
            sim_scores.append(np.nan)
            continue

        if eid != holdout_event_id:
            # LOEO: Query against historical training library excluding own event
            ref_words = [tw for (te, _), tw in training_library.items() if te != eid]
        else:
            # Final holdout: Query strictly against training library
            ref_words = list(training_library.values())

        if ref_words:
            sims = [compute_hamming_similarity(w, rw) for rw in ref_words]
            sim_scores.append(float(np.nanmean(sims)))
        else:
            sim_scores.append(np.nan)

    valid_cy[SAX_FEATURE_NAME] = sim_scores

    out = valid_cy[['Event_ID', 'Column', 'Depth_cm', 'SAX_Word', SAX_FEATURE_NAME]].copy()

    if save_outputs:
        out.to_csv(RESULTS_DIR / 'r_sax_cycle_similarity.csv', index=False)
        print(f"SAX analysis complete (word_length={word_size}, alphabet={alphabet_size}).")
        print(f"Standardized feature: {SAX_FEATURE_NAME}")
        print(f"Reference library strictly isolated: {len(training_library)} cycles from training events.")

    return out


if __name__ == '__main__':
    run_sax_cycle_analysis()
