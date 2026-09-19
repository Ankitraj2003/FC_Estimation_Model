# Scientifically Audited and Defensible In-Situ Field Capacity Estimation Pipeline

**Project Location:** `D:\M.Tech_final\Antigravity\Phase_2.1`  
**Scientific Objective:** Strict peer-reviewed vadose-zone methodology (Bean et al. 2018; Fazackerley & Lawrence 2012; Jiang et al. 2025; Zhou et al. 2025). Process-guided residual machine learning evaluated under hydrological-event temporal validation.

---

## 1. Executive Summary & Research Question
> **Can a process-based soil-water recession model provide a robust baseline for in-situ field-capacity estimation, and does leakage-controlled residual machine learning improve that baseline across independent hydrological events?**

This pipeline implements a **process-guided residual hybrid architecture**:

$$\hat{\theta}_{\text{FC}} = \text{FC}_{\text{SWDP-R}} + f_{\text{CatBoost}}(\mathbf{X})$$

- $\text{FC}_{\text{SWDP-R}}$: Physical baseline derived from Bean et al. (2018) peak-relative nocturnal recession regression: $\theta_i = a \cdot (1 / \tau_i) + \text{FC}_{\text{SWDP-R}}$, where $\tau_i = t_i - t_{\text{peak}} > 0$.
- $f_{\text{CatBoost}}(\mathbf{X})$: Regularized gradient-boosted decision tree predicting only the residual discrepancy against pre-irrigation sensor references ($\text{FC}_{\text{Direct\_SensorReference}}$).

---

## 2. Experimental Verification & Baseline Challenge

### Chronological Future-Event Holdout (Event 15, n=4 depths)

| Model / Architecture | Validation Protocol | Test R² | Test RMSE (m³/m³) | Test MAE (m³/m³) | Test Bias (m³/m³) | ΔRMSE vs SWDP-R (m³/m³) | Classification Accuracy | Macro Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SWDP-R Physics Baseline** | Chronological Holdout (Event 15) | **0.9997** | **0.00061** | **0.00049** | **+0.00046** | 0.00000 (Reference) | **100.0%** | **100.0%** |
| **R-FP-style Plateau Estimator** | Chronological Holdout (Event 15) | 0.9995 | 0.00077 | 0.00052 | +0.00052 | +0.00016 | 100.0% | 100.0% |
| **Process-Guided Hybrid CatBoost** | Chronological Holdout (Event 15) | 0.9962 | 0.00205 | 0.00202 | +0.00202 | +0.00145 | 100.0% | 100.0% |
| **Rate-Based Recession Knee** | Chronological Holdout (Event 15) | 0.9954 | 0.00226 | 0.00167 | +0.00167 | +0.00166 | 100.0% | 100.0% |
| **Direct CatBoost (Pure ML)** | Chronological Holdout (Event 15) | 0.3818 | 0.02615 | 0.02164 | -0.00637 | +0.02554 (Degradation) | 75.0% | 55.6% |

### Event-Level Model Comparison

| Event ID | Partition | Depths (n) | SWDP-R RMSE | Hybrid RMSE | ΔRMSE (Hybrid - SWDP) | Preferred Model |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Event 3** | Training | 1 | 0.00121 | 0.00045 | -0.00076 | **Hybrid** (63% error reduction) |
| **Event 4** | Training | 2 | 0.00401 | 0.00205 | -0.00196 | **Hybrid** (49% error reduction) |
| **Event 10** | Training | 1 | 0.00095 | 0.00030 | -0.00065 | **Hybrid** (68% error reduction) |
| **Event 14** | Training | 3 | 0.00042 | 0.00061 | +0.00019 | **SWDP-R** |
| **Event 15** | Holdout Test | 4 | 0.00061 | 0.00205 | +0.00145 | **SWDP-R** |

### Key Scientific Takeaways:
1. **Catastrophic Failure of Pure ML:** Without a physical baseline, direct empirical regression collapses on unseen future events ($R^2 = 0.3818$, RMSE = $0.02615\text{ m}^3/\text{m}^3$).
2. **Dominance of the SWDP-R Physics Baseline:** The peak-relative SWDP-R inverse-time model demonstrates exceptional physical accuracy ($R^2 = 0.9997$, RMSE = $0.00061\text{ m}^3/\text{m}^3$), operating at the sensor noise floor.
3. **Selective Residual Added Value:** On events with initial drainage discrepancies (Events 3, 4, 10), the hybrid model cuts error by 49% to 68%. On events where the physical baseline is already accurate to $<0.0006\text{ m}^3/\text{m}^3$, SWDP-R is preferred.

---

## 3. Core Methodological Corrections Implemented

1. **Peak-Relative SWDP-R Time Origin (Critical Fix #1):**
   - Transformed: $\tau_i = t_i - t_{\text{peak}} > 0$.
   - Predictor: $x_i = 1 / \tau_i$.
   - Model: $\theta_i = a \cdot x_i + \text{FC}_{\text{SWDP-R}}$. Asymptotic FC is the y-intercept.
2. **Rate-Based Recession Knee Estimator (Critical Fix #2):**
   - Forward-search heuristic scanning chronologically for the first persistent rapid-to-slow drainage rate transition.
3. **R-FP Terminology & Framing (Critical Fix #3):**
   - Renamed to "R-FP-style Recession Plateau Estimator" to faithfully describe the sequential low-derivative search.
4. **Nested Fold-Specific / Nested Out-of-Sample SAX Architecture (Critical Fix #4):**
   - **Self-similarity bias eliminated:** Each training event $E \in \{A, B, C, D\}$ queries only $\{ \text{train\_events} \setminus \{E\} \}$. Prior implementation queried itself, inflating similarity to 1.0 artificially.
   - **Symmetric construction:** The held-out validation event $V$ queries all training events $\{A, B, C, D\}$, creating mathematical parity between training and validation feature spaces.
   - **Zero global SAX features:** No static global SAX columns are present in the ML base feature matrix — all SAX features are dynamically rebuilt per fold.
   - **No synthetic imputation:** Short series ($< \text{word\_length}$) emit `None`; empty reference libraries emit `NaN`. CatBoost handles missing values natively via `nan_mode='Min'`.
   - **Deliberate negative test:** The SAX constructor raises an `AssertionError` if a validation event or self-event is injected into the reference library, verified in `05_leakage_audit.py` Test 7.
   - Standardized feature name: `LOEO_SAX_Hamming_Similarity`.
5. **One-Factor-at-a-Time (OFAT) Production Sensitivity Analysis (Critical Fix #5):**
   - Re-executed pipeline across 21 real parameter configurations via exact production functions (gap thresholds, SG smoothing, IETD, SAX parameters, rebound tolerances).
6. **Computational Leakage & Provenance Audit:**
   - Programmatically audited using set-theoretic verification ($|\text{Train} \cap \text{Test}| = 0$, feature provenance graph, SAX fold provenance audit, and hyperparameter provenance).
   - All 7 audit tests PASSED. Pre/Post SAX comparison preserved in `Results/pre_post_sax_leakage_fix_comparison.csv`.

---

## 4. End-to-End Pipeline Execution

Execute all 9 scripts sequentially via the master orchestrator:
```powershell
python Code/run_pipeline.py
```

### Pipeline Script Architecture:
1. `config.py`: Authoritative centralized configuration file.
2. `01_data_preprocessing.py`: Short-gap interpolation ($\le 2\text{ h}$), segment tracking, continuous SG smoothing.
3. `02_event_detection.py`: 24h IETD event detection and Bean Criteria A–E screening.
4. `03_fc_estimation_physics.py`: Peak-relative SWDP-R regression, R-FP plateau, Rate-Based Knee.
5. `04b_r_sax_cycle_analysis.py`: Leakage-controlled SAX symbolic representation (`LOEO_SAX_Hamming_Similarity`).
6. `04_fc_estimation_catboost.py`: Process-guided residual CatBoost, fold-rebuilt SAX LOEO-CV, and holdout test.
7. `05_leakage_audit.py`: Computational set-theoretic audit of all forms of methodological leakage.
8. `06_sensitivity_and_ablation.py`: Model ablation (A1–A5) and OFAT sensitivity analysis (S1–S6).
9. `07_visualizations.py`: Dynamic generation and verification of publication-grade figures in `Plots/`.
10. `08_generate_reports.py`: Master multi-sheet Excel workbook and comprehensive Markdown report.

---

## 5. Peer-Reviewed References
1. Bean, E.Z., Huffaker, R.G., & Migliaccio, K.W. (2018). *Vadose Zone Journal*, 17(1), 180073.
2. Fazackerley, S., & Lawrence, R. (2012). *Irrigation and Drainage*, 61(3), 416–424.
3. Savitzky, A., & Golay, M.J.E. (1964). *Analytical Chemistry*, 36(8), 1627–1639.
4. Lin, J., Keogh, E., Wei, L., & Lonardi, S. (2007). *Data Mining and Knowledge Discovery*, 15(2), 107–144.
5. Roberts, D.R., et al. (2017). *Ecography*, 40(8), 913–929.
6. USDA-NRCS. *Soil Quality Indicators: Available Water Capacity.*
7. Jiang, X., et al. (2025). *Advances in Water Resources*, 206, 105177.
8. Zhou, J., et al. (2025). *Journal of Hydrology*, 652, 132639.
9. Li, Y., et al. (2026). *Agricultural Communications*, 4(1), 100156.
