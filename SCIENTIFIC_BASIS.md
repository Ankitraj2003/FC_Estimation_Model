# Scientific Basis & Literature Alignment

## 1. Central Research Question
> **Can a process-based soil-water recession model provide a robust baseline for in-situ field-capacity estimation, and does leakage-controlled residual machine learning improve that baseline across independent hydrological events?**

This question shifts the focus from metric optimization (e.g. arbitrarily chasing $R^2$) to evaluating whether hybrid process-guided machine learning provides genuine scientific and operational advantages over established physical drainage models.

---

## 2. Research Gaps Addressed
- **Gap 1 (Static vs Dynamic FC):** Conventional irrigation management assumes a single static laboratory water retention value (e.g. -33 kPa matric potential). In reality, field capacity is a dynamic drainage state governed by in-situ hydraulic conductivity and profile layering.
- **Gap 2 (Fixed-Time vs Dynamic Drainage Cessation):** Fixed-time drainage conventions (e.g. measuring arbitrarily at 24h or 48h) fail across variable soil textures. This work infers field capacity from observed soil-water recession dynamics.
- **Gap 3 (Pure ML vs Process-Guided Residual Learning):** Direct machine learning regression suffers from poor out-of-distribution generalization on small hydrological datasets. We employ process-guided residual learning where physics anchors the prediction.
- **Gap 4 (Random Observation CV vs Structured Event-Blocked CV):** Standard random K-fold cross-validation suffers severe data leakage in time-series data. We enforce hydrological-event-level blocking (Roberts et al. 2017).
- **Gap 5 (Symbolic Cycle Representation with Leakage Control):** Time-series shape features must be derived strictly from historical training events to prevent information leakage to future predictions.

---

## 3. Core Methodological Novelties
1. **Process-Guided Residual FC Estimation:** SWDP-R asymptotic drainage serves as the physical anchor; CatBoost models only systematic residual errors.
2. **Hydrological Event-Level Temporal Validation:** Observations are grouped and blocked by complete hydrological events rather than individual sensor records.
3. **Nested Fold-Specific / Nested Out-of-Sample SAX Representation:** Each training event queries a reference library of $\{\text{train\_events} \setminus \{E\}\}$ (self-excluded), while the validation event queries all training events. This eliminates artificial self-similarity bias (prior 1.0 inflation) and enforces symmetric feature distributions. Feature name: `LOEO_SAX_Hamming_Similarity`.
4. **Physically Defensible Nocturnal Recession Analysis:** Isolates nocturnal intervals where evapotranspiration is sufficiently attenuated to ensure drainage dominance.
5. **Explicit Scientific Baseline Challenge:** Directly reports $\Delta\text{RMSE} = \text{RMSE}_{\text{Hybrid}} - \text{RMSE}_{\text{SWDP-R}}$ to verify whether machine learning truly adds value.

---

## 4. Literature-Supported vs Project-Specific Component Matrix

| Component | Literature-Supported Basis | Project-Specific Adaptation |
| :--- | :--- | :--- |
| **Data Smoothing & Derivatives** | Savitzky & Golay (1964) | Window $w=7$, Polynomial $p=2$; restricted to continuous segments |
| **Wetting Magnitude Threshold** | Bean et al. (2018) | $\Delta\theta \ge 0.006\text{ m}^3/\text{m}^3$ applied to multi-depth profiling probes |
| **Event Grouping** | Hydrological precipitation event standards | 24-hour Inter-Event Time Definition (IETD) baseline |
| **Valid-Cycle Screening** | Bean et al. (2018) automated cycle screening | Criteria A–E (nocturnal decline, rebound tolerance $\le 0.002$) |
| **SWDP-R Recession Model** | Bean et al. (2018); Fazackerley & Lawrence (2012) | Peak-relative time $\tau = t - t_{\text{peak}} > 0$, $1/\tau$ transformation, $n \ge 4$ |
| **R-FP Plateau Estimator** | Bean et al. (2018) findpeaks concept | First sustained low-derivative plateau after wetting peak |
| **SWDP-K Knee Estimator** | Bean et al. (2018) drainage inflection | Rate-based forward-search transition heuristic |
| **SAX Symbolic Representation** | Lin et al. (2007) | $w=8, \alpha=5$; historical training library querying (`LOEO_SAX_Hamming_Similarity`) |
| **Hybrid Residual Modeling** | Jiang et al. (2025); Zhou et al. (2025) | Soil-water FC residual formulation with CatBoost |
| **Structured Validation** | Roberts et al. (2017) | Leave-One-Event-Out CV + Chronological Holdout (Event 15) |

---

## 5. Explicit Assumptions & Limitations

### Assumptions:
- **A1 (Drainage Dominance):** Post-wetting nocturnal recession is assumed to be dominated by vertical redistribution.
- **A2 (Reduced Nocturnal ET):** Nighttime evapotranspiration and root water uptake are assumed to be sufficiently attenuated, not zero.
- **A3 (Sensor Stability):** Calibrated capacitance sensor responses reflect true volumetric water changes.
- **A4 (Hydraulic Boundary):** Shallow water table is assumed not to impede free gravitational drainage.
- **A5 (Residual Learnability):** Residual discrepancies between asymptotic drainage and operational pre-irrigation states are learnable from early-event descriptors.
- **A6 (Event Independence):** Storm/irrigation episodes separated by $\ge 24\text{ h}$ represent distinct hydrological validation units.

### Limitations:
- Chronological holdout evaluation is limited to one independent future event (Event 15, $n=4$ depths).
- Field capacity reference is sensor-derived immediately prior to scheduled irrigation rather than determined by independent laboratory extraction.
- Single sensor nest at a single experimental field site.
- Heuristic knee detection uses forward search rather than MATLAB's findchangepts.
