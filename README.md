# Phase 2.2 – Scientifically Revised Field Capacity Estimation

This project estimates field capacity (FC) from volumetric soil-water time series using hydrologically validated recession cycles and a physics-informed CatBoost hybrid.

## Pipeline
1. Temporal/sensor QC and elapsed-time derivatives
2. Rainfall/irrigation-driven event detection
3. Literature-based valid-cycle screening
4. Direct FC, R-FP, SWDP-R and SWDP-K estimates
5. R-SAX diagnostic (not used as an unsupervised truth label)
6. Physics-informed CatBoost
7. Chronological future-event holdout evaluation
8. Plots and Excel report

## Final tested performance
- Chronological hybrid test R²: **0.9972**
- Hybrid RMSE: **0.00172 m³/m³**
- Hybrid MAE: **0.00153 m³/m³**
- Hybrid Bias: **-0.00153 m³/m³**
- SWDP-R baseline test R²: **0.9994**
- SWDP-R baseline RMSE: **0.00077 m³/m³**
- Training events: 4
- Future test events: 2 (23 and 24)
- Test samples: 6

These metrics are a future-event holdout result, not laboratory accuracy. Only two independent test events are available, so external validation with expert/manual or laboratory FC observations is still required.

## Key scientific references
See `SCIENTIFIC_BASIS.md` and `REFERENCES.md`.

- Bean et al. (2018), DOI 10.2136/vzj2018.04.0073
- Fazackerley & Lawrence (2012), DOI 10.1002/ird.646
- USDA-NRCS soil-water guidance
- Wang et al. (2026), DOI 10.1016/j.agrcom.2026.100156
