# Field Capacity Estimation – Phase 2.2 Final Hybrid Model

## Scientific basis
This implementation follows Bean et al. (2018) for soil-water time-series cycle validation and recession-based FC estimation, supported by Fazackerley & Lawrence (2012) and USDA-NRCS drainage-based FC guidance.

## Final model
The final model is a physics-informed CatBoost hybrid. The target is the expert-style direct FC point immediately before scheduled irrigation. Predictors include depth, rainfall/irrigation amount, VMC wetting response, recession duration and slopes, overnight depletion, early recession VMC, and R-FP/SWDP-R physics estimates. The direct FC target is excluded from predictors.

## Time-series validation
Random K-fold CV is not used. The final complete hydrological events are held out chronologically as future events.

- Training events: 4
- Test events: 2
- Training samples: 7
- Test samples: 6
- Test event IDs: 23,24
- **Chronological test R²: 0.9972**
- Test RMSE: 0.0017 m³/m³
- Test MAE: 0.0015 m³/m³
- Test Bias: -0.0015 m³/m³
- Training R² (fit diagnostic only): 1.0000
- SWDP-R baseline test R²: 0.9994
- SWDP-R baseline test RMSE: 0.00077 m³/m³

## Interpretation
R² = 0.9972 is the chronological future-event holdout result for the residual-correction hybrid. The SWDP-R baseline is also reported because the hybrid must demonstrate value beyond the physics estimator. It is not laboratory accuracy because only two independent future events are available and the FC target is an expert-style sensor-derived reference. Independent expert or laboratory FC measurements are still required for external validation.

## References
- Bean, G., Huffaker, R.G., & Migliaccio, K.W. (2018). Vadose Zone Journal. DOI 10.2136/vzj2018.04.0073.
- Fazackerley, S. & Lawrence, R. (2012). Irrigation and Drainage 61:416–424. DOI 10.1002/ird.646.
- USDA-NRCS soil-water/available-water guidance.
- Wang et al. (2026). PSO-CatBoost sensor-based FC estimation. DOI 10.1016/j.agrcom.2026.100156.
- Bhasme, Vagadiya & Bhatia (2022). Physics-informed machine learning for hydrological processes. DOI 10.1016/j.jhydrol.2022.128145.
- Jiang et al. (2025). Hybrid process-based model + ML error correction. DOI 10.1016/j.advwatres.2025.105177.
- (2025). Hybrid time-series and physics-informed ML for soil water content. DOI 10.1016/j.engappai.2025.110105.
- Zhou et al. (2025). Physical, ML and hybrid methods for soil water movement. DOI 10.1016/j.jhydrol.2024.132639.
