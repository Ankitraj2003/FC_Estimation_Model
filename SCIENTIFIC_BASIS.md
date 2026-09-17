# Scientific basis – Phase 2.2 final hybrid FC estimator

## 1. Field capacity concept
Field capacity is treated as a drainage-state quantity: water remaining after rapid gravitational drainage has substantially diminished. USDA-NRCS guidance states that FC depends on soil texture/structure and that drainage time is not universally fixed. Therefore this project does not impose an arbitrary universal 24/48-hour FC time.

## 2. Event and cycle validation
The governing time-series reference is Bean, Huffaker & Migliaccio (2018), *Estimating Field Capacity from Volumetric Soil Water Content Time Series Using Automated Processing Algorithms*, Vadose Zone Journal, DOI 10.2136/vzj2018.04.0073. The implementation requires a measured rainfall/irrigation event, a sensor response of at least 0.006 m3/m3, no additional input between 22:00 and the scheduled irrigation time, and a steady overnight recession. This prevents VMC-only false events and contaminated recession cycles.

## 3. Physics estimators
The pipeline calculates:
- an expert-style direct FC point immediately before scheduled irrigation;
- an R-FP-style peak/plateau recession estimate;
- an SWDP-R inverse-time recession regression;
- SWDP-K as a sensitivity knee estimate.
Bean et al. (2018) provide the scientific basis for using soil-water dynamics and recession behavior, including SWDP-R and R-FP-type approaches. Fazackerley & Lawrence (2012) independently support drainage-model-based in-situ FC estimation using soil-moisture sensors.

## 4. Hybrid physics-informed CatBoost
The final predictive model is deliberately a **hybrid physics-informed model** rather than a pure black-box ML model. The target is the expert-style direct pre-irrigation FC point. Predictors include depth, event water input, wetting rise, recession duration, recession slopes, overnight depletion, early recession VMC, and the independent physics-derived R-FP and SWDP-R estimates.

The direct FC target value is NOT included as a predictor. This avoids the obvious target-copying error. The physics estimates are retained because they are scientifically derived summaries of the same recession process that field-capacity theory uses.

CatBoost is regularized (300 trees, depth 2, learning rate 0.03, L2=5) because the dataset has only six independent valid hydrological events. The 2026 PSO-CatBoost paper supports CatBoost as a candidate for sensor-based FC estimation, but PSO optimization is intentionally not claimed here because this dataset is too small to justify a large hyperparameter search without an independent validation set.

## 5. Time-series validation
Random K-fold CV is not used. The last two complete hydrological events are held out chronologically. Training therefore uses only earlier events and testing uses future events. This is the appropriate deployment-style validation for an event time series.

## 6. Current tested result
- Valid depth-events: 13
- Independent hydrological events represented: 6
- Training events: 4
- Future test events: 2 (IDs 23 and 24)
- Training samples: 7
- Test samples: 6
- **Chronological hybrid test R2: 0.9972**
- Hybrid Test RMSE: 0.00172 m3/m3
- Hybrid Test MAE: 0.00153 m3/m3
- Hybrid Test bias: -0.00153 m3/m3
- SWDP-R baseline test R2: 0.9994
- SWDP-R baseline RMSE: 0.00077 m3/m3

The training R2 (~1.00) is only a fit diagnostic and must not be presented as model accuracy.

## 7. Important limitation
The test set contains only two independent future events. Therefore 0.9972 is a promising chronological holdout result, not a universal or laboratory-validated accuracy. Independent expert-labelled or laboratory FC observations are still required for external validation. The hybrid model is scientifically stronger than the previous target-ensemble approach because it predicts the expert-style FC point using physics-derived recession descriptors without copying that target into the features.

## References
1. Bean, G., Huffaker, R.G., & Migliaccio, K.W. (2018). Estimating Field Capacity from Volumetric Soil Water Content Time Series Using Automated Processing Algorithms. Vadose Zone Journal. DOI: 10.2136/vzj2018.04.0073.
2. Fazackerley, S., & Lawrence, R. (2012). Automatic In Situ Determination of Field Capacity Using Soil Moisture Sensors. Irrigation and Drainage, 61, 416–424. DOI: 10.1002/ird.646.
3. USDA-NRCS. Soil Quality Indicators / Available Water Capacity guidance: field capacity is water remaining after thorough saturation and free drainage, with drainage time dependent on soil properties.
4. Wang et al. (2026). A PSO-CatBoost Algorithm Based on Soil Sensor Data: A Novel In-Situ Intelligent Method for Estimating Field Capacity. Agricultural Communications. DOI: 10.1016/j.agrcom.2026.100156.

## 8. Physics-first residual correction
The final ML architecture uses SWDP-R as the primary physics estimator and CatBoost only to predict the residual between the expert-style pre-irrigation FC point and SWDP-R. This follows a physics-informed/hybrid modelling principle: a process-based estimate supplies the physically meaningful baseline and machine learning models only the remaining systematic error. This is preferable to replacing a validated drainage estimator with a black-box model when the available dataset is small.

The test report therefore gives both the hybrid metrics and the SWDP-R baseline metrics. A high hybrid R² is not interpreted as independent laboratory accuracy because the target and physics estimator are derived from the same soil-water recession record. External expert/laboratory FC observations remain necessary for independent validation.

## 9. Scientific guardrails
No random K-fold cross-validation is used. Future events are held out chronologically. No arbitrary accuracy target is imposed. The hybrid is accepted because it has a process-based baseline, uses only physically meaningful recession descriptors, and reports its performance alongside the underlying SWDP-R estimator rather than hiding the baseline.

## 10. Additional hybrid-model literature
The residual-correction architecture is also consistent with recent hydrological physics-informed ML literature. Bhasme et al. describe hybrid process/ML modelling as a way to retain process understanding while using ML predictive flexibility. A 2025 *Advances in Water Resources* study by Jiang et al. explicitly evaluates ML error correction of a process-based hydrological model and reports that process/error-correction hybridisation can improve predictive performance while retaining process interpretability. A 2025 *Engineering Applications of Artificial Intelligence* study on soil-water content likewise uses a hydrological model as a baseline and trains ML models on residuals before combining the correction with the physical prediction. These papers support the architectural choice; they do not prove that a particular FC dataset must achieve a particular R².

Additional references:
5. Bhasme, P., Vagadiya, J., & Bhatia, U. (2022). Enhancing predictive skills in physically-consistent way: Physics Informed Machine Learning for hydrological processes. *Journal of Hydrology*, 612, 128145. https://doi.org/10.1016/j.jhydrol.2022.128145.
6. Jiang, X., Hu, L., Fu, X., Gupta, H., Xu, Y., Zhao, C., Zhang, G., & Lu, M. (2025). A novel hybrid framework for combining process-based models with machine learning for streamflow prediction. *Advances in Water Resources*, 206, 105177. https://doi.org/10.1016/j.advwatres.2025.105177.
7. (2025). A hybrid time series and physics-informed machine learning framework to predict soil water content. *Engineering Applications of Artificial Intelligence*, 144, 110105. https://doi.org/10.1016/j.engappai.2025.110105.
8. Zhou, J., Huang, T., Wang, H., Du, W., Zhan, Y., Duan, A., & Fu, G. (2025). Using physical method, machine learning and hybrid method to model soil water movement. *Journal of Hydrology*, 652, 132639. https://doi.org/10.1016/j.jhydrol.2024.132639.
