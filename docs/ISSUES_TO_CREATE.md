# Issues to open on GitHub

Create these in the repository's *Issues* tab so the roadmap is visible
(this file can be deleted afterwards).

1. **Loader for the public "Cold Chain Milk" logs** (`enhancement`) -
   convert the Kaggle logs to the 5-minute CSV format and reproduce the
   early-warning comparison of the thesis paper on real data.
2. **Tier 2 classifiers** (`enhancement`) - Random Forest and Gradient
   Boosting on the features from `features.py`, as an optional
   `scikit-learn` extra.
3. **Tune the cascade thresholds tau0 and tau1** (`enhancement`) - measure
   how the trigger changes detection rate and computing cost.
4. **Grouped cross-validation** (`enhancement`) - replace the single
   calibration / held-out split by repeated leave-series-out folds.
5. **Resampling of raw sensor exports** (`enhancement`) - irregular
   timestamps, gaps, interpolation of short gaps.
6. **Extension to other national dairy products** (`help wanted`) -
   kumis and shubat storage and fermentation monitoring (future work).
