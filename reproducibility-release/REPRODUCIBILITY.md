# Reproducibility instructions

Use Python 3.9+ with NumPy and pandas. The release supports postprocessing of v8 results; it does not train RL or reconstruction models.

Legacy main-table RMSE averages 100 prediction and target curves before calculating RMSE. Trajectory-wise RMSE mean, sample SD, and t 95% CI are supplementary uncertainty measures.

For v8, `k_rec` is 3, 7, 14, or 21 and the RL observation window is always 3. Each scenario/window restores the same f858 pre-evaluation RNG snapshot after fixed artifact construction.

```bash
python reproducibility-release/scripts/smoke_test.py
python reproducibility-release/scripts/compute_legacy_ensemble_rmse.py
python reproducibility-release/scripts/compute_trajectory_statistics.py
```

Exact fresh checkpoint evaluation requires the historical main-sequence protocol. Training reproduction is not claimed because the complete data/split/RNG/early-stopping provenance has not been released as one package.
