from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; r=pd.read_csv(ROOT/'results/v8/raw_window_ablation_historical_aligned.csv')
o=ROOT/'results/generated';o.mkdir(parents=True,exist_ok=True)
r.groupby(['scenario','window']).size().reset_index(name='n_runs').to_csv(o/'window_ablation_counts.csv',index=False)
pd.read_csv(ROOT/'results/v8/legacy_ensemble_rmse_bootstrap_historical_aligned.csv').to_csv(o/'window_ablation_legacy_bootstrap.csv',index=False)
pd.read_csv(ROOT/'results/v8/paired_window_difference_statistics.csv').to_csv(o/'window_ablation_paired_statistics.csv',index=False)
