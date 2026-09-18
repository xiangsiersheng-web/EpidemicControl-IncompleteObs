from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];r=pd.read_csv(ROOT/'results/v8/raw_window_ablation_historical_aligned.csv');out=ROOT/'results/generated';out.mkdir(parents=True,exist_ok=True)
r.groupby(['scenario','window'])[['rmse_global_run','rmse_local_day2_run','C_I','N_test','N_iso','C_Q','score']].agg(['mean','std','median']).to_csv(out/'trajectory_statistics.csv')
