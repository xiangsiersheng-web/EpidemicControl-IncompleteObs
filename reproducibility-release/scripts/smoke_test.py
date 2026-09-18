from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
r=pd.read_csv(ROOT/'results/v8/raw_window_ablation_historical_aligned.csv')
a=pd.read_csv(ROOT/'results/v8/seq7_historical_evaluation_anchor_audit.csv')
assert len(r)==800 and not r.duplicated(['scenario','window','run_id']).any()
assert r.groupby(['scenario','window']).size().eq(100).all()
assert a.anchor_passed.all()
print('800 run-level records; 2/2 anchors passed')
