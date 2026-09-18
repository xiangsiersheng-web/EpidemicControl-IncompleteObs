from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; o=ROOT/'results/generated';o.mkdir(parents=True,exist_ok=True)
a=pd.read_csv(ROOT/'results/v8/seq7_historical_evaluation_anchor_audit.csv')
a.to_csv(o/'supported_paper_anchor_table.csv',index=False)
print('Wrote only v8-supported decoupled ODE-DynNet anchors; no incompatible main-table rows were mixed in.')
