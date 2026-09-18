from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; rows=[]
for p in (ROOT/'results/v8/evaluations').glob('*/*_curves.npz'):
 d=np.load(p); x=d['prediction_runs'].mean(0)-d['target_runs'].mean(0); rows.append({'scenario':p.parent.name,'window':int(p.stem.split('_')[0][3:]),'legacy_ensemble_rmse':float(np.sqrt(np.mean(x*x)))})
out=ROOT/'results/generated';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).sort_values(['scenario','window']).to_csv(out/'legacy_ensemble_rmse.csv',index=False)
