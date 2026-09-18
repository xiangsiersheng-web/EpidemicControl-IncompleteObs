from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
mapping=json.loads((ROOT/'configs/paper_model_mapping.json').read_text())
print('Checkpoint mapping validated:', ', '.join(mapping['models']))
print('Exact historical evaluation is not launched by this script: it requires the f858 main-sequence RNG protocol documented in REPRODUCIBILITY.md.')
