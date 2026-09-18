# Release audit

## Reviewed sources

| Source | Status | Notes |
|---|---|---|
| Public repository root | reviewed | Existing environment, baseline code, public data, and 55 checkpoint files are present. |
| Private research repository | reviewed selectively | Used only as a whitelisted source for final v8 result artifacts and portable postprocessing logic. |
| v8 historical-evaluation-aligned output | reviewed | Anchor audit, raw records, summaries, bootstrap, paired statistics, evaluation curves, and RNG manifest are present. |
| Final manuscript LaTeX source | not available in this workspace | Paper values used here are those retained in the verified v8 anchor audit. |

No release candidate file depends on a private absolute path.

## Reproducibility coverage

| Paper item | Scenarios | Required entry/artifacts | Public support | Release status | Limitation |
|---|---|---|---|---|---|
| SPO | high, low | environment, actor, state norm, protocol | partial | evaluation_reproducible_with_checkpoint | Historical RNG/main-sequence wrapper is not yet a standalone public evaluator. |
| NPO no reconstruction | high, low | environment, actor, state norm | partial | evaluation_reproducible_with_checkpoint | Same protocol boundary. |
| Pure ODE | high, low | environment, ODE baseline, actor | partial | evaluation_reproducible_with_checkpoint | Same protocol boundary. |
| IDW | high, low | environment, IDW baseline, actor | partial | evaluation_reproducible_with_checkpoint | Same protocol boundary. |
| GRU/Hao | high, low | `hao_gru`, `RebuildHao`, fixed checkpoint | partial | evaluation_reproducible_with_checkpoint | Historical checkpoint/protocol manifest still required. |
| GCN-GRU ordinary | high, low | `gnn_gru_ordinary`, `RebuildGruGNNModel_v1` | partial | results_record_only | High RMSE 29.30 is a preserved historical exception; no compatible CI. |
| Decoupled ODE-DynNet | high, low | `gnn_gru_agent`, fixed checkpoint, f858 RNG snapshot | v8 verified | postprocessing_reproducible_from_released_results | v8 provides exact seq=7 anchor and aligned window records, not a claim of fresh training reproduction. |
| Joint NPO no reconstruction | high, low | joint actor/state norm, sequence protocol | partial | evaluation_reproducible_with_checkpoint | Exact historical ordering must be retained. |
| Joint ODE-DynNet with-action | high, low | high actor19/predictor19; low actor18/predictor18 | partial | evaluation_reproducible_with_checkpoint | Requires historical paired checkpoint/protocol selection. |
| v8 window ablation | high, low | v8 raw curves/results, postprocessing scripts | verified | postprocessing_reproducible_from_released_results | v8 only; v7 is superseded and excluded. |

## Training provenance boundary

The private audit recovered deterministic seq=7 checkpoint equivalence under the historical training sequence, but a public training-reproduction claim requires publishing the complete trajectory data, split artifacts, RNG sequence, and early-stopping record as a cohesive protocol. This candidate therefore does not provide a nominal training script.

## Data, model, and security findings

* The public repository currently has no identifiable `LICENSE`/`COPYING` file. Redistribution authorization has been confirmed by the repository owner, but a formal license choice is still required before a public release claim.
* Existing public data include mobility, population, and geographic files. Their original third-party attribution and redistribution terms need to be added before a release tag.
* Existing public checkpoint inventory: 55 `.pth` files, approximately 41.5 MB. This candidate does not duplicate them.
* Whitelisted v8 files were scanned before inclusion. No token, private key, `.env`, or private absolute path is intentionally included.

## Minimal additions selected

`results/v8/` receives only final aligned records, curves, bootstrap, paired statistics, and anchor audit. `src/` and `scripts/` contain portable postprocessing only. Checkpoint/data pointers are described rather than duplicating unrelated files.
