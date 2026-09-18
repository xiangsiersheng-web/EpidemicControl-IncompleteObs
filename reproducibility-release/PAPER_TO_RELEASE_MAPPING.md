# Paper-to-release mapping

| Item | Artifacts | Status |
|---|---|---|
| v8 window ablation | `results/v8/`, postprocessing scripts | reproducible from released results |
| Seq=7 anchors | `seq7_historical_evaluation_anchor_audit.csv` | high and low verified |
| GRU | `hao_gru` / `RebuildHao` / `rebuild_hao_baseline.pth` | checkpoint protocol required |
| GCN-GRU | `gnn_gru_ordinary` / `RebuildGruGNNModel_v1` / `rebuild_gru_gnn_model_v1_ordinary.pth` | high 29.30 is historical record only; no new CI |
| Decoupled ODE-DynNet | `gnn_gru_agent` / `RebuildGruGNNModel_v1` / `rebuild_gru_gnn_model_v1_agent.pth` | v8 postprocessing reproducible |
| Joint ODE-DynNet | with-action: high actor19/predictor19, low actor18/predictor18 | paired protocol required |
