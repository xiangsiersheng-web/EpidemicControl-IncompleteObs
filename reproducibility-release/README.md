# Reproducibility release candidate

This directory is a local release candidate for *Epidemic control under incomplete observations through mechanistic state reconstruction and multi-agent learning*. It has not been published separately.

The supported workflow is postprocessing of the released v8 window-ablation records. The v8 records are the only manuscript-valid window-ablation source in this package. The older v7 protocol and its numbers are not included.

## Quick smoke test

From the repository root, run:

```bash
python reproducibility-release/scripts/smoke_test.py
```

It checks the eight scenario/window groups, 800 unique run-level records, the two seq=7 anchors, and the legacy RMSE bootstrap table. Expected output includes `800 run-level records` and `2/2 anchors passed`.

## Scope

* **Training reproduction:** not claimed. The historical training split, RNG sequence, and early-stopping provenance are documented in `docs/training_provenance_note.md`.
* **Checkpoint evaluation:** the repository already contains selected fixed weights. Exact historical main-table evaluation additionally requires the recorded f858 evaluation sequence and RNG snapshots; see `REPRODUCIBILITY.md`.
* **Table postprocessing:** supported for the v8 window ablation from files in `results/v8/`.
* **Results records:** the package contains the run-level and curve artifacts underlying v8 statistics.

Definitions: the fully observed target is `E_un + E_de + I_un + I_de + I_re`. Under partial observation, the policy uses the observation-derived proxy rather than `E_un` or `I_un`. NPO and SPO use the same proxy construction; only NPO has spatiotemporally heterogeneous self-reporting. ODE-DynNet uses a reconstruction history `k_rec=7` in the paper main table, and all RL evaluations use `w_RL=3`.

The high-transmission GCN-GRU RMSE 29.30 is a preserved historical main-table value. It has no compatible newly generated CI and this release does not attach one.

See `RELEASE_AUDIT.md`, `REPRODUCIBILITY.md`, and `PAPER_TO_RELEASE_MAPPING.md` for boundaries and file mappings.
