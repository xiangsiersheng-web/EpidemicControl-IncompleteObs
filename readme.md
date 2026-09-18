# Uncertain-EPC-RL: Spatiotemporal Epidemic Control Strategy Optimization under Incomplete Observation

A three-stage framework integrating observation modeling, state reconstruction, and policy optimization for epidemic control under incomplete observations.

## Overview

Real-world epidemic control is hampered by incomplete observations—limited testing, reporting delays, and asymptomatic infections—that distort situational awareness and compromise intervention decisions. This project develops a comprehensive framework that:

1. **Models** spatiotemporally heterogeneous observation gaps in epidemic dynamics
2. **Reconstructs** true infection states from distorted observations using ODE-DynNet
3. **Optimizes** multi-region coordinated control via Multi-Agent Reinforcement Learning

### Key Results

| Scenario | Reconstruction | Infection Cost | Score |
|----------|---------------|----------------|-------|
| Steady Partial Observable (baseline) | -- | 618 | 3.19 |
| Non-steady PO (no reconstruction) | ✗ | 4,930 | >10,000 |
| Non-steady PO + ODE-DynNet | ✓ | **590** | **4.64** |
| Non-steady PO + ODE-DynNet (joint) | ✓ | **488** | **4.58** |

> ODE-DynNet reduces infection cost by **88%** compared to direct policy optimization under incomplete observations.

## Framework

### Three-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Observation Modeling                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SEIQR Metapopulation Model with Observation Mechanisms │   │
│  │  • 9 compartments: S, E_un, E_de, I_un, I_de, I_re,    │   │
│  │    QE, QI, R                                           │   │
│  │  • Three observability levels                          │   │
│  │  • Mobility-driven transmission                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: State Reconstruction                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    ODE-DynNet                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │  Mechanistic │→ │   Feature    │→ │   Temporal   │  │   │
│  │  │  Diffusion   │  │ Transform    │  │ Modeling(GRU)│  │   │
│  │  │  (PureODE)   │  │              │  │              │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 3: Policy Optimization                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MAPPO with Responsibility-Aware Reward     │   │
│  │  • Centralized training, decentralized execution        │   │
│  │  • Flow-based credit assignment                         │   │
│  │  • Multi-region coordination                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Epidemic Model

### SEIQR Metapopulation Model

The extended SEIQR model distinguishes between observed and unobserved infection states with **9 compartments**:

| Compartment | Symbol | Description |
|-------------|--------|-------------|
| Susceptible | S | Individuals who can be infected |
| Exposed (Undetected) | E_un | Latent individuals not yet detected |
| Exposed (Detected) | E_de | Latent individuals detected via testing |
| Infectious (Undetected) | I_un | Infectious individuals not detected |
| Infectious (Detected) | I_de | Infectious individuals detected via testing |
| Infectious (Reported) | I_re | Self-reported symptomatic cases |
| Quarantined Exposed | QE | Isolated latent individuals |
| Quarantined Infected | QI | Isolated infectious individuals |
| Recovered | R | Recovered individuals |

### State Transition Diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │                                                     │
                    ▼                                                     │
    S ───β───> E_undetected ───σ───> I_undetected ───γ───> R             │
    │              │                      │                               │
    │          detection              detection                          │
    │              ↓                      ↓                               │
    │        E_detected ───σ───> I_detected ───γ───> R                    │
    │              │                      │                               │
    │          quarantine            quarantine                          │
    │              ↓                      ↓                               │
    │            QE ───γ_q───> R     I_reported ───γ───> R               │
    │                                     │                               │
    │                                 quarantine                         │
    │                                     ↓                               │
    └──────────────────────────────────> QI ───γ_q───> R                 │
```

### Three Observability Levels

| Level | Description | Observation Formula |
|-------|-------------|---------------------|
| **Fully Observable** | All compartments perfectly observable | I_combined = E_un + E_de + I_un + I_de + I_re |
| **Steady Partial Observable (SPO)** | Observation-derived proxy, constant reporting rate | Policy receives a proxy; it does not receive E_un or I_un |
| **Non-steady Partial Observable (NPO)** | Observation-derived proxy, heterogeneous self-reporting | Same proxy construction as SPO; only P_re is spatiotemporally heterogeneous |

For partial observation, the policy input is an observation-derived proxy: known burden plus a sensitivity correction formed from detected increments `D_E`, `D_I`, and effective detection sensitivities. It is not simply `E_de + I_de + I_re`. NPO and SPO use this same proxy form; only `P_re,i(t)` is spatiotemporally heterogeneous in NPO.

## State Reconstruction: ODE-DynNet

ODE-DynNet is a hybrid model that integrates **mechanistic ODE constraints** with **data-driven graph-temporal networks**:

### Architecture Components

1. **Mechanistic Diffusion Module**
   - Encodes spatial propagation via mobility network
   - PureODE-based diffusion process
   ```
   h₁ = N · M · (M^T · Obs_t) / (M^T · N)
   h₂ = (∑Obs_t[¬mask] / ∑h₁[¬mask]) · h₁
   ```

2. **Feature Transformation**
   - Learnable linear layer for latent representation
   - Adapts mechanistic prior to data context

3. **Temporal Modeling (GRU)**
   - Processes historical feature sequence
   - Captures epidemic state evolution

The paper main-table RMSE uses the historical legacy ensemble definition. The aligned v8 window-ablation records and statistics are in `reproducibility-release/results/v8/`; the fixed reconstruction history for the main-table anchor is 7 days, while the RL observation window is always 3. The high GCN-GRU RMSE 29.30 is a preserved historical value and has no compatible new confidence interval.

## Multi-Agent Reinforcement Learning

### MAPPO Framework

- **Centralized Training**: Global critic accesses all agent information
- **Decentralized Execution**: Each agent acts on local observations
- **Parameter Sharing**: Improves sample efficiency and stability

### Action Space

Each region selects two intervention levels:
- **Testing intensity** (L_test): Controls detection rate
- **Isolation intensity** (L_iso): Controls quarantine rate

### Responsibility-Aware Reward

The infection cost incorporates cross-regional transmission responsibility:

```
C_EI^resp = (1/N_i) Σ_j [ (m_ij · I_eff,i / Σ_l m_lj · I_eff,l) 
                          × (Σ_l m_lj · S_l) 
                          × (β · Σ_l m_lj · I_eff,l / Σ_l m_lj · N_l) ]
```

This mechanism:
- Allocates responsibility based on mobility contributions
- Promotes cooperative behavior across regions
- Satisfies conservation property: Σ C_EI^resp · N_i = Σ λ_i · S_i

### Reward Mechanism Comparison

| Reward Type | C_I | N_test | C_Q | Score |
|-------------|-----|--------|-----|-------|
| Local-only (SPO) | 29,109 | 847 | 133,641 | >10,000 |
| **Responsibility-aware (SPO)** | **618** | **155** | **19,970** | **3.19** |

## Project Structure

```
UNCERTAIN_EPC_RL/
├── actor_critic_models/          # Neural network architectures
│   ├── fully_connected.py        # MLP actor-critic
│   ├── gru_model.py              # GRU-based models
│   └── lstm_model.py             # LSTM-based models
├── algorithm/                    # RL algorithms
│   └── ppo_discrete_gpu.py       # PPO with discrete action space
├── environment/                  # Epidemic simulation
│   └── uncertain_seir_vector_v4.py  # SEIQR metapopulation model
├── uncertainty/                  # Uncertainty handling modules
│   └── obs_imperfect/            # Incomplete observation handling
│       ├── evaluate_info_rebuild.py   # Reconstruction evaluation
│       ├── gru_gnn_model_v1.py        # ODE-DynNet (GNN-GRU)
│       ├── gru_gnn_model_v2.py        # GNN-GRU variant
│       ├── generic_predictor.py       # Unified predictor interface
│       └── main_rebuild.py            # Reconstruction main script
├── draw/                         # Visualization & analysis
│   ├── partial_observability_analysis/
│   ├── natural_transmission_observation_levels/
│   ├── map_comparison/
│   ├── detection_efficiency_analysis/
│   ├── ablation_responsibility_distribution/
│   └── missing_rate_analysis/
├── data/                         # Dataset files
│   └── sz/                       # Shenzhen city data
│       ├── community_654/        # Filtered 654 communities
│       └── Shenzhen_geo_data/    # Geographic data
├── model/                        # Trained checkpoints
├── utils/                        # Utility functions
├── config.py                     # Hyperparameters
├── train_gpu.py                  # Main training script
└── readme.md
```

## Quick Start

### Requirements

```bash
pip install torch numpy pandas matplotlib geopandas scikit-learn scipy
```

### Reproducibility entry points

```bash
python reproducibility-release/scripts/smoke_test.py
python reproducibility-release/scripts/run_checkpoint_evaluation.py --runs 2 --seed 3047 --device cpu
```

These commands evaluate fixed published artifacts or postprocess released results; they do not train. See `reproducibility-release/REPRODUCIBILITY.md` for checkpoint paths, output files, and the training-reproduction boundary.

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_train_steps` | 1.2e6 | Maximum training steps |
| `batch_size` | 2,400 | Batch size for PPO |
| `lr_a` / `lr_c` | 3e-4 | Learning rate (actor/critic) |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 0.2 | PPO clip parameter |
| `hidden_width` | 256 | Hidden layer size |
| `R0` | high | Transmission scenario (high: R₀≈5.5, low: R₀≈2.6) |
| `ODE_period` | 120 | Simulation period (days) |
| `detection_efficiency_exp_param` | 0.6 | Detection efficiency exponent η |

## Experimental Results

### Study Area

Experiments use **Shenzhen, China** as a case study:
- Population: ~17 million residents
- Communities: 654 (filtered from 673)
- Data: Real-world mobility network from mobile phone signaling

### Reconstruction Accuracy

| Transmission | Method | RMSE_global | C_I | Score |
|--------------|--------|-------------|-----|-------|
| High (R₀≈5.5) | ODE-DynNet (decoupled) | 13.05 | 590 | 4.64 |
| High (R₀≈5.5) | ODE-DynNet (joint) | **11.83** | **488** | **4.58** |
| Low (R₀≈2.6) | ODE-DynNet (decoupled) | 20.04 | 381 | 4.15 |
| Low (R₀≈2.6) | ODE-DynNet (joint) | **13.21** | **413** | **3.28** |

### Impact of Observation Missing Rate

Control performance degradation as missing rate increases:

| Missing Rate | No Reconstruction | ODE-DynNet |
|--------------|-------------------|------------|
| Low (~0.03) | Score >10,000 | Score ~4.6 |
| Medium (~0.2) | Score >10,000 | Score ~5-7 |
| High (>0.4) | Score >10,000 | Score ~8-15 |

## Key Findings

1. **Incomplete observations severely degrade control**: Without reconstruction, infection and control costs increase ~8× and ~7× respectively under NPO.

2. **ODE-DynNet achieves superior reconstruction**: Hybrid approach outperforms both purely mechanistic (PureODE) and purely data-driven (GCN-GRU) methods.

3. **Joint training enhances performance**: End-to-end optimization aligns reconstruction with policy objectives.

4. **Responsibility-aware reward enables coordination**: 97.9% reduction in infection cost compared to local-only rewards.

5. **Framework generalizes across scenarios**: Consistent performance across different transmission intensities and missing rates.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2025spatiotemporal,
  title={Spatiotemporal Epidemic Control Strategy Optimization under Incomplete Observation},
  author={Wang, Shang and Yin, Ling and Luo, Yuxiao and Cui, Yunduan},
  journal={},
  year={}
}
```

## License

This project is for academic research purposes only.

## Acknowledgments

This work uses real-world mobility data from Shenzhen, China for epidemic simulation and validation.
