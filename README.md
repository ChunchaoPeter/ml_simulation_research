# ML Simulation Research

State estimation and filtering algorithms for linear and non-linear state-space models using TensorFlow.

## Overview

This repository implements and analyzes simulation-based filtering methods for probabilistic state estimation:

- **Kalman Filter (KF)** - Optimal filtering for linear Gaussian state-space models
- **Extended Kalman Filter (EKF)** - Linearization-based filtering for non-linear systems
- **Unscented Kalman Filter (UKF)** - Sigma point transform for non-linear systems
- **Particle Filter (PF)** - Sequential Monte Carlo for general non-linear/non-Gaussian systems
- **Deterministic Kernel Flows** - EDH, LEDH, PFPF, and PFF for high-dimensional state estimation
- **Differentiable Particle Flows** - A reusable OOP-based SMC framework with differentiable resampling via entropy-regularized optimal transport, designed to generalise to new state-space models

The report JPM_MLCOE__Particle_Flow___Differentiable_PFs__Chunchao_.pdf include:

- A literature review of potential methods for solving the problem.
- Propose your choice of methods and the reason for the choice.
- Present the answers to the question in clear and understandable language.
- Testing plans and testing results that show your implementation is correct and your results are valid. 
## Repository Structure

```
ml_simulation_research/
├── kf_lgssm/              # Kalman Filter for Linear Gaussian Systems
│   ├── kf_lgssm.py        # KF implementation
│   ├── lgss_sample.py     # Data generation
│   ├── tests/             # 20 unit and integration tests
│   └── *.ipynb            # Demo notebooks and analysis
│
├── ekf_ukf_pf/            # Non-linear Filtering Algorithms
│   ├── ekf.py             # Extended Kalman Filter
│   ├── ukf.py             # Unscented Kalman Filter
│   ├── pf.py              # Particle Filter
│   ├── range_bearing_model.py  # Non-linear state-space model
│   ├── tests/             # 40 unit and integration tests
│   └── *.ipynb            # Demo and comparison notebooks
│
├── deterministic_kernel_flows/  # Deterministic Kernel Flow Methods
│   ├── edh.py             # Exact Daum-Huang filter
│   ├── pfpf_edh.py        # Particle Flow Particle Filter (EDH)
│   ├── pfpf_ledh.py       # Localized PFPF
│   ├── pff.py             # Particle Flow Filter
│   ├── acoustic_function.py  # Multi-target tracking model
│   ├── tests/             # Comprehensive test coverage
│   └── *.ipynb            # Demo and comparison notebooks
│
├── differentiable_particle_flows/  # Differentiable Particle Filtering (OOP Framework)
│   ├── dpf/               # Core package
│   │   ├── base.py        # State, StateSeries, Module
│   │   ├── smc.py         # SMC orchestrator
│   │   ├── observation/   # Observation models p(y|x)
│   │   ├── transition/    # Transition models p(x_t|x_{t-1})
│   │   ├── proposal/      # Proposal distributions q(x|...)
│   │   └── resampling/    # Multinomial, Soft, and OT resampling
│   ├── examples/          # Demo notebooks
│   └── tests/             # Comprehensive test coverage
```

> **Note on `differentiable_particle_flows/`**: Unlike the other subprojects which are standalone implementations, `differentiable_particle_flows` is a proper **object-oriented SMC framework**. The codebase is reorganized into abstract base classes (`ObservationModelBase`, `TransitionModelBase`, `ProposalModelBase`, `ResamplerBase`) and a generic `SMC` orchestrator that composes them via dependency injection. This makes it a **reusable foundation** — to apply the particle filter to a new state-space model (e.g., nonlinear dynamics, non-Gaussian observations, neural network proposals), you only need to implement the relevant abstract interfaces and plug them in. The core filtering logic and all resampling strategies (multinomial, soft, regularised OT) work unchanged. See [`differentiable_particle_flows/README.md`](differentiable_particle_flows/README.md) for the full architecture guide and extension examples.


## Documentation

- **`kf_lgssm/README.md`** - Kalman Filter documentation and usage
- **`ekf_ukf_pf/README.md`** - Non-linear filtering algorithms overview
- **`ekf_ukf_pf/range_bearing_model_documentation.md`** - Mathematical formulation
- **`deterministic_kernel_flows/README.md`** - Deterministic kernel flow methods (EDH, LEDH, PFPF, PFF)
- **`differentiable_particle_flows/README.md`** - Differentiable particle filtering with OT resampling

## Notebooks

### Kalman Filter
- `kalman_filter_implementation_demo.ipynb` - Basic KF implementation
- `kalman_filter_analysis_optimality_stability.ipynb` - Numerical stability analysis

### Non-linear Filters
- `ekf_demo.ipynb` - Extended Kalman Filter demonstration
- `ukf_demo.ipynb` - Unscented Kalman Filter with parameter tuning
- `pf_demo.ipynb` - Particle Filter with ESS analysis
- `compare_pf_ekf_ukf.ipynb` - Performance comparison

### Deterministic Kernel Flows
- `edh_ledh_demo.ipynb` - EDH and LEDH filtering comparison
- `pfpf_edh_demo.ipynb` - PFPF-EDH with importance weighting
- `pfpf_ledh_demo_with_comparison.ipynb` - Localized PFPF performance analysis
- `pff_demo_100_dimension.ipynb` - High-dimensional PFF data assimilation
- `pff_showcase_matric_valued_kernel.ipynb` - Matrix-valued kernel demonstration
- `compare_edh_ledh_pff_with_range_model.ipynb` - Comparison with EKF/UKF/PF
- `acoustic_function_demo.ipynb` - Acoustic multi-target tracking

### Differentiable Particle Flows
- `examples/pf_demo_smc.ipynb` - Standard particle filter with multinomial resampling
- `examples/pf_demo_soft_resampling.ipynb` - Soft differentiable resampling
- `examples/pf_demo_reqularised_ot_resampling.ipynb` - Regularised OT resampling (DET)

## License

MIT License - See LICENSE file for details
