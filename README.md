# ML Simulation Research

State estimation and filtering algorithms for linear and non-linear state-space models using TensorFlow.

## Overview

This repository implements and analyzes simulation-based filtering methods for probabilistic state estimation:

- **Kalman Filter (KF)** - Optimal filtering for linear Gaussian state-space models
- **Extended Kalman Filter (EKF)** - Linearization-based filtering for non-linear systems
- **Unscented Kalman Filter (UKF)** - Sigma point transform for non-linear systems
- **Particle Filter (PF)** - Sequential Monte Carlo for general non-linear/non-Gaussian systems
- **TODO** - TODO deterministic kernel flows

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
```


## Documentation

- **`kf_lgssm/README.md`** - Kalman Filter documentation and usage
- **`ekf_ukf_pf/README.md`** - Non-linear filtering algorithms overview
- **`ekf_ukf_pf/range_bearing_model_documentation.md`** - Mathematical formulation

## Notebooks

### Kalman Filter
- `kalman_filter_implementation_demo.ipynb` - Basic KF implementation
- `kalman_filter_analysis_optimality_stability.ipynb` - Numerical stability analysis

### Non-linear Filters
- `ekf_demo.ipynb` - Extended Kalman Filter demonstration
- `ukf_demo.ipynb` - Unscented Kalman Filter with parameter tuning
- `pf_demo.ipynb` - Particle Filter with ESS analysis
- `compare_pf_ekf_ukf.ipynb` - Performance comparison

## License

MIT License - See LICENSE file for details
