# Kalman Filter Implementations

TensorFlow implementations of Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filter (PF) for non-linear state estimation.

## Overview

Three filtering algorithms for range-bearing tracking with linear motion (constant velocity) and non-linear observations (polar coordinates):

- **EKF** - Linearizes dynamics using Jacobians, Joseph form covariance
- **UKF** - Sigma point transform, no Jacobians required
- **PF** - Sequential Monte Carlo with resampling

### Key Features

- TensorFlow implementation with `@tf.function` decorators for performance
- Joseph form covariance update for numerical stability (EKF)
- Comprehensive test coverage (40 tests)

## Installation

```bash
conda create -n ekf_ukf_pf python=3.10
conda activate ekf_ukf_pf
pip install -r requirements.txt
```

## Quick Start

Run the demo notebooks to see each algorithm in action:
- `ekf_demo.ipynb` - EKF filtering and performance analysis
- `ukf_demo.ipynb` - UKF with sigma point tuning
- `pf_demo.ipynb` - Particle filter with degeneracy analysis
- `compare_pf_ekf_ukf.ipynb` - Side-by-side comparison of all three algorithms

## Project Structure

### Core Implementation

| File | Description |
|------|-------------|
| `ekf.py` | Extended Kalman Filter with Jacobian-based linearization and Joseph form covariance update |
| `ukf.py` | Unscented Kalman Filter with configurable sigma point generation |
| `pf.py` | Particle Filter with systematic resampling and ESS computation |
| `range_bearing_model.py` | Non-linear state-space model with constant velocity motion and polar observations |
| `utils.py` | RMSE, visualization, and statistical analysis utilities |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `ekf_demo.ipynb` | EKF filtering on simulated trajectories with performance metrics |
| `ukf_demo.ipynb` | UKF demonstration with sigma point parameter tuning |
| `pf_demo.ipynb` | Particle filter analysis with ESS tracking and resampling behavior |
| `range_bearing_model_demo.ipynb` | Model demonstration and trajectory visualization |
| `compare_pf_ekf_ukf.ipynb` | Comprehensive performance and efficiency comparison |

### Documentation

- `range_bearing_model_documentation.md` - Mathematical formulation including state space, motion model, observation function, and Jacobian derivation
- `tests/TEST_README.md` - Test suite documentation and coverage details

## Range-Bearing Model

The state-space model consists of:
- **State**: 4D vector (x position, x velocity, y position, y velocity)
- **Motion**: Linear constant velocity model in Cartesian coordinates
- **Observation**: Non-linear transformation to polar coordinates (range, bearing)
- **Noise**: Gaussian process and measurement noise

This combination of linear motion and non-linear observations produces a non-Gaussian posterior, making it ideal for testing non-linear filtering algorithms.

## References

For detailed mathematical formulation and implementation details, see `range_bearing_model_documentation.md`.
