# Kalman Filter Implementations

TensorFlow implementations of Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filter (PF) for non-linear state estimation.

## Overview

Three filtering algorithms for range-bearing tracking with linear motion (constant velocity) and non-linear observations (polar coordinates):

- **EKF** (`ekf.py`) - Linearizes dynamics using Jacobians, Joseph form covariance
- **UKF** (`ukf.py`) - Sigma point transform, no Jacobians required
- **PF** (`pf.py`) - Sequential Monte Carlo with resampling

## Installation

1. Create a new conda environment with Python 3.10:
```bash
conda create -n ekf_ukf_pf python=3.10
```

2. Activate the environment:
```bash
conda activate ekf_ukf_pf
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Demos

- `ekf_demo.ipynb` - EKF range-bearing tracking
- `ukf_demo.ipynb` - UKF range-bearing tracking
- `pf_demo.ipynb` - PF with degeneracy analysis

## Testing

```bash
pytest tests/ -v
```

40 tests covering EKF (10), UKF (11), PF (9), and range-bearing model (10). See `TEST_README.md` for details.

## Features

- `@tf.function` decorators for performance
- Joseph form covariance update (EKF)
- Comprehensive test coverage (40 tests)