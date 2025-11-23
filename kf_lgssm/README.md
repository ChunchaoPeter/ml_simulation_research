# Kalman Filter - Linear Gaussian State Space Model

## Installation

### Using Conda (Recommended)

1. Create a new conda environment with Python 3.12:
```bash
conda create -n kf_lgssm python=3.12
```

2. Activate the environment:
```bash
conda activate kf_lgssm
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Using pip only

```bash
pip install -r requirements.txt
```

## Documentation

### Jupyter Notebooks

1. **[kalman_filter_implementation_demo.ipynb](kalman_filter_implementation_demo.ipynb)** - Basic implementation and usage examples
2. **[kalman_filter_analysis_optimality_stability.ipynb](kalman_filter_analysis_optimality_stability.ipynb)** - Comprehensive analysis of Kalman filter numerical stability, comparing standard vs Joseph stabilized covariance update forms

### Core Modules

- **`kf_lgssm.py`** - Kalman Filter implementation for Linear Gaussian State Space Models
- **`lgss_sample.py`** - Sampling functions for generating synthetic LGSSM data

### Examples

See the Jupyter notebooks above for detailed examples. Basic usage:

```python
from kf_lgssm import KalmanFilter
from lgss_sample import sample

# Define system parameters
F, H, Q, R = ...  # State transition, observation, and noise matrices
x0, Sigma0 = ...  # Initial state and covariance

# Create filter and run
kf = KalmanFilter(F, H, Q, R)
filtered_means, filtered_covs = kf.filter(observations, x0, Sigma0)
```
