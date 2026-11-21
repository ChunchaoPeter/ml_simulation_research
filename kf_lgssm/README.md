# Kalman Filter for Linear Gaussian State-Space Models

This project implements a TensorFlow Kalman filter and a matching simulator for the model

```text
x_t = F x_{t-1} + B u_t + w_t,   w_t ~ N(0, Q)
z_t = H x_t + v_t,               v_t ~ N(0, R)
```

The Kalman filter is the exact Bayesian filter for this linear Gaussian model. The implementation supports optional control inputs and both standard and Joseph-stabilised covariance updates.

## Features

- TensorFlow implementation with graph-compatible filtering loops.
- One-dimensional or column-vector initial means.
- Optional control matrix `B` and control sequence.
- Synthetic LGSSM state and observation generation.
- Joseph covariance update for improved numerical symmetry and stability.
- Unit and integration tests for prediction, updates, controls, reproducibility, and filtering accuracy.

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

```python
import tensorflow as tf

from kf_lgssm import KalmanFilter
from lgss_sample import sample

# Constant-velocity state: [position, velocity].
F = tf.constant([[1.0, 1.0], [0.0, 1.0]])
H = tf.constant([[1.0, 0.0]])
Q = 0.01 * tf.eye(2)
R = tf.constant([[0.1]])
x0 = tf.constant([0.0, 1.0])
Sigma0 = tf.eye(2)

true_states, observations = sample(
    F=F,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    Sigma0=Sigma0,
    T=50,
    seed=42,
)

kf = KalmanFilter(
    F=F,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    Sigma0=Sigma0,
    use_joseph_form=True,
)
filtered_states, predicted_states = kf.filter(observations)

print(true_states.shape)       # (2, 51)
print(observations.shape)      # (1, 50)
print(filtered_states.shape)   # (2, 51)
print(predicted_states.shape)  # (2, 50)
```

`filter()` returns filtered and predicted state estimates. It does not return covariance histories. Call `predict()` and `update()` directly if an application needs each step's covariance.

## Control inputs

Pass `B` to both the simulator and filter, and provide one control vector per time step:

```python
B = tf.constant([[0.5], [1.0]])
controls = tf.ones([1, 50])

states, observations = sample(
    F, H, Q, R, x0, Sigma0,
    T=50,
    B=B,
    controls=controls,
    seed=42,
)

kf = KalmanFilter(F, H, Q, R, x0, Sigma0, B=B)
filtered_states, predicted_states = kf.filter(
    observations, controls=controls
)
```

## Public API

### `KalmanFilter(...)`

Required arguments are `F`, `H`, `Q`, `R`, `x0`, and `Sigma0`. Optional arguments are:

- `B`: control-input matrix;
- `use_joseph_form=True`: select the Joseph covariance update.

### `predict(x, Sigma_post, u=None)`

Returns the predicted state and covariance before incorporating the current observation.

### `update(z, x_pred, Sigma_pred)`

Returns the posterior state and covariance after the measurement update.

### `filter(observations, controls=None)`

Runs prediction and update steps over the complete observation sequence and returns `(filtered_states, predicted_states)`.

### `sample(...)`

Samples `x_0` from `N(x0, Sigma0)`, then generates `T` transitions and observations. Supplying a seed makes repeated calls reproducible within the supported TensorFlow random-number behaviour.

## Shape conventions

| Value | Shape |
|---|---|
| Initial state `x0` | `(state_dim,)` or `(state_dim, 1)` |
| Transition matrix `F` | `(state_dim, state_dim)` |
| Observation matrix `H` | `(obs_dim, state_dim)` |
| Process covariance `Q` | `(state_dim, state_dim)` |
| Observation covariance `R` | `(obs_dim, obs_dim)` |
| Control matrix `B` | `(state_dim, control_dim)` |
| States returned by `sample()` | `(state_dim, T + 1)` |
| Observations | `(obs_dim, T)` |
| Controls | `(control_dim, T)` |
| Filtered states | `(state_dim, T + 1)` |
| Predicted states | `(state_dim, T)` |

## Covariance update choices

The standard update is

```text
Sigma_post = (I - K H) Sigma_pred.
```

The default Joseph form is

```text
Sigma_post = (I - K H) Sigma_pred (I - K H)^T + K R K^T.
```

The Joseph form requires more matrix operations but better preserves symmetry and positive-semidefinite structure under floating-point error.

## Files

| File | Purpose |
|---|---|
| [`kf_lgssm.py`](kf_lgssm.py) | `KalmanFilter`, including prediction, update, and sequence filtering |
| [`lgss_sample.py`](lgss_sample.py) | Synthetic LGSSM state and observation generation |
| [`kf.md`](kf.md) | Mathematical notes and derivation |
| [`kalman_filter_implementation_demo.ipynb`](kalman_filter_implementation_demo.ipynb) | End-to-end example |
| [`kalman_filter_analysis_optimality_stability.ipynb`](kalman_filter_analysis_optimality_stability.ipynb) | Numerical-stability analysis |

## Tests

```bash
python -m pytest
```

For verbose output or a single test:

```bash
python -m pytest -v
python -m pytest tests/test_kf_lgssm.py -v
```

See [`tests/TEST_README.md`](tests/TEST_README.md) for the test-suite guide.
