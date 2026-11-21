# Nonlinear State Estimation with EKF, UKF, and PF

TensorFlow implementations of three filters for nonlinear state-space models:

- **Extended Kalman filter (EKF)** — user-supplied Jacobians and an optional Joseph-stabilised covariance update.
- **Unscented Kalman filter (UKF)** — sigma-point propagation with configurable `alpha`, `beta`, and `kappa`.
- **Particle filter (PF)** — bootstrap/prior proposal, multinomial resampling, effective sample size, and ancestry tracking.

The included range-bearing example tracks 2D position and velocity using constant-velocity dynamics and nonlinear polar observations.

## Range-bearing model

The state is

```text
x = [x_position, x_velocity, y_position, y_velocity]^T.
```

The transition is a linear constant-velocity model, while the observation is nonlinear:

```text
range   = sqrt(x_position^2 + y_position^2)
bearing = atan2(y_position, x_position).
```

Process and measurement noise are Gaussian, and measured bearings are wrapped to `[-pi, pi]`. The nonlinear observation makes this a useful comparison problem for EKF, UKF, and particle filtering.

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start: EKF

```python
import tensorflow as tf

from ekf import ExtendedKalmanFilter
from utils import (
    create_model,
    create_observation_fn,
    create_observation_jacobian_fn,
    create_state_jacobian_fn,
    create_state_transition_fn,
    generate_trajectory,
)

model = create_model(seed=42)
true_states, observations = generate_trajectory(model, T=50)

x0 = tf.zeros([4, 1], dtype=tf.float32)
Sigma0 = 100.0 * tf.eye(4, dtype=tf.float32)

ekf = ExtendedKalmanFilter(
    state_transition_fn=create_state_transition_fn(model),
    observation_fn=create_observation_fn(),
    Q=model.Q,
    R=model.R,
    x0=x0,
    Sigma0=Sigma0,
    state_transition_jacobian_fn=create_state_jacobian_fn(model),
    observation_jacobian_fn=create_observation_jacobian_fn(model),
)

filtered_states, predicted_states = ekf.filter(observations)
print(true_states.shape)       # (4, 51)
print(observations.shape)      # (2, 50)
print(filtered_states.shape)   # (4, 51)
print(predicted_states.shape)  # (4, 50)
```

## Creating the UKF

The UKF uses the same deterministic transition and observation callables but does not require Jacobians:

```python
from ukf import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(
    state_transition_fn=create_state_transition_fn(model),
    observation_fn=create_observation_fn(),
    Q=model.Q,
    R=model.R,
    x0=x0,
    Sigma0=Sigma0,
    alpha=1.0,
    beta=2.0,
    kappa=0.0,
)
filtered_states, predicted_states = ukf.filter(observations)
```

`alpha` controls sigma-point spread, `beta=2` is commonly used for Gaussian priors, and `kappa` is a secondary scaling parameter. The implementation validates combinations that would make sigma-point scaling invalid and warns about numerically extreme settings.

## Creating the particle filter

The particle filter needs process-noise, observation-likelihood, and initial-state samplers. The utilities construct these for the range-bearing model:

```python
from pf import ParticleFilter
from utils import create_pf_functions

pf_functions = create_pf_functions(model, x0, Sigma0)
pf = ParticleFilter(
    **pf_functions,
    num_particles=500,
    resample_method="multinomial",
)

filtered_states = pf.filter(observations)
details = pf.filter(observations, return_details=True)
```

With `return_details=True`, the returned tuple contains filtered states, predicted states, particle history, weight history, ESS history, and ancestry history.

## Filter comparison

| Property | EKF | UKF | PF |
|---|---|---|---|
| Posterior representation | Gaussian | Gaussian | Weighted particles |
| Nonlinearity handling | First-order Jacobian | Sigma points | Monte Carlo sampling |
| Needs Jacobians | Yes | No | No |
| Handles strongly non-Gaussian posteriors | Limited | Limited | Yes, given enough particles |
| Main cost driver | Jacobians and matrix solves | `2n + 1` sigma points | Number of particles |

## Shape and return conventions

| Filter | Input observations | Default output |
|---|---|---|
| EKF | `(obs_dim, T)` | `(filtered_states, predicted_states)` |
| UKF | `(obs_dim, T)` | `(filtered_states, predicted_states)` |
| PF | `(obs_dim, T)` | `filtered_states` |

For EKF and UKF, filtered states have shape `(state_dim, T + 1)` and predicted states have shape `(state_dim, T)`. The first filtered column is the initial estimate.

## Files

| File | Purpose |
|---|---|
| [`ekf.py`](ekf.py) | `ExtendedKalmanFilter` with explicit Jacobian callbacks |
| [`ukf.py`](ukf.py) | `UnscentedKalmanFilter` and sigma-point transforms |
| [`pf.py`](pf.py) | `ParticleFilter` with multinomial resampling |
| [`range_bearing_model.py`](range_bearing_model.py) | Simulator and analytical observation Jacobian |
| [`utils.py`](utils.py) | Model factories, PF callables, metrics, timing, and plots |
| [`range_bearing_model_documentation.md`](range_bearing_model_documentation.md) | Model equations and Jacobian derivation |

## Notebooks

| Notebook | Purpose |
|---|---|
| [`range_bearing_model_demo.ipynb`](range_bearing_model_demo.ipynb) | Model simulation and visualisation |
| [`ekf_demo.ipynb`](ekf_demo.ipynb) | EKF workflow and error analysis |
| [`ukf_demo.ipynb`](ukf_demo.ipynb) | UKF workflow and sigma-point tuning |
| [`pf_demo.ipynb`](pf_demo.ipynb) | PF weights, ESS, resampling, and path degeneracy |
| [`compare_pf_ekf_ukf.ipynb`](compare_pf_ekf_ukf.ipynb) | Accuracy and performance comparison |

## Evaluation utilities

`utils.py` provides position errors, RMSE, grouped metrics, timing and memory measurements, trajectory plots, component plots, PF degeneracy analysis, and particle-count sensitivity plots.

## Tests

```bash
python -m pytest
```

For a single component:

```bash
python -m pytest tests/test_ekf.py -v
python -m pytest tests/test_ukf.py -v
python -m pytest tests/test_pf.py -v
python -m pytest tests/test_range_bearing_model.py -v
```

See [`tests/TEST_README.md`](tests/TEST_README.md) for details.
