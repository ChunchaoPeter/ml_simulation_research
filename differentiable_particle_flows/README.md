# Differentiable Particle Flows

## Overview

This is a **reusable, object-oriented SMC framework** — not a one-off script for a single model. The codebase is organized around abstract base classes and dependency injection so that each component (observation model, transition model, proposal, resampling strategy) can be developed, tested, and swapped independently. To apply the particle filter to a new state-space model, you only need to implement the relevant abstract interfaces and compose them — the core filtering logic (`SMC`) stays unchanged. See [Architecture: How to Extend for New Models](#architecture-how-to-extend-for-new-models) for details.

The framework currently ships with three resampling strategies, enabling gradient-based training of state-space models through end-to-end differentiable filtering:

- **Multinomial Resampling** - Standard CDF inversion resampling (non-differentiable)
- **Soft Resampling** - Differentiable mixture-based resampling (Karkus et al. 2018)
- **Regularised OT Resampling** - Differentiable Ensemble Transform via entropy-regularized optimal transport (Corenflos et al. 2021)

### Key Features

- Fully differentiable particle filter enabling gradient-based parameter optimization
- Stabilised Sinkhorn algorithm with epsilon-scaling for efficient OT computation
- Modular composition via dependency injection (observation, transition, proposal, resampling)
- Immutable state representation using attrs for safe `@tf.function` tracing
- Batch-vectorized operations across all components

## Installation

```bash
conda create -n dpf python=3.12
conda activate dpf
pip install -r requirements.txt
```

## Quick Start

Run the demo notebooks to see each resampling strategy in action:
- `examples/pf_demo_smc.ipynb` - Standard particle filter with multinomial resampling
- `examples/pf_demo_soft_resampling.ipynb` - Soft differentiable resampling
- `examples/pf_demo_reqularised_ot_resampling.ipynb` - Regularised OT resampling (DET)

## Project Structure

### Core Implementation

| File | Description |
|------|-------------|
| `dpf/base.py` | Core data structures: `State` (particle cloud at time t), `StateSeries` (trajectory), `Module` (base class) |
| `dpf/smc.py` | `SMC` orchestrator implementing the full particle filtering loop |
| `dpf/constants.py` | Default configuration: dtype (`float64`), ESS threshold (`0.5`), seed |

### Observation Models

| File | Description |
|------|-------------|
| `dpf/observation/base.py` | Abstract `ObservationModelBase` defining p(y_t \| x_t) interface |
| `dpf/observation/linear.py` | `LinearObservationModel`: y_t = H x_t + v_t with Gaussian noise |

### Transition Models

| File | Description |
|------|-------------|
| `dpf/transition/base.py` | Abstract `TransitionModelBase` defining p(x_t \| x_{t-1}) interface |
| `dpf/transition/linear_gaussian_state.py` | `LinearGaussianTransition`: x_t = F x_{t-1} + w_t with Gaussian noise |

### Proposal Models

| File | Description |
|------|-------------|
| `dpf/proposal/base.py` | Abstract `ProposalModelBase` defining q(x_t \| x_{t-1}, y_t) interface |
| `dpf/proposal/bootstrap.py` | `BootstrapProposal`: uses transition prior as proposal distribution |

### Resampling Strategies

| File | Description |
|------|-------------|
| `dpf/resampling/base.py` | Abstract `ResamplerBase` |
| `dpf/resampling/resampling_base.py` | `CdfInversionResamplerBase` using Template Method pattern |
| `dpf/resampling/criterion.py` | `NeffCriterion`: resample when ESS < threshold * N |
| `dpf/resampling/standard/multinomial.py` | `MultinomialResampler`: CDF inversion multinomial resampling |
| `dpf/resampling/differentiable/soft_resample.py` | `SoftResampler`: mixture-based differentiable resampling |
| `dpf/resampling/differentiable/regularised_ot_resample.py` | `RegularisedOTResampler`: DET resampling via optimal transport |

### Optimal Transport Modules

| File | Description |
|------|-------------|
| `dpf/resampling/differentiable/regularized_optimal_transport/ot_utils.py` | Cost matrices, softmin operator, particle cloud diameter |
| `dpf/resampling/differentiable/regularized_optimal_transport/sinkhorn.py` | Stabilised Sinkhorn algorithm with epsilon-scaling (Algorithm 2) |
| `dpf/resampling/differentiable/regularized_optimal_transport/plan.py` | Transport matrix computation with custom gradients (Algorithm 3) |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `examples/pf_demo_smc.ipynb` | Standard particle filter with multinomial resampling on linear Gaussian model |
| `examples/pf_demo_soft_resampling.ipynb` | Soft resampling demonstration and weight evolution analysis |
| `examples/pf_demo_reqularised_ot_resampling.ipynb` | OT-based DET resampling with transport matrix visualization |
| `linear_regression_tf_demo.ipynb` | TensorFlow Probability basics and linear regression demo |

### Documentation

- `tests/README.md` - Test suite documentation and coverage details

## Architecture: How to Extend for New Models

The framework is designed around **dependency injection** — the `SMC` orchestrator does not know which concrete models it runs. It only calls abstract interfaces. This means you can build any state-space model by implementing pluggable components and composing them into `SMC`:

```
┌─────────────────────────────────────────────────────────┐
│                     SMC Orchestrator                     │
│                                                          │
│  for t = 1, ..., T:                                      │
│    1. criterion.apply(state)  →  should we resample?     │
│    2. resampler.apply(state)  →  resample particles      │
│    3. proposal.propose(state, y_t)  →  new particles     │
│    4. observation.loglikelihood(state, y_t)  →  weights   │
│                                                          │
│  Components are swapped via constructor injection:       │
│  SMC(observation, transition, proposal, criterion,       │
│      resampler)                                          │
└─────────────────────────────────────────────────────────┘
         │              │             │            │
         ▼              ▼             ▼            ▼
  ┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Observation │ │Transition│ │ Proposal │ │Resampler │
  │  Model Base │ │Model Base│ │Model Base│ │   Base   │
  │  p(y|x)     │ │ p(x_t|   │ │ q(x_t|   │ │          │
  │             │ │  x_{t-1})│ │x_{t-1},y)│ │          │
  ├─────────────┤ ├──────────┤ ├──────────┤ ├──────────┤
  │ Linear      │ │ Linear   │ │Bootstrap │ │Multinomia│
  │ Gaussian    │ │ Gaussian │ │          │ │Soft      │
  │ (provided)  │ │(provided)│ │(provided)│ │OT (DET)  │
  ├─────────────┤ ├──────────┤ ├──────────┤ ├──────────┤
  │ Your custom │ │Your custom│ │Your custom│ │Your custom│
  │ model here  │ │model here│ │model here│ │  here    │
  └─────────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Usage

```python
from dpf.observation.linear import LinearObservationModel
from dpf.transition.linear_gaussian_state import LinearGaussianTransition
from dpf.proposal.bootstrap import BootstrapProposal
from dpf.resampling.criterion import NeffCriterion
from dpf.resampling.differentiable.regularised_ot_resample import RegularisedOTResampler
from dpf.smc import SMC
from dpf.base import State

# Create components
obs_model = LinearObservationModel(H, observation_noise)
trans_model = LinearGaussianTransition(F, transition_noise)
proposal = BootstrapProposal(trans_model)
criterion = NeffCriterion(threshold_ratio=0.5)
resampler = RegularisedOTResampler(epsilon=0.5)

# Compose into SMC
smc = SMC(obs_model, trans_model, proposal, criterion, resampler)

# Run filtering
initial_state = State(particles=initial_particles, log_weights=initial_log_weights)
final_state, trajectory = smc(initial_state, observations, return_series=True)
```

For more details, please see `pf_demo_smc.ipynb` as an example.