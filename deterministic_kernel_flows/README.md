# Deterministic Particle and Kernel Flows

TensorFlow implementations of deterministic particle-flow and kernel-flow methods for nonlinear state estimation and high-dimensional data assimilation.

## Implemented methods

- **EDH** — Exact Daum–Huang flow using a shared observation linearisation.
- **LEDH** — local Exact Daum–Huang flow using particle-specific linearisations.
- **PFPF-EDH** — EDH used as an invertible particle-filter proposal with importance weighting and resampling.
- **PFPF-LEDH** — local-flow PFPF with particle-specific flow parameters and Jacobian-determinant corrections.
- **PFF** — kernel particle-flow filter with scalar or matrix-valued kernels and covariance localisation.

The examples cover nonlinear acoustic multi-target tracking and Lorenz–96 data assimilation.

## Why particle flow?

A conventional particle filter proposes particles from a transition distribution and then corrects them using importance weights. In informative or high-dimensional observation models, most weights can collapse toward zero. Particle-flow methods instead move particles continuously from a prior distribution toward an observation-informed posterior.

EDH and LEDH use Daum–Huang flow equations. PFPF variants retain importance weights so that the flow acts as a proposal inside a particle filter. The kernel PFF follows a different approach, moving an ensemble along a gradient flow constructed from scalar or matrix-valued kernels.

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some comparison notebooks import modules from `ekf_ukf_pf`. Run those notebooks from the repository checkout so the sibling project is available.

## Where to start

The notebooks contain the complete model-specific setup required by these algorithms:

| Notebook | Purpose |
|---|---|
| [`acoustic_function_demo.ipynb`](acoustic_function_demo.ipynb) | Acoustic observation model and multi-target simulation |
| [`edh_ledh_demo.ipynb`](edh_ledh_demo.ipynb) | EDH versus LEDH |
| [`pfpf_edh_demo.ipynb`](pfpf_edh_demo.ipynb) | PFPF-EDH, importance weights, and effective sample size |
| [`pfpf_ledh_demo_with_comparison.ipynb`](pfpf_ledh_demo_with_comparison.ipynb) | PFPF-LEDH and local linearisation |
| [`pff_demo_100_dimension.ipynb`](pff_demo_100_dimension.ipynb) | High-dimensional Lorenz–96 assimilation |
| [`pff_showcase_matric_valued_kernel.ipynb`](pff_showcase_matric_valued_kernel.ipynb) | Scalar and matrix-valued kernel behaviour |
| [`compare_edh_ledh_pff_with_range_model.ipynb`](compare_edh_ledh_pff_with_range_model.ipynb) | Flow methods compared with EKF, UKF, and PF |

For a first particle-flow run, open `edh_ledh_demo.ipynb`. For the high-dimensional kernel method, start with `pff_demo_100_dimension.ipynb`.

## Core modules

| File | Public entry point | Purpose |
|---|---|---|
| [`edh.py`](edh.py) | `EDHFilter` | Global or optional local Daum–Huang flow and optional covariance tracking |
| [`pfpf_edh.py`](pfpf_edh.py) | `PFPF_EDH` | Invertible EDH proposal with importance weights and resampling |
| [`pfpf_ledh.py`](pfpf_ledh.py) | `PFPF_LEDH` | Per-particle flow parameters and Jacobian corrections |
| [`pff.py`](pff.py) | `ParticleFlowFilter` | Kernel flow for sequential data assimilation |
| [`acoustic_function.py`](acoustic_function.py) | Model helper functions | Acoustic multi-target model, observation Jacobian, and simulator |
| [`utils_pff_l96_rk4.py`](utils_pff_l96_rk4.py) | Lorenz–96 utilities | RK4 dynamics, observation operators, kernels, and initial ensembles |
| [`evaluation_metrics.py`](evaluation_metrics.py) | Evaluation helpers | Trajectory plots, OMAT, and position-error metrics |

## EDH and PFPF inputs

`EDHFilter` requires:

- an observation function;
- its Jacobian;
- the number of particles and pseudo-time steps; and
- a model-parameter dictionary supplied to `initialize()` and filtering methods.

`PFPF_EDH` and `PFPF_LEDH` additionally require transition and vectorised observation functions so proposal densities and importance weights can be evaluated. The acoustic notebooks demonstrate the required parameter dictionary and tensor shapes.

Important configuration values include:

- `n_particle`: ensemble size;
- `n_lambda`: number of pseudo-time flow steps;
- `lambda_ratio`: spacing of flow steps;
- `use_local`: global EDH versus local linearisation where supported;
- `use_ekf`: use a supplied EKF-style object for covariance tracking; and
- `redraw`: optionally redraw the ensemble from a Gaussian approximation.

## Kernel PFF configuration

`ParticleFlowFilter` is configured with the state dimension, ensemble size, observation schedule, covariance `R`, and observation/adjoint operators. Its main numerical parameters include:

- `kernel_type="scalar"` or `kernel_type="matrix"`;
- `alpha`: kernel bandwidth parameter;
- `max_pseudo_step`: maximum gradient-flow iterations per assimilation cycle;
- `eps_init`: initial learning rate;
- `stop_cri_percentage`: relative stopping criterion;
- `inflation_fac`: prior-covariance inflation; and
- `r_influ`: localisation radius.

Matrix-valued kernels can encode cross-component structure, while scalar kernels are simpler and less expensive.

## Model examples

### Acoustic multi-target tracking

The acoustic utilities build a nonlinear time-difference-of-arrival observation model. A joint state contains the target states, while sensor measurements create a nonlinear and potentially challenging likelihood.

### Lorenz–96

The PFF utilities use RK4 integration for the chaotic Lorenz–96 system. Observation operators can select only part of the state, allowing experiments with sparse observations and covariance localisation.

## Evaluation

`evaluation_metrics.py` includes trajectory visualisation, optimal mass transfer/assignment-style metrics, per-time-step errors, and acoustic position-error helpers. The notebooks combine these metrics with effective sample size and runtime comparisons.

## Tests

```bash
python -m pytest
```

For an individual algorithm:

```bash
python -m pytest tests/test_edh.py -v
python -m pytest tests/test_pfpf_edh.py -v
python -m pytest tests/test_pfpf_ledh.py -v
python -m pytest tests/test_pff.py -v
```

See [`tests/TEST_README.md`](tests/TEST_README.md) for test-suite details.

## References

- Daum and Huang, exact particle-flow methods for nonlinear filtering.
- Ding and Coates (2012), implementation of the Daum–Huang exact-flow particle filter.
- Li and Coates (2017), particle filtering with invertible particle flow.
- Hu and van Leeuwen (2021), particle flow filtering for high-dimensional data assimilation.
