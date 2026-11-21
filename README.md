# ML Simulation Research

TensorFlow implementations of classical Bayesian filters, deterministic particle flows, kernel flows, and differentiable Sequential Monte Carlo (SMC) methods for state estimation in linear, nonlinear, and non-Gaussian state-space models.

## What is included

| Project | Methods | Typical use |
|---|---|---|
| [`kf_lgssm/`](kf_lgssm/) | Kalman filter for linear Gaussian state-space models | Baseline filtering, controls, and covariance-stability analysis |
| [`ekf_ukf_pf/`](ekf_ukf_pf/) | Extended KF, unscented KF, and bootstrap particle filter | Nonlinear range-bearing tracking |
| [`deterministic_kernel_flows/`](deterministic_kernel_flows/) | EDH, LEDH, PFPF-EDH, PFPF-LEDH, and kernel PFF | Nonlinear tracking and high-dimensional data assimilation |
| [`differentiable_particle_flows/`](differentiable_particle_flows/) | Modular SMC with multinomial, soft, and regularised-OT resampling | Differentiable filtering and reusable model components |

The accompanying [`Particle_Flow___Differentiable_PFs_Report.pdf`](Particle_Flow___Differentiable_PFs_Report.pdf) provides the theoretical background:

- a literature review of state-estimation methods;
- a progression from classical Kalman and particle filters to deterministic particle flows; and
- stochastic particle flow and differentiable particle-filter methods.

## Repository layout

```text
ml_simulation_research/
├── kf_lgssm/                       # Linear Gaussian models and Kalman filter
├── ekf_ukf_pf/                     # Nonlinear EKF, UKF, and particle filter
├── deterministic_kernel_flows/     # EDH, LEDH, PFPF, and kernel PFF
├── differentiable_particle_flows/  # Reusable object-oriented SMC package
├── Particle_Flow___Differentiable_PFs_Report.pdf
└── LICENSE
```

Each subproject has its own dependency file, tests, notebooks, and detailed README. Run commands from the selected subproject directory so its local imports resolve correctly.

## Choosing an implementation

- Use **`kf_lgssm`** when both dynamics and observations are linear and Gaussian.
- Use **EKF** when differentiable nonlinear models and analytical Jacobians are available.
- Use **UKF** when Jacobians are inconvenient and the state dimension is moderate.
- Use **PF** for nonlinear or non-Gaussian posteriors when sampling cost is acceptable.
- Use **EDH/LEDH/PFPF** to migrate particles toward observations with deterministic flows.
- Use **PFF** for high-dimensional ensemble data assimilation with kernel localisation.
- Use **`differentiable_particle_flows`** when model parameters must be trained through the filtering computation.

## Installation

Create a separate environment for the subproject you want to run. For example:

```bash
git clone <repository-url>
cd ml_simulation_research/kf_lgssm

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Replace `kf_lgssm` with another project directory as needed. The requirement files are independently pinned, so installing every project into the same environment is not recommended.

## Run the tests

From the repository root:

```bash
(cd kf_lgssm && python -m pytest)
(cd ekf_ukf_pf && python -m pytest)
(cd deterministic_kernel_flows && python -m pytest)
(cd differentiable_particle_flows && python -m pytest)
```

To run a single test file, enter its project directory and pass the file to `pytest`, for example:

```bash
cd kf_lgssm
python -m pytest tests/test_kf_lgssm.py -v
```

## Notebook guide

### Linear Kalman filter

- [`kalman_filter_implementation_demo.ipynb`](kf_lgssm/kalman_filter_implementation_demo.ipynb) — end-to-end filtering workflow
- [`kalman_filter_analysis_optimality_stability.ipynb`](kf_lgssm/kalman_filter_analysis_optimality_stability.ipynb) — standard and Joseph covariance updates

### Nonlinear filters

- [`range_bearing_model_demo.ipynb`](ekf_ukf_pf/range_bearing_model_demo.ipynb) — nonlinear tracking model
- [`ekf_demo.ipynb`](ekf_ukf_pf/ekf_demo.ipynb) — extended Kalman filter
- [`ukf_demo.ipynb`](ekf_ukf_pf/ukf_demo.ipynb) — unscented Kalman filter
- [`pf_demo.ipynb`](ekf_ukf_pf/pf_demo.ipynb) — particle weights, resampling, and degeneracy
- [`compare_pf_ekf_ukf.ipynb`](ekf_ukf_pf/compare_pf_ekf_ukf.ipynb) — accuracy and performance comparison

### Deterministic particle and kernel flows

- [`acoustic_function_demo.ipynb`](deterministic_kernel_flows/acoustic_function_demo.ipynb) — acoustic multi-target model
- [`edh_ledh_demo.ipynb`](deterministic_kernel_flows/edh_ledh_demo.ipynb) — EDH and LEDH comparison
- [`pfpf_edh_demo.ipynb`](deterministic_kernel_flows/pfpf_edh_demo.ipynb) — PFPF-EDH with importance weights
- [`pfpf_ledh_demo_with_comparison.ipynb`](deterministic_kernel_flows/pfpf_ledh_demo_with_comparison.ipynb) — local particle flow
- [`pff_demo_100_dimension.ipynb`](deterministic_kernel_flows/pff_demo_100_dimension.ipynb) — high-dimensional Lorenz–96 assimilation
- [`pff_showcase_matric_valued_kernel.ipynb`](deterministic_kernel_flows/pff_showcase_matric_valued_kernel.ipynb) — matrix-valued kernels

### Differentiable particle filtering

- [`pf_demo_smc.ipynb`](differentiable_particle_flows/examples/pf_demo_smc.ipynb) — standard SMC
- [`pf_demo_soft_resampling.ipynb`](differentiable_particle_flows/examples/pf_demo_soft_resampling.ipynb) — soft resampling
- [`pf_demo_reqularised_ot_resampling.ipynb`](differentiable_particle_flows/examples/pf_demo_reqularised_ot_resampling.ipynb) — differentiable ensemble-transform resampling

## Tensor conventions

The classical-filter projects generally store trajectories in columns:

- states: `(state_dim, T + 1)`;
- observations: `(obs_dim, T)`; and
- controls: `(control_dim, T)`.

The differentiable SMC package is batch-first:

- particles: `[batch, particles, state_dim]`; and
- observations: `[time, batch, obs_dim]`.

Refer to each subproject README before composing modules because dtype and shape conventions differ between the classical and differentiable implementations.

## License

This project is released under the [MIT License](LICENSE).
