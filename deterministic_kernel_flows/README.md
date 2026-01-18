# Deterministic Kernel Flow Implementations

TensorFlow implementations of Exact Daum-Huang (EDH), Particle Flow Particle Filter (PFPF), and Particle Flow Filter (PFF) for nonlinear state estimation and data assimilation.

## Overview

Five filtering algorithms for high-dimensional state estimation with nonlinear dynamics and observations:

- **EDH** - Exact Daum-Huang filter with deterministic particle flow migration
- **LEDH** - Local Exact Daum-Huang (LEDH) filter with deterministic particle flow migration
- **PFPF-EDH** - Particle filtering with particle flow, importance weighting and resampling
- **PFPF-LEDH** - Localized PFPF with per-particle linearization for improved accuracy
- **PFF** - Particle flow filter for high-dimensional data assimilation with covariance localization

### Key Features

- Deterministic particle migration via flow-based methods
- Support for acoustic multi-target tracking and Lorenz 96 chaotic dynamics
- Matrix-valued and scalar kernel options for PFF
- Comprehensive test coverage (EDH, PFPF-EDH, PFPF-LEDH, PFF)

## Installation

```bash
conda create -n deterministic_kernel_flows python=3.12
conda activate deterministic_kernel_flows
pip install -r requirements.txt
```

## Quick Start

Run the demo notebooks to see each algorithm in action:
- `acoustic_function_demo.ipynb` - Acoustic sensor model and multi-target tracking
- `edh_ledh_demo.ipynb` - EDH and LEDH filtering comparison
- `pfpf_edh_demo.ipynb` - PFPF-EDH with importance weighting
- `pfpf_ledh_demo_with_comparison.ipynb` - Localized PFPF performance analysis
- `pff_demo_100_dimension.ipynb` - High-dimensional PFF data assimilation
- `pff_showcase_matric_valued_kernel.ipynb` - Matrix-valued kernel demonstration
- `compare_edh_ledh_pff_with_range_model.ipynb` - Comparison with EKF/UKF/PF on range-bearing model

## Project Structure

### Core Implementation

| File | Description |
|------|-------------|
| `edh.py` | Local Exact Daum-Huang filter with particle flow migration and optional EKF/UKF covariance tracking |
| `pfpf_edh.py` | Particle filtering with particle flow, adds importance weights and resampling to EDH |
| `pfpf_ledh.py` | Localized PFPF with per-particle linearization and Jacobian determinant weight updates |
| `pff.py` | Particle flow filter for high-dimensional systems with covariance localization |
| `acoustic_function.py` | Nonlinear acoustic multi-target tracking model with time-difference-of-arrival observations |
| `utils_pff_l96_rk4.py` | Lorenz 96 model with RK4 integration for chaotic dynamics simulation |
| `evaluation_metrics.py` | Trajectory plotting, RMSE computation, and performance visualization utilities |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `acoustic_function_demo.ipynb` | Acoustic sensor model demonstration and multi-target tracking setup |
| `edh_ledh_demo.ipynb` | EDH vs LEDH comparison showing benefits of local linearization |
| `pfpf_edh_demo.ipynb` | PFPF-EDH filtering with weight evolution and effective sample size tracking |
| `pfpf_ledh_demo_with_comparison.ipynb` | Comprehensive PFPF-LEDH analysis with performance metrics |
| `pff_demo_100_dimension.ipynb` | High-dimensional data assimilation with Lorenz 96 system |
| `pff_showcase_matric_valued_kernel.ipynb` | Matrix-valued kernel demonstration for improved accuracy |
| `compare_edh_ledh_pff_with_range_model.ipynb` | Side-by-side comparison with classical filters on range-bearing tracking |
