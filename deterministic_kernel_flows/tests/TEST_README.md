# Deterministic Kernel Flows Tests

Comprehensive test suite for EDH, PFPF-EDH, PFPF-LEDH and supporting models.

## Test Files

```
deterministic_kernel_flows/
├── tests/
│   ├── test_edh.py                  # Ensemble Diffusion H-infinity Filter tests
│   ├── test_pff.py                  # Particle Flow Filter tests
│   ├── test_pfpf_edh.py            # PFPF-EDH Filter tests
│   ├── test_pfpf_ledh.py           # PFPF-LEDH Filter tests
│   ├── conftest.py                  # Shared fixtures
│   ├── __init__.py                  # Package init
│   └── TEST_README.md               # This file
```

## Test Coverage Overview

### 1. Ensemble Diffusion H-infinity Filter (`test_edh.py`)

Tests for the EDH filter implementation with acoustic multi-target tracking model.

### 2. Particle Flow Filter (`test_pff.py`)

Tests for the PFF implementation with Lorenz 96 dynamical system, including:
- Matrix-valued and scalar kernel implementations
- Prior covariance computation and localization
- Gradient computations and adaptive pseudo-time stepping
- Full data assimilation cycles

### 3. PFPF-EDH Filter (`test_pfpf_edh.py`)

Tests for the Particle Flow Particle Filter with EDH integration.

### 4. PFPF-LEDH Filter (`test_pfpf_ledh.py`)

Tests for the Particle Flow Particle Filter with Localized EDH integration.

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_edh.py -v
pytest tests/test_pff.py -v
pytest tests/test_pfpf_edh.py -v
pytest tests/test_pfpf_ledh.py -v

# Run specific test class
pytest tests/test_edh.py::TestEDHUnit -v
pytest tests/test_pff.py::TestPFFUnit -v
pytest tests/test_pfpf_edh.py::TestPFPFEDHUnit -v

# Run specific test
pytest tests/test_edh.py::TestEDHUnit::test_initialization -v
pytest tests/test_pff.py::TestPFFUnit::test_initialization -v
```

