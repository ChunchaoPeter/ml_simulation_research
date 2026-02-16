# Testing Guide


## Quick Start

```bash
pytest tests/ -v
```

## Run Specific Test Files

```bash
# Test core data structures (State, StateSeries, validators)
pytest tests/test_base.py -v

# Test observation model
pytest tests/observation/test_linear.py -v

# Test transition model
pytest tests/transition/test_linear_gaussian_state.py -v

# Test proposal model
pytest tests/proposal/test_bootstrap.py -v

# Test resampling criterion
pytest tests/resampling/test_criterion.py -v

# Test multinomial resampler
pytest tests/resampling/test_multinomial.py -v

# Test SMC orchestrator (end-to-end)
pytest tests/test_smc.py -v

# Test factory function
pytest tests/test_models.py -v
```

## Run Tests by Directory

```bash
# All resampling tests
pytest tests/resampling/ -v

# All observation tests
pytest tests/observation/ -v

# All transition tests
pytest tests/transition/ -v

# All proposal tests
pytest tests/proposal/ -v
```

## Run a Single Test

```bash
pytest tests/test_base.py::TestStateCreation::test_state_creation -v
```
