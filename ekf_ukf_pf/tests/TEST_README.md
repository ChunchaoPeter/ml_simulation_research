# Filter Tests

Comprehensive test suite for EKF, UKF and supporting models.

## Test Files

```
ekf_ukf_pf/
├── tests/
│   ├── test_ekf.py                    # Extended Kalman Filter tests 
│   ├── test_ukf.py                    # Unscented Kalman Filter tests 
│   ├── test_range_bearing_model.py    # Range-Bearing Model tests
│   ├── conftest.py                    # Shared fixtures
│   ├── __init__.py                    # Package init
│   └── TEST_README.md                 # This file
├── pytest.ini                         # Pytest configuration
└── run_tests.sh                       # Test runner script
```

## Test Coverage Overview

### 1. Extended Kalman Filter (`test_ekf.py`)

### 2. Unscented Kalman Filter (`test_ukf.py`)

### 3. Range-Bearing Model (`test_range_bearing_model.py`)

---

## Running Tests

### Basic Commands

```bash
# Run all tests
./run_tests.sh
# or
pytest tests/

# Run specific test file
./run_tests.sh test_ukf.py
# or
pytest tests/test_ukf.py -v

# Run specific test class
pytest tests/test_ukf.py::TestUKFUnit -v
pytest tests/test_ekf.py::TestEKFIntegration -v

# Run specific test
pytest tests/test_ukf.py::TestUKFUnit::test_initialization -v
pytest tests/test_range_bearing_model.py::TestRangeBearingModelUnit::test_observation_model_range_bearing -v
```

### Coverage Reports

```bash
# Run with coverage (all modules)
pytest tests/ --cov=. --cov-report=term-missing

# Run with coverage (specific modules)
pytest tests/test_ukf.py --cov=ukf --cov-report=term
pytest tests/test_ekf.py --cov=ekf --cov-report=term
pytest tests/test_range_bearing_model.py --cov=range_bearing_model --cov-report=term

# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html  # macOS
```
