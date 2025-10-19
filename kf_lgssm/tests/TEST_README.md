# Kalman Filter Tests

Simple test suite for `kf_lgssm.py` with unit tests and integration tests.

## Test Files

```
kf_lgssm/
├── tests/
│   ├── test_kf_lgssm.py    # Main test file
│   ├── conftest.py         # Shared fixtures
│   ├── __init__.py         # Package init
│   └── TEST_README.md      # This file
├── pytest.ini              # Pytest configuration
└── run_tests.sh            # Test runner script
```

## Test Coverage

### Coverage with TensorFlow's @tf.function

The Kalman Filter uses TensorFlow's `@tf.function` decorator for performance optimization. However, this creates compiled graphs that traditional coverage tools (like `coverage.py`) cannot trace properly.

**Solution:** We enable **eager execution mode** in `conftest.py`:
```python
tf.config.run_functions_eagerly(True)
```

This disables graph compilation during testing, allowing accurate coverage measurement. Note that this only affects tests - production code still uses compiled graphs for performance.

### Unit Tests (8 tests)
Tests for individual methods:
- Initialization with correct dimensions
- Initialization with 1D vector conversion
- Prediction step without control
- Prediction step with control input
- Update reduces uncertainty (covariance)
- Update with Joseph form (symmetry)
- Update with standard form (symmetry)
- Covariance positive definiteness

### Integration Tests (12 tests)
Tests for full workflows:
- Filter output shapes
- Filter with sampled data
- Filter reduces error (filtered < predicted)
- Filter with control inputs
- Filter initial state consistency
- Filter reproducibility
- Forms comparison (Joseph vs standard)
- Full workflow (sample → filter → evaluate)
- 1D system
- NaN/Inf detection
- **10D very high dimensional system (stress test)**
- **Multidimensional with varying dimensions (3D, 5D, 6D)**

**Total: 20 tests**

## Running Tests

### Basic Commands

```bash
# Install dependencies (from root directory)
pip install -r requirements.txt

# Run all tests (from root directory)
pytest tests/
# or simply
pytest

# Run with verbose output
pytest tests/ -v

# Run using the test script
./run_tests.sh

# Run specific test class
pytest tests/test_kf_lgssm.py::TestKalmanFilterUnit -v
pytest tests/test_kf_lgssm.py::TestKalmanFilterIntegration -v

# Run specific test
pytest tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_initialization -v
```

### Coverage Reports

```bash
# Run with coverage
pytest tests/ --cov=kf_lgssm

# Run with HTML coverage report
pytest tests/ --cov=kf_lgssm --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
```

### Debugging

```bash
# Stop at first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Show print statements
pytest tests/ -s
```

## Expected Output

```
======================== test session starts =========================
collected 20 items

tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_initialization PASSED
tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_initialization_with_1d_vector PASSED
tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_predict_without_control PASSED
tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_predict_with_control PASSED
tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_update_reduces_uncertainty PASSED
tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_update_joseph_form PASSED
tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_update_standard_form PASSED
tests/test_kf_lgssm.py::TestKalmanFilterUnit::test_covariance_positive_definite PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_filter_output_shapes PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_filter_with_sampled_data PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_filter_reduces_error PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_filter_with_control_input PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_filter_initial_state PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_filter_reproducibility PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_forms_comparison PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_full_workflow PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_1d_system PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_no_nan_or_inf PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_10d_very_high_dimensional_system PASSED
tests/test_kf_lgssm.py::TestKalmanFilterIntegration::test_multidimensional_with_varying_dimensions PASSED

======================== 20 passed in 2-3s ==========================
```

With coverage (using eager execution):
```
---------- coverage: platform darwin, python 3.12.11-final-0 ----------
Name            Stmts   Miss  Cover
-----------------------------------
kf_lgssm.py        52      0   100%
-----------------------------------
TOTAL              52      0   100%
```

**Note:** We achieve 100% coverage by enabling eager execution mode (`tf.config.run_functions_eagerly(True)`) in `conftest.py`. This disables TensorFlow's `@tf.function` graph compilation during testing, allowing coverage.py to trace all code paths.

## Test Structure

### Unit Tests (`TestKalmanFilterUnit`) - 8 tests
1. `test_initialization` - Verify KalmanFilter initialization with correct dimensions
2. `test_initialization_with_1d_vector` - Test 1D initial state vector converts to column vector
3. `test_predict_without_control` - Test prediction step matches mathematical formula (no control)
4. `test_predict_with_control` - Test prediction step with control input (B @ u)
5. `test_update_reduces_uncertainty` - Verify update step reduces covariance (trace)
6. `test_update_joseph_form` - Test Joseph form produces symmetric covariance
7. `test_update_standard_form` - Test standard form produces symmetric covariance
8. `test_covariance_positive_definite` - Verify covariances remain positive definite (eigenvalues > 0)

### Integration Tests (`TestKalmanFilterIntegration`) - 12 tests
1. `test_filter_output_shapes` - Verify filter produces correct output shapes
2. `test_filter_with_sampled_data` - Test filter with data from sample() function
3. `test_filter_reduces_error` - Verify filtered estimates are better than predicted
4. `test_filter_with_control_input` - Test filter with control inputs
5. `test_filter_initial_state` - Verify first filtered state matches initial state
6. `test_filter_reproducibility` - Test filter produces reproducible results
7. `test_forms_comparison` - Compare Joseph form vs standard form (should be similar)
8. `test_full_workflow` - Complete workflow: sample → filter → evaluate
9. `test_1d_system` - Edge case: 1D state space system
10. `test_no_nan_or_inf` - Test filter doesn't produce NaN or Inf values (100 timesteps)
11. `test_10d_very_high_dimensional_system` - Stress test: 10D state space, 5D observations
12. `test_multidimensional_with_varying_dimensions` - Test 3D, 5D, 6D systems with partial observability

## Fixtures

Shared fixtures in `conftest.py`:
- `simple_2d_system`: 2D constant velocity system
- `simple_1d_system`: 1D system for edge cases
- `system_with_control`: 2D system with control input
- `high_dimensional_system`: 4D system (2D position + 2D velocity)
- `very_high_dimensional_system`: 10D system (stress test)

## Requirements

use the requirements file:
```bash
pip install -r requirements.txt
```

**Note**: Tests use TensorFlow exclusively (no NumPy dependency).
