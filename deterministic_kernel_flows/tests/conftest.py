"""
Shared pytest fixtures and helper functions for EDH, PFPF_EDH, and PFPF_LEDH tests.
"""

import pytest
import tensorflow as tf
import sys
import os

# Add parent directory to path to import acoustic_function
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from acoustic_function import (
    initialize_acoustic_model,
    state_transition,
    observation_model,
    observation_model_general,
    compute_observation_jacobian,
    simulate_trajectory
)

from utils_pff_l96_rk4 import (
    L96_RK4,
    H_linear,
    H_linear_adjoint,
    generate_Hx_si,
    generate_observations,
    generate_L96_trajectory
)

# Enable eager execution for coverage testing
tf.config.run_functions_eagerly(True)


@pytest.fixture
def assert_allclose():
    """Assert that two tensors are element-wise equal within tolerance."""
    def _fn(actual, expected, rtol=1e-5, atol=1e-5):
        diff = tf.abs(actual - expected)
        tolerance = atol + rtol * tf.abs(expected)
        assert tf.reduce_all(diff <= tolerance), (
            f"Arrays not close enough.\n"
            f"Max difference: {tf.reduce_max(diff)}\n"
            f"Max tolerance: {tf.reduce_max(tolerance)}"
        )
    return _fn


@pytest.fixture
def assert_equal():
    """Assert that two tensors are exactly equal."""
    def _fn(actual, expected):
        assert tf.reduce_all(tf.equal(actual, expected)), (
            f"Arrays not equal.\n"
            f"Actual: {actual}\n"
            f"Expected: {expected}"
        )
    return _fn


@pytest.fixture
def acoustic_system():
    """
    Acoustic multi-target tracking system for EDH filter testing.

    Returns a dictionary with:
        - model_params: All acoustic model parameters
        - observation_jacobian: Function to compute observation Jacobian
        - observation_model_fn: Function to generate observations
        - observation_model_general_fn: Function for vectorized observations
        - state_transition_fn: Function for state propagation
    """
    # Initialize acoustic model with default parameters
    model_params = initialize_acoustic_model(
        n_targets=4,
        measurement_noise_std=0.1,
        sim_area_size=40.0
    )

    # Define observation Jacobian function
    def observation_jacobian(x, params):
        return compute_observation_jacobian(x, params)

    # Define observation model function
    def observation_model_fn(x, params, no_noise=False):
        return observation_model(x, params, no_noise=no_noise)

    # Define observation model general function
    def observation_model_general_fn(x, params, no_noise=False):
        return observation_model_general(x, params, no_noise=no_noise)

    # Define state transition function
    def state_transition_fn(x, params, use_real_noise=False, no_noise=False):
        return state_transition(x, params, use_real_noise=use_real_noise, no_noise=no_noise)

    return {
        'model_params': model_params,
        'observation_jacobian': observation_jacobian,
        'observation_model_fn': observation_model_fn,
        'observation_model_general_fn': observation_model_general_fn,
        'state_transition_fn': state_transition_fn
    }


@pytest.fixture
def acoustic_system_simple():
    """
    Simplified acoustic system with fewer targets for faster testing.

    Returns a dictionary with:
        - model_params: Acoustic model with 2 targets
        - observation_jacobian: Function to compute observation Jacobian
        - observation_model_fn: Function to generate observations
        - observation_model_general_fn: Function for vectorized observations
        - state_transition_fn: Function for state propagation
    """
    # Initialize with fewer targets for faster tests
    model_params = initialize_acoustic_model(
        n_targets=2,
        measurement_noise_std=0.1,
        sim_area_size=40.0
    )

    # Override initial state for 2 targets
    model_params['x0_initial_target_states'] = tf.expand_dims(
        tf.constant([
            12.0, 6.0, 0.001, 0.001,      # Target 1
            32.0, 32.0, -0.001, -0.005,   # Target 2
        ], dtype=tf.float32),
        axis=1
    )

    # Define observation Jacobian function
    def observation_jacobian(x, params):
        return compute_observation_jacobian(x, params)

    # Define observation model function
    def observation_model_fn(x, params, no_noise=False):
        return observation_model(x, params, no_noise=no_noise)

    # Define observation model general function
    def observation_model_general_fn(x, params, no_noise=False):
        return observation_model_general(x, params, no_noise=no_noise)

    # Define state transition function
    def state_transition_fn(x, params, use_real_noise=False, no_noise=False):
        return state_transition(x, params, use_real_noise=use_real_noise, no_noise=no_noise)

    return {
        'model_params': model_params,
        'observation_jacobian': observation_jacobian,
        'observation_model_fn': observation_model_fn,
        'observation_model_general_fn': observation_model_general_fn,
        'state_transition_fn': state_transition_fn
    }


@pytest.fixture
def ekf_for_acoustic():
    """
    Create EKF filter instance for acoustic model testing.

    This fixture provides a pre-configured EKF that can be used with
    EDH filters that have use_ekf=True.
    """
    # Import EKF from ekf_ukf_pf
    ekf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'ekf_ukf_pf'
    )
    sys.path.insert(0, ekf_path)

    try:
        from ekf import ExtendedKalmanFilter

        # Get acoustic model parameters
        model_params = initialize_acoustic_model(n_targets=4)

        # Create state transition function
        def state_transition_fn(x, u=None):
            return tf.linalg.matmul(model_params['Phi'], x)

        # Create observation function
        def observation_fn(x):
            return observation_model(x, model_params, no_noise=True)

        # Create state Jacobian function
        def state_jacobian_fn(x, u=None):
            return model_params['Phi']

        # Create observation Jacobian function
        def observation_jacobian_fn(x):
            return compute_observation_jacobian(x, model_params)

        # Initialize EKF
        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model_params['Q'],
            R=model_params['R'],
            x0=model_params['x0_initial_target_states'],
            Sigma0=model_params['P0'],
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=observation_jacobian_fn
        )

        return ekf
    except ImportError:
        pytest.skip("EKF module not available")


@pytest.fixture
def l96_system():
    """
    Lorenz 96 dynamical system for PFF testing.

    Returns a dictionary with:
        - dim: State dimension
        - dt: Time step
        - F: Forcing parameter
        - nx: Full model dimension
        - obs_interval: Observation interval
        - dim_interval: Spatial observation density
        - R: Observation error covariance
        - Q: Process noise covariance
        - model_step: Function to advance model one timestep
        - generate_Hx_si_fn: Observation operator function
        - H_linear_adjoint_fn: Adjoint of observation operator
    """
    # Model parameters
    dim = 40
    dt = 0.025
    F = 8.0
    nx = 40
    obs_interval = 4
    dim_interval = 2

    # Observation error covariance
    ny_obs = len(tf.range(3, nx, dim_interval, dtype=tf.int32))
    R = tf.eye(ny_obs, dtype=tf.float32) * 1.0

    # Process noise (for ensemble generation)
    Q = tf.eye(dim, dtype=tf.float32) * 0.1

    # Model step function
    def model_step(X):
        return L96_RK4(X, dt, F)

    # Observation operator
    def generate_Hx_si_fn(X):
        return generate_Hx_si(X, dim_interval, nx)

    # Adjoint
    def H_linear_adjoint_fn(X):
        return H_linear_adjoint(X)

    return {
        'dim': dim,
        'dt': dt,
        'F': F,
        'nx': nx,
        'obs_interval': obs_interval,
        'dim_interval': dim_interval,
        'R': R,
        'Q': Q,
        'ny_obs': ny_obs,
        'model_step': model_step,
        'generate_Hx_si_fn': generate_Hx_si_fn,
        'H_linear_adjoint_fn': H_linear_adjoint_fn
    }


@pytest.fixture
def l96_system_simple():
    """
    Simplified Lorenz 96 system with smaller dimension for faster testing.

    Returns a dictionary with the same structure as l96_system but with
    reduced dimension for faster test execution.
    """
    # Model parameters (reduced dimension)
    dim = 20
    dt = 0.025
    F = 8.0
    nx = 20
    obs_interval = 4
    dim_interval = 2

    # Observation error covariance
    ny_obs = len(tf.range(3, nx, dim_interval, dtype=tf.int32))
    R = tf.eye(ny_obs, dtype=tf.float32) * 1.0

    # Process noise
    Q = tf.eye(dim, dtype=tf.float32) * 0.1

    # Model step function
    def model_step(X):
        return L96_RK4(X, dt, F)

    # Observation operator
    def generate_Hx_si_fn(X):
        return generate_Hx_si(X, dim_interval, nx)

    # Adjoint
    def H_linear_adjoint_fn(X):
        return H_linear_adjoint(X)

    return {
        'dim': dim,
        'dt': dt,
        'F': F,
        'nx': nx,
        'obs_interval': obs_interval,
        'dim_interval': dim_interval,
        'R': R,
        'Q': Q,
        'ny_obs': ny_obs,
        'model_step': model_step,
        'generate_Hx_si_fn': generate_Hx_si_fn,
        'H_linear_adjoint_fn': H_linear_adjoint_fn
    }
