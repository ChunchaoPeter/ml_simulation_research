"""
Test suite for Particle Filtering with Particle Flow using Localized EDH (PFPF_LEDH) implementation.

Includes unit tests and integration tests for pfpf_ledh.py
"""

import pytest
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pfpf_ledh import PFPF_LEDH
from acoustic_function import simulate_trajectory

def create_ekf_for_ledh(model_params, system):
    """
    Create EKF instance for PFPF_LEDH testing.

    This helper function creates an EKF instance following the pattern
    from pfpf_ledh_demo.ipynb.
    """
    # Add path to EKF module
    ekf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'ekf_ukf_pf'
    )
    if ekf_path not in sys.path:
        sys.path.insert(0, ekf_path)

    try:
        from ekf import ExtendedKalmanFilter

        # Create state transition function
        def state_transition_fn(x, u=None):
            return system['state_transition_fn'](x, model_params, use_real_noise=False, no_noise=True)

        # Create observation function
        def observation_fn(x):
            return system['observation_model_fn'](x, model_params, no_noise=True)

        # Create state Jacobian function (returns Phi matrix)
        def state_jacobian_fn(x, u=None):
            return model_params['Phi']

        # Create observation Jacobian function
        def observation_jacobian_fn(x):
            return system['observation_jacobian'](x, model_params)

        # Initialize EKF
        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model_params['Q'],
            R=model_params['R'],
            x0=model_params['x0_initial_target_states'],
            Sigma0=model_params['P0'],
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            use_joseph_form=True
        )

        return ekf
    except ImportError:
        pytest.skip("EKF module not available")


class TestPFPFLEDHUnit:
    """Unit tests for individual PFPF_LEDH methods."""

    def test_initialization(self, acoustic_system_simple):
        """Test PFPF_LEDH initialization with correct dimensions."""
        system = acoustic_system_simple
        model_params = system['model_params']

        # Create EKF for LEDH
        ekf = create_ekf_for_ledh(model_params, system)

        pfpf_ledh = PFPF_LEDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            lambda_ratio=1.2,
            use_ekf=True,
            ekf_filter=ekf,
            verbose=False
        )

        # Initialize particles
        particles, m0 = pfpf_ledh.initialize(model_params)

        # Check that LEDH-specific state is initialized
        assert pfpf_ledh.log_jacobian_det_sum is not None
        assert pfpf_ledh.log_jacobian_det_sum.shape == (pfpf_ledh.n_particle,)

        # Check that use_local is set to True
        assert pfpf_ledh.use_local == True

        # Check P_all initialization
        assert pfpf_ledh.P_all is not None
        assert pfpf_ledh.P_all.shape == (pfpf_ledh.n_particle, model_params['state_dim'], model_params['state_dim'])



    def test_particle_propagation_deterministic(self, acoustic_system_simple):
        """Test deterministic particle propagation (without noise)."""
        system = acoustic_system_simple
        model_params = system['model_params']

        # Create EKF for LEDH
        ekf = create_ekf_for_ledh(model_params, system)

        pfpf_ledh = PFPF_LEDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            use_ekf=True,
            ekf_filter=ekf,
            verbose=False
        )

        particles, _ = pfpf_ledh.initialize(model_params)
        particles_det = pfpf_ledh._propagate_particles_deterministic(particles, model_params)

        # Check shape
        assert particles_det.shape == particles.shape

        # Deterministic propagation should follow Phi exactly
        Phi = model_params['Phi']
        expected = tf.matmul(Phi, particles)
        assert tf.reduce_all(tf.abs(particles_det - expected) < 1e-5)

    def test_particle_estimate(self, acoustic_system_simple):
        """Test weighted particle estimate computation."""
        system = acoustic_system_simple
        model_params = system['model_params']

        # Create EKF for LEDH
        ekf = create_ekf_for_ledh(model_params, system)

        pfpf_ledh = PFPF_LEDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=100,
            use_ekf=True,
            ekf_filter=ekf,
            verbose=False
        )

        # Create test particles
        state_dim = model_params['state_dim']
        particles = tf.random.normal((state_dim, 100), dtype=tf.float32)

        # Uniform weights should give mean
        log_weights = tf.math.log(tf.ones(100, dtype=tf.float32) / 100.0)
        estimate, weights_normalized = pfpf_ledh._particle_estimate(log_weights, particles)

        # Check that estimate is close to mean
        expected_mean = tf.reduce_mean(particles, axis=1)
        assert tf.reduce_all(tf.abs(estimate - expected_mean) < 1e-4)

        # Check that weights are normalized
        assert abs(tf.reduce_sum(weights_normalized).numpy() - 1.0) < 1e-5


class TestPFPFLEDHIntegration:
    """Integration tests for full PFPF_LEDH workflow."""

    def test_filter_with_acoustic_data(self, acoustic_system_simple):
        """Test PFPF_LEDH filter with simulated acoustic trajectory."""
        system = acoustic_system_simple
        model_params = system['model_params']

        # Generate trajectory
        T = 10
        tf.random.set_seed(42)
        true_states, observations = simulate_trajectory(model_params, T=T)

        # Create EKF for LEDH
        ekf = create_ekf_for_ledh(model_params, system)

        # Initialize filter
        pfpf_ledh = PFPF_LEDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            use_ekf=True,
            ekf_filter=ekf,
            verbose=False
        )

        # Run filter
        estimates, particles_all, covariances_all, Neff_all = pfpf_ledh.run(observations, model_params)

        # Check output shapes
        assert estimates.shape == (model_params['state_dim'], T)
        assert particles_all.shape == (model_params['state_dim'], pfpf_ledh.n_particle, T)
        assert covariances_all.shape == (model_params['state_dim'], model_params['state_dim'], T)
        assert Neff_all.shape == (T, 1)

    def test_filter_reproducibility(self, acoustic_system_simple):
        """Test that PFPF_LEDH filter produces reproducible results with same seed."""
        system = acoustic_system_simple
        model_params = system['model_params']

        T = 5
        tf.random.set_seed(123)
        true_states, observations = simulate_trajectory(model_params, T=T)

        # Create EKF for LEDH
        ekf = create_ekf_for_ledh(model_params, system)

        # Run filter twice with same seed
        pfpf_ledh1 = PFPF_LEDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            use_ekf=True,
            ekf_filter=ekf,
            verbose=False
        )

        tf.random.set_seed(456)
        estimates1, _, _, _ = pfpf_ledh1.run(observations, model_params)

        pfpf_ledh2 = PFPF_LEDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            use_ekf=True,
            ekf_filter=ekf,
            verbose=False
        )

        tf.random.set_seed(456)
        estimates2, _, _, _ = pfpf_ledh2.run(observations, model_params)

        # Results should be very similar (allowing for some numerical differences)
        assert tf.reduce_all(tf.abs(estimates1 - estimates2) < 1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
