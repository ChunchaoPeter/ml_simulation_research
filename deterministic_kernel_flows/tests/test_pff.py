"""
Test suite for Particle Flow Filter (PFF) implementation.

Includes unit tests and integration tests for pff.py
"""

import pytest
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pff import ParticleFlowFilter
from utils_pff_l96_rk4 import L96_RK4, generate_L96_trajectory, generate_observations


class TestPFFUnit:
    """Unit tests for individual PFF methods."""

    def test_initialization(self, l96_system_simple):
        """Test PFF initialization with correct dimensions."""
        system = l96_system_simple

        pff = ParticleFlowFilter(
            dim=system['dim'],
            np_particles=30,
            nt=20,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=20 // system['obs_interval'],
            nx=system['nx'],
            R=system['R'],
            alpha=1.0 / 30,
            max_pseudo_step=50,
            eps_init=5e-2,
            kernel_type='matrix'
        )

        assert pff.dim == system['dim']
        assert pff.np_particles == 30
        assert pff.nt == 20
        assert pff.obs_interval == system['obs_interval']
        assert pff.dim_interval == system['dim_interval']
        assert pff.nx == system['nx']
        assert pff.kernel_type == 'matrix'
        assert pff.alpha == 1.0 / 30

    def test_initialization_scalar_kernel(self, l96_system_simple):
        """Test PFF initialization with scalar kernel."""
        system = l96_system_simple

        pff = ParticleFlowFilter(
            dim=system['dim'],
            np_particles=30,
            nt=20,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=20 // system['obs_interval'],
            nx=system['nx'],
            R=system['R'],
            kernel_type='scalar'
        )

        assert pff.kernel_type == 'scalar'

    def test_invalid_kernel_type(self, l96_system_simple):
        """Test that invalid kernel type raises error."""
        system = l96_system_simple

        with pytest.raises(ValueError, match="kernel_type must be 'scalar' or 'matrix'"):
            ParticleFlowFilter(
                dim=system['dim'],
                np_particles=30,
                nt=20,
                obs_interval=system['obs_interval'],
                dim_interval=system['dim_interval'],
                total_obs=5,
                nx=system['nx'],
                R=system['R'],
                kernel_type='invalid'
            )

    def test_prior_covariance_computation(self, l96_system_simple):
        """Test prior covariance computation from ensemble."""
        system = l96_system_simple

        pff = ParticleFlowFilter(
            dim=system['dim'],
            np_particles=50,
            nt=20,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=5,
            nx=system['nx'],
            R=system['R']
        )

        # Create test particles
        tf.random.set_seed(42)
        particles = tf.random.normal((system['dim'], 50), dtype=tf.float32)

        B, mean = pff.compute_prior_covariance(particles)

        # Check shapes
        assert B.shape == (system['dim'], system['dim'])
        assert mean.shape == (system['dim'], 1)

        # Check positive definiteness
        eigenvalues = tf.linalg.eigvalsh(B)
        assert tf.reduce_all(eigenvalues > 0)

        # Check symmetry
        assert tf.reduce_all(tf.abs(B - tf.transpose(B)) < 1e-5)

    def test_localization_mask_construction(self, l96_system_simple):
        """Test localization mask construction."""
        system = l96_system_simple

        pff = ParticleFlowFilter(
            dim=system['dim'],
            np_particles=30,
            nt=20,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=5,
            nx=system['nx'],
            R=system['R'],
            r_influ=4
        )

        mask = pff.build_localization_mask()

        # Check shape
        assert mask.shape == (system['dim'], system['dim'])

        # Check diagonal is all ones
        diagonal = tf.linalg.diag_part(mask)
        assert tf.reduce_all(tf.abs(diagonal - 1.0) < 1e-6)

        # Check symmetry
        assert tf.reduce_all(tf.abs(mask - tf.transpose(mask)) < 1e-6)

        # Check values are in [0, 1]
        assert tf.reduce_all(mask >= 0.0)
        assert tf.reduce_all(mask <= 1.0)

    def test_matrix_kernel_computation(self, l96_system_simple):
        """Test matrix-valued kernel computation."""
        system = l96_system_simple

        pff = ParticleFlowFilter(
            dim=system['dim'],
            np_particles=20,
            nt=20,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=5,
            nx=system['nx'],
            R=system['R'],
            kernel_type='matrix'
        )

        # Create test particles and covariance
        tf.random.set_seed(42)
        particles = tf.random.normal((system['dim'], 20), dtype=tf.float32)
        B = tf.eye(system['dim'], dtype=tf.float32)

        # Compute kernel for dimension 0
        K, grad_K = pff.compute_matrix_kernel_and_gradient(particles, d=0, B=B)

        # Check shapes
        assert K.shape == (20, 20)
        assert grad_K.shape == (20, 20)

        # Check kernel is non-negative (can be very small/zero for distant particles)
        assert tf.reduce_all(K >= 0)

        # Check kernel diagonal is 1 (particle with itself)
        diagonal = tf.linalg.diag_part(K)
        assert tf.reduce_all(tf.abs(diagonal - 1.0) < 1e-5)

        # Check symmetry
        assert tf.reduce_all(tf.abs(K - tf.transpose(K)) < 1e-5)

    def test_scalar_kernel_computation(self, l96_system_simple):
        """Test scalar kernel computation."""
        system = l96_system_simple

        pff = ParticleFlowFilter(
            dim=10,  # Smaller dimension for faster inverse
            np_particles=20,
            nt=20,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=5,
            nx=10,
            R=tf.eye(4, dtype=tf.float32),
            kernel_type='scalar'
        )

        # Create test particles and covariance
        tf.random.set_seed(42)
        particles = tf.random.normal((10, 20), dtype=tf.float32)
        B = tf.eye(10, dtype=tf.float32) * 2.0

        # Compute kernel for dimension 0
        K, grad_K = pff.compute_scalar_kernel_and_divergence(particles, d=0, B=B)

        # Check shapes
        assert K.shape == (20, 20)
        assert grad_K.shape == (20, 20)

        # Check kernel is non-negative (can be very small/zero for distant particles)
        assert tf.reduce_all(K >= 0)

        # Check kernel diagonal is 1 (particle with itself)
        diagonal = tf.linalg.diag_part(K)
        assert tf.reduce_all(tf.abs(diagonal - 1.0) < 1e-5)

        # Check symmetry of kernel
        assert tf.reduce_all(tf.abs(K - tf.transpose(K)) < 1e-5)

    def test_regularized_inverse(self, l96_system_simple):
        """Test regularized matrix inverse."""
        system = l96_system_simple

        pff = ParticleFlowFilter(
            dim=system['dim'],
            np_particles=30,
            nt=20,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=5,
            nx=system['nx'],
            R=system['R']
        )

        # Create a nearly singular matrix
        A = tf.constant([
            [1.0, 0.99, 0.0],
            [0.99, 1.0, 0.0],
            [0.0, 0.0, 1e-10]
        ], dtype=tf.float32)

        # Compute regularized inverse
        A_inv = pff.regularized_inverse(A, cond_num=-8)

        # Check shape
        assert A_inv.shape == A.shape

        # Check no NaNs or Infs
        assert not tf.reduce_any(tf.math.is_nan(A_inv))
        assert not tf.reduce_any(tf.math.is_inf(A_inv))

class TestPFFIntegration:
    """Integration tests for full PFF workflow."""

    def test_single_assimilation_step_matrix_kernel(self, l96_system_simple):
        """Test a single PFF assimilation step with matrix kernel."""
        system = l96_system_simple

        # Generate short trajectory
        warm_nt = 100
        nt = 20
        tf.random.set_seed(42)
        Xt = generate_L96_trajectory(
            system['dim'], warm_nt, nt, system['dt'], system['F'], L96_RK4
        )

        # Generate observations
        y_obs, obs_indices, dim_indices = generate_observations(
            Xt, nt, warm_nt, system['obs_interval'],
            system['dim_interval'], system['R'], nx=system['nx']
        )

        total_obs = nt // system['obs_interval']

        # Initialize PFF
        pff = ParticleFlowFilter(
            dim=system['dim'],
            np_particles=20,
            nt=nt,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=total_obs,
            nx=system['nx'],
            R=system['R'],
            generate_Hx_si=system['generate_Hx_si_fn'],
            H_linear_adjoint=system['H_linear_adjoint_fn'],
            max_pseudo_step=30,
            kernel_type='matrix'
        )

        # Initialize ensemble
        tf.random.set_seed(123)
        X_list = []
        ctlmean = Xt[:, warm_nt] + tf.random.normal((system['dim'],), dtype=tf.float32)
        L_chol = tf.linalg.cholesky(system['Q'])
        initial_ensemble = tf.expand_dims(ctlmean, 1) + tf.matmul(
            L_chol, tf.random.normal((system['dim'], 20), dtype=tf.float32)
        )
        X_list.append(initial_ensemble)

        # Forward to observation time
        for _ in range(system['obs_interval']):
            X_next = system['model_step'](X_list[-1])
            X_list.append(X_next)

        # Perform assimilation at t=obs_interval-1 (which corresponds to obs_time=0)
        X_updated, s_end = pff.assimilate(
            X_list, y_obs, t=system['obs_interval']-1, obs_time=0, verbose=False
        )

        # Check outputs
        assert X_updated.shape == (system['dim'], 20)
        assert s_end > 0
        assert not tf.reduce_any(tf.math.is_nan(X_updated))
        assert not tf.reduce_any(tf.math.is_inf(X_updated))

    def test_single_assimilation_step_scalar_kernel(self, l96_system_simple):
        """Test a single PFF assimilation step with scalar kernel."""
        system = l96_system_simple

        # Generate short trajectory
        warm_nt = 100
        nt = 20
        tf.random.set_seed(42)
        Xt = generate_L96_trajectory(
            system['dim'], warm_nt, nt, system['dt'], system['F'], L96_RK4
        )

        # Generate observations
        y_obs, obs_indices, dim_indices = generate_observations(
            Xt, nt, warm_nt, system['obs_interval'],
            system['dim_interval'], system['R'], nx=system['nx']
        )

        total_obs = nt // system['obs_interval']

        # Initialize PFF with scalar kernel
        pff = ParticleFlowFilter(
            dim=system['dim'],
            np_particles=20,
            nt=nt,
            obs_interval=system['obs_interval'],
            dim_interval=system['dim_interval'],
            total_obs=total_obs,
            nx=system['nx'],
            R=system['R'],
            generate_Hx_si=system['generate_Hx_si_fn'],
            H_linear_adjoint=system['H_linear_adjoint_fn'],
            max_pseudo_step=30,
            kernel_type='scalar'
        )

        # Initialize ensemble
        tf.random.set_seed(123)
        X_list = []
        ctlmean = Xt[:, warm_nt] + tf.random.normal((system['dim'],), dtype=tf.float32)
        L_chol = tf.linalg.cholesky(system['Q'])
        initial_ensemble = tf.expand_dims(ctlmean, 1) + tf.matmul(
            L_chol, tf.random.normal((system['dim'], 20), dtype=tf.float32)
        )
        X_list.append(initial_ensemble)

        # Forward to observation time
        for _ in range(system['obs_interval']):
            X_next = system['model_step'](X_list[-1])
            X_list.append(X_next)

        # Perform assimilation at t=obs_interval-1 (which corresponds to obs_time=0)
        X_updated, s_end = pff.assimilate(
            X_list, y_obs, t=system['obs_interval']-1, obs_time=0, verbose=False
        )

        # Check outputs
        assert X_updated.shape == (system['dim'], 20)
        assert s_end > 0
        assert not tf.reduce_any(tf.math.is_nan(X_updated))
        assert not tf.reduce_any(tf.math.is_inf(X_updated))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
