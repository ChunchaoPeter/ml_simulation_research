"""
PFPF (Particle Filtering with Particle Flow) Implementation

This module implements the PFPF algorithm based on:
J. Li and M. J. Coates, "Particle filtering with invertible particle flow,"
IEEE Trans. Signal Process., vol. 65, no. 15, pp. 4102-4116, Aug. 2017.

The PFPF extends the EDH filter by adding:
- Importance weight updates accounting for the flow transformation
- Resampling based on effective sample size
- Proper proposal density calculation
"""

import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, Dict, Callable, Optional
from pfpf_edh import PFPF_EDH

# Import utility functions from prove_function.py
from prove_function import (
    particle_estimate,
    log_proposal_density,
    log_process_density,
    log_likehood_density
)

tfd = tfp.distributions


class PFPF_LEDH(PFPF_EDH):
    """
    Particle Filtering with Particle Flow using Localized EDH (PFPF_LEDH)

    Extends PFPF_EDH with local linearization at each particle.
    Implements Algorithm 1 from Li & Coates (2017).

    Key differences from PFPF_EDH:
        - Linearization performed at each particle's position (local)
        - Each particle has its own flow parameters A^i(λ) and b^i(λ)
        - Jacobian determinants computed for weight updates
        - Higher computational cost but better accuracy for nonlinear models

    Additional Attributes:
        theta: Jacobian determinant products for each particle, shape (n_particle,)
    """

    def __init__(
        self,
        observation_jacobian: Callable,
        observation_model: Callable,
        state_transition: Callable,
        n_particle: int = 100,
        n_lambda: int = 20,
        lambda_ratio: float = 1.2,
        use_ekf: bool = False,
        ekf_filter: Optional['ExtendedKalmanFilter'] = None,
        resample_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize PFPF_LEDH filter.

        Note: use_local is forced to True for LEDH implementation.

        Args:
            observation_jacobian: Callable that computes the observation Jacobian
            observation_model: Callable that maps state → observation
            state_transition: Callable for state propagation
            n_particle: Number of particles (default: 100)
            n_lambda: Number of lambda steps (default: 20)
            lambda_ratio: Exponential spacing ratio (default: 1.2)
            use_ekf: Use EKF for covariance tracking (default: False)
            ekf_filter: Pre-configured EKF filter instance
            resample_threshold: Resample when N_eff/N < threshold (default: 0.5)
            verbose: Print progress information (default: True)
        """
        # Initialize parent PFPF_EDH
        super().__init__(
            observation_jacobian, observation_model, state_transition,
            n_particle, n_lambda, lambda_ratio,
            use_local=False,  # We'll override flow method
            use_ekf=use_ekf, ekf_filter=ekf_filter,
            resample_threshold=resample_threshold, verbose=verbose
        )

        # LEDH-specific state
        self.theta = None  # Jacobian determinant products for each particle

    def initialize(self, model_params: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initialize particles with uniform weights.

        Extends parent initialization to include theta initialization.

        Args:
            model_params: Dictionary with model parameters

        Returns:
            particles: Initial particles, shape (state_dim, n_particle)
            m0: Random initial mean, shape (state_dim, 1)
        """
        particles, m0 = super().initialize(model_params)

        # Initialize theta for Jacobian determinants (Algorithm 1, Line 8)
        self.theta = tf.ones(self.n_particle, dtype=tf.float32)

        if self.verbose:
            print("  Initialized PFPF_LEDH (local linearization)")

        return particles, m0

    def _particle_flow_ledh(
        self,
        model_params: Dict,
        measurement: tf.Tensor
    ) -> tf.Tensor:
        """
        Migrate particles using LEDH flow (local linearization at each particle).

        Implements Algorithm 1, Lines 11-21 from Li & Coates (2017).

        Algorithm Lines 11-21 (LEDH):
        11  Set λ = 0
        12  for j = 1, ..., N_λ do
        13    Set λ = λ + ε_j
        14    for i = 1, ..., Np do
        15      Set η̄₀ = η̄ᵢ and P = Pᵢ
        16      Calculate Aⁱⱼ(λ) and bⁱⱼ(λ) with linearization at η̄ᵢ
        17      Migrate η̄ᵢ: η̄ᵢ = η̄ᵢ + εⱼ(Aⁱⱼ(λ)η̄ᵢ + bⁱⱼ(λ))
        18      Migrate particles: ηᵢ₁ = ηᵢ₁ + εⱼ(Aⁱⱼ(λ)ηᵢ₁ + bⁱⱼ(λ))
        19      Calculate θⁱ = θⁱ / |det(I + εⱼAⁱⱼ(λ))|
        20    endfor
        21  endfor

        Args:
            model_params: Dictionary with observation model
            measurement: Current measurement, shape (n_sensor, 1)

        Returns:
            theta: Jacobian determinant products for each particle
        """
        P_pred = self.P_pred

        # Initialize theta for tracking Jacobian determinants (Line 8)
        theta = tf.ones(self.n_particle, dtype=tf.float32)

        # Each particle gets its own auxiliary trajectory η̄ᵢ
        # Initialize with predicted particles
        eta_bar_i = tf.identity(self.particles)  # shape: (state_dim, n_particle)
        eta_bar_mu_0 = tf.expand_dims(self.mu_0, 1) if len(self.mu_0.shape) == 1 else self.mu_0

        # Algorithm Line 11: Set λ = 0
        # Algorithm Line 12: for j = 1, ..., N_λ do
        for j in range(self.n_lambda):
            # Algorithm Line 13: Set λ = λ + ε_j
            epsilon_j = self.lambda_steps[j]   # Step size
            lambda_j = self.lambda_values[j]   # Current lambda value

            # Algorithm Line 14: for i = 1, ..., Np do
            # We'll process all particles in parallel for efficiency

            # Lists to store per-particle flow parameters
            A_list = []
            b_list = []
            det_list = []

            for i in range(self.n_particle):
                # Algorithm Line 15: Set η̄₀ = η̄ᵢ and P = Pᵢ
                eta_bar_i_current = tf.expand_dims(eta_bar_i[:, i], 1)  # (state_dim, 1)

                # Algorithm Line 16: Calculate Aⁱⱼ(λ) and bⁱⱼ(λ)
                # with linearization at η̄ᵢ (local linearization)
                A_i, b_i = self._compute_flow_parameters(
                    eta_bar_i_current,  # Linearize at this particle
                    eta_bar_mu_0,
                    P_pred,
                    measurement,
                    lambda_j,
                    model_params
                )

                A_list.append(A_i)
                b_list.append(b_i)

                # Algorithm Line 19: Calculate θⁱ = θⁱ / |det(I + εⱼAⁱⱼ(λ))|
                I = tf.eye(self.particles.shape[0], dtype=tf.float32)
                det_i = tf.abs(tf.linalg.det(I + epsilon_j * A_i))
                det_list.append(det_i)

            # Stack flow parameters for vectorized computation
            A_stacked = tf.stack(A_list, axis=0)  # (n_particle, state_dim, state_dim)
            b_stacked = tf.stack(b_list, axis=0)  # (n_particle, state_dim)
            det_stacked = tf.stack(det_list, axis=0)  # (n_particle,)

            # Update theta (product of determinants)
            theta = theta / det_stacked

            # Algorithm Line 17: Migrate η̄ᵢ
            # Algorithm Line 18: Migrate particles
            for i in range(self.n_particle):
                A_i = A_stacked[i]
                b_i = b_stacked[i]

                # Migrate auxiliary trajectory
                slope_bar_i = tf.linalg.matvec(A_i, eta_bar_i[:, i]) + b_i
                eta_bar_i = tf.tensor_scatter_nd_update(
                    eta_bar_i,
                    [[k, i] for k in range(eta_bar_i.shape[0])],
                    eta_bar_i[:, i] + epsilon_j * slope_bar_i
                )

                # Migrate particle
                slope_i = tf.linalg.matvec(A_i, self.particles[:, i]) + b_i
                self.particles = tf.tensor_scatter_nd_update(
                    self.particles,
                    [[k, i] for k in range(self.particles.shape[0])],
                    self.particles[:, i] + epsilon_j * slope_i
                )

        return theta

    def step(
        self,
        measurement: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Perform one PFPF_LEDH filter step.

        Implements Algorithm 1 from Li & Coates (2017).
        Uses local linearization for particle flow.

        Args:
            measurement: Current measurement, shape (n_sensor, 1) or (n_sensor,)
            model_params: Dictionary with model parameters

        Returns:
            particles_updated: Updated particles
            mean_estimate: Weighted state estimate
            P_updated: Posterior covariance
            N_eff: Effective sample size
        """
        # Ensure measurement is (n_sensor, 1)
        if len(measurement.shape) == 1:
            measurement = tf.expand_dims(measurement, 1)

        # Store previous estimate
        x_est_prev = tf.expand_dims(self.M, 1) if len(self.M.shape) == 1 else self.M

        # Step 1: Propagate particles (Algorithm 1, Lines 6-7)
        self.particles_pred = self._propagate_particles(self.particles, model_params)
        self.particles_pred_deterministic = self._propagate_particles_deterministic(
            self.particles, model_params
        )

        # Step 2: Estimate prior covariance (Algorithm 1, Line 5)
        if self.use_ekf:
            x_ekf_pred, self.P_pred = self._ekf_predict(self.x_ekf, self.P)
        else:
            self.P_pred = self._estimate_covariance(self.particles_pred, model_params)

        # Step 3: Update mean trajectory mu_0 (Algorithm 1, Line 9)
        self.mu_0 = self.state_transition(x_est_prev, model_params, no_noise=True)

        self.particles_previous = self.particles
        self.particles = self.particles_pred

        # Step 4: Update auxiliary trajectory for linearization
        mean_estimate, _ = particle_estimate(self.log_weights, self.particles)
        self.auxiliary_trajectory = mean_estimate
        self.particles_mean = mean_estimate

        # Step 5: LEDH Particle flow (Algorithm 1, Lines 11-21)
        theta = self._particle_flow_ledh(model_params, measurement)

        # Step 6: PFPF weight update with Jacobian determinants (Algorithm 1, Line 24)
        # w^i_k = p(x^i_k|x^i_{k-1})p(z_k|x^i_k)θ^i / p(η^i_0|x^i_{k-1}) * w^i_{k-1}
        weights, log_weights = self._update_weights_ledh(
            self.particles,
            self.particles_pred,
            self.particles_pred_deterministic,
            measurement,
            theta,  # Use Jacobian determinants
            model_params
        )

        # Step 7: Weighted estimate (Algorithm 1, Line 27)
        mean_estimate, _ = particle_estimate(log_weights, self.particles)
        self.particles_mean = mean_estimate

        # Step 8: Covariance update (Algorithm 1, Line 26)
        if self.use_ekf:
            x_ekf_updated, P_updated = self._ekf_update(x_ekf_pred, self.P_pred, measurement)
            self.x_ekf = x_ekf_updated
        else:
            P_updated = self._estimate_covariance(self.particles, model_params)

        # Step 9: Resample if needed
        particles_resampled, weights_resampled, log_weights_resampled, N_eff = self._resample(
            self.particles, weights
        )

        # Step 10: Update all internal state
        self.particles = particles_resampled
        self.weights = weights_resampled
        self.log_weights = log_weights_resampled
        self.particles_mean = mean_estimate
        self.P = P_updated
        self.M = mean_estimate

        return particles_resampled, mean_estimate, P_updated, N_eff

    def _update_weights_ledh(
        self,
        particles_flowed: tf.Tensor,
        particles_pred: tf.Tensor,
        particles_pred_deterministic: tf.Tensor,
        measurement: tf.Tensor,
        theta: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Update particle weights using PFPF_LEDH weight formula with Jacobian determinants.

        Implements Algorithm 1, Line 24.

        Weight update formula (with invertible mapping):
            w^i_k ∝ [p(x^i_k|x^i_{k-1}) * p(z_k|x^i_k) * θ^i] / p(η^i_0|x^i_{k-1}) * w^i_{k-1}

        where θ^i = ∏_j |det(I + ε_j A^i_j(λ))| is the Jacobian determinant product.

        Args:
            particles_flowed: Particles after flow (x_k)
            particles_pred: Propagated particles WITH noise (η_0)
            particles_pred_deterministic: Deterministic propagation (Φ·x_{k-1})
            measurement: Current measurement
            theta: Jacobian determinant products for each particle
            model_params: Dictionary with model parameters

        Returns:
            weights: Updated normalized weights
            log_weights: Updated log weights
        """
        Q = model_params['Q']

        # Calculate log proposal density
        # For LEDH, we use the Jacobian determinant in the proposal
        # log q(x_k|x_{k-1}, z_k) = log p(η_0|x_{k-1}) - log θ^i
        log_proposal = log_proposal_density(
            particles_pred, particles_pred_deterministic, Q, 0.0  # No log_jacobian in standard calculation
        )
        # Subtract log(theta) to account for the invertible mapping
        log_proposal = log_proposal - tf.math.log(theta)

        # Calculate log process/prior density
        log_prior = log_process_density(
            particles_flowed, particles_pred_deterministic, Q
        )

        # Calculate log likelihood
        llh = log_likehood_density(particles_flowed, measurement, model_params)

        # Weight update
        log_weights_updated = log_prior + llh - log_proposal + self.log_weights

        # Normalize by subtracting max for numerical stability
        log_weights_updated = log_weights_updated - tf.reduce_max(log_weights_updated)

        # Convert to linear weights
        _, weights = particle_estimate(log_weights_updated, particles_flowed)

        return weights, log_weights_updated

    def run(
        self,
        measurements: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Run PFPF_LEDH filter on measurement sequence.

        Args:
            measurements: Measurements, shape (n_sensor, T)
            model_params: Dictionary with model parameters

        Returns:
            estimates: State estimates, shape (state_dim, T)
            particles_all: All particles, shape (state_dim, n_particle, T)
            covariances_all: All covariances, shape (state_dim, state_dim, T)
            Neff_all: Effective sample sizes, shape (T, 1)
        """
        T = tf.shape(measurements)[1].numpy()

        if self.verbose:
            print(f"\nRunning PFPF_LEDH Filter (Local Linearization):")
            print(f"  Particles: {self.n_particle}")
            print(f"  Lambda steps: {self.n_lambda}")
            print(f"  Lambda ratio: {self.lambda_ratio}")
            print(f"  Linearization: Local (LEDH)")
            print(f"  EKF Covariance: {'Enabled' if self.use_ekf else 'Disabled'}")
            print(f"  Resample threshold: {self.resample_threshold}")
            print(f"  Time steps: {T}")

        # Initialize
        self.initialize(model_params)

        # Storage
        estimates_list = []
        particles_list = []
        covariances_list = []
        Neff_list = []

        # Main loop
        if self.verbose:
            print("\nProcessing time steps...")

        for t in range(T):
            if self.verbose and (t + 1) % 10 == 0:
                print(f"  Step {t+1}/{T}")

            z_t = tf.expand_dims(measurements[:, t], 1)

            # Filter step
            particles, mean_estimate, P, Neff = self.step(z_t, model_params)

            # Store results
            estimates_list.append(tf.expand_dims(mean_estimate, 1))
            particles_list.append(tf.expand_dims(particles, 2))
            covariances_list.append(tf.expand_dims(P, 2))
            Neff_list.append(Neff)

        # Concatenate results
        estimates = tf.concat(estimates_list, axis=1)
        particles_all = tf.concat(particles_list, axis=2)
        covariances_all = tf.concat(covariances_list, axis=2)
        combined = tf.stack(Neff_list, axis=0)
        Neff_all = tf.reshape(combined, (-1, 1))

        if self.verbose:
            print("\nPFPF_LEDH filter completed successfully!")
            print(f"  Estimates shape: {estimates.shape}")
            print(f"  Particles shape: {particles_all.shape}")
            print(f"  Covariances shape: {covariances_all.shape}")
            print(f"  Effective sample size: {Neff_all.shape}")

        return estimates, particles_all, covariances_all, Neff_all
