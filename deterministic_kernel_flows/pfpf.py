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
from edh import EDHFilter

# Import utility functions from prove_function.py
from prove_function import (
    particle_estimate,
    log_proposal_density,
    log_process_density,
    log_likehood_density,
    multinomial_resample
)

tfd = tfp.distributions


class PFPF(EDHFilter):
    """
    Particle Filtering with Particle Flow (PFPF)

    Extends EDHFilter with importance weighting and resampling.
    Uses particle flow as a proposal distribution with proper weight updates.

    This class replaces the vg dictionary from PFPF_function_clean_EDH.ipynb
    with proper instance variables (self.*).

    Mapping from notebook's vg dictionary to class attributes:
        vg['particles']                -> self.particles
        vg['logW']                     -> self.log_weights
        vg['ml_weights']               -> self.weights
        vg['particles_m']              -> self.particles_mean
        vg['M']                        -> self.m0 (initial mean)
        vg['PU']                       -> self.P (posterior covariance)
        vg['PP']                       -> self.P_pred (prior covariance)
        vg['mu_0']                     -> self.mu_0 (mean trajectory)
        vg['xp_prop']                  -> self.particles_pred
        vg['xp_prop_deterministic']    -> self.particles_pred_deterministic
        vg['xp_auxiliary_individual']  -> self.auxiliary_trajectory

    Additional Attributes:
        weights: Particle weights, shape (n_particle,)
        log_weights: Log particle weights, shape (n_particle,)
        resample_threshold: Effective sample size threshold for resampling
        particles_pred: Propagated particles WITH noise
        particles_pred_deterministic: Propagated particles WITHOUT noise
        particles_mean: Mean of particle distribution
        mu_0: Mean trajectory for linearization
        auxiliary_trajectory: Auxiliary trajectory point
        P_pred: Prior covariance estimate
    """

    def __init__(
        self,
        observation_jacobian: Callable,
        observation_model: Callable,
        n_particle: int = 100,
        n_lambda: int = 20,
        lambda_ratio: float = 1.2,
        use_local: bool = False,
        use_ekf: bool = False,
        ekf_filter: Optional['ExtendedKalmanFilter'] = None,
        resample_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize PFPF filter.

        Args:
            observation_jacobian: Callable that computes the observation Jacobian
            observation_model: Callable that maps state → observation
            n_particle: Number of particles (default: 100)
            n_lambda: Number of lambda steps (default: 20)
            lambda_ratio: Exponential spacing ratio (default: 1.2)
            use_local: Use local linearization (default: False)
            use_ekf: Use EKF for covariance tracking (default: False)
            ekf_filter: Pre-configured EKF filter instance
            resample_threshold: Resample when N_eff/N < threshold (default: 0.5)
            verbose: Print progress information (default: True)
        """
        super().__init__(
            observation_jacobian, observation_model,
            n_particle, n_lambda, lambda_ratio,
            use_local, use_ekf, ekf_filter, verbose
        )

        self.resample_threshold = resample_threshold

        # Additional PFPF state
        self.weights = None
        self.log_weights = None

        # Intermediate computation states 
        self.particles_pred = None  # Propagated particles WITH noise
        self.particles_pred_deterministic = None  # Propagated WITHOUT noise
        self.particles_mean = None  # Mean of particles
        self.mu_0 = None  # Mean trajectory η̄_μ0
        self.auxiliary_trajectory = None  # Auxiliary trajectory for linearization
        self.P_pred = None  # Prior covariance P_{k|k-1}

    def initialize(self, model_params: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initialize particles with uniform weights.

        Overrides EDHFilter.initialize to add weight initialization.
        Initializes all vg-like state variables.

        Args:
            model_params: Dictionary with model parameters

        Returns:
            particles: Initial particles, shape (state_dim, n_particle)
            m0: Random initial mean, shape (state_dim, 1)
        """
        # Call parent initialization
        particles, m0 = super().initialize(model_params)

        # Initialize uniform weights
        self.weights = tf.ones(self.n_particle, dtype=tf.float32) / self.n_particle
        self.log_weights = tf.math.log(self.weights)

        # Initialize intermediate states
        self.particles_mean = tf.squeeze(m0)  # Convert (state_dim, 1) to (state_dim,)
        self.mu_0 = m0  # Mean trajectory
        self.auxiliary_trajectory = m0  # Initially same as mu_0
        self.P_pred = self.P  # Initial covariance

        if self.verbose:
            print("  Initialized uniform particle weights")
            print("  Initialized PFPF state variables")

        return particles, m0

    def _propagate_particles_deterministic(
        self,
        particles: tf.Tensor,
        model_params: Dict
    ) -> tf.Tensor:
        """
        Propagate particles through motion model WITHOUT noise.

        Used for computing proposal density.

        Args:
            particles: Current particles, shape (state_dim, n_particle)
            model_params: Dictionary with model parameters

        Returns:
            particles_pred: Predicted particles (deterministic), shape (state_dim, n_particle)
        """
        Phi = model_params['Phi']
        return tf.matmul(Phi, particles)

    def _update_weights(
        self,
        particles_flowed: tf.Tensor,
        particles_pred: tf.Tensor,
        particles_pred_deterministic: tf.Tensor,
        measurement: tf.Tensor,
        log_jacobian_det_sum: float,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Update particle weights using PFPF weight formula.

        Weight update formula:
            w_k ∝ [p(x_k|x_{k-1}) * p(z_k|x_k) / q(x_k|x_{k-1},z_k)] * w_{k-1}

        In log space:
            log w_k = log p(x_k|x_{k-1}) + log p(z_k|x_k) - log q(...) + log w_{k-1}

        Args:
            particles_flowed: Particles after flow (x_k)
            particles_pred: Propagated particles WITH noise (η_0)
            particles_pred_deterministic: Deterministic propagation (Φ·x_{k-1})
            measurement: Current measurement
            log_jacobian_det_sum: Sum of log Jacobian determinants
            model_params: Dictionary with model parameters

        Returns:
            weights: Updated normalized weights
            log_weights: Updated log weights
        """
        Q = model_params['Q']

        # Calculate log proposal density using prove_function.py
        log_proposal = log_proposal_density(
            particles_pred, particles_pred_deterministic, Q, log_jacobian_det_sum
        )

        # Calculate log process/prior density using prove_function.py
        log_prior = log_process_density(
            particles_flowed, particles_pred_deterministic, Q
        )

        # Calculate log likelihood using prove_function.py
        llh = log_likehood_density(particles_flowed, measurement, model_params)

        # Weight update
        log_weights_updated = log_prior + llh - log_proposal + self.log_weights

        # Normalize by subtracting max for numerical stability
        log_weights_updated = log_weights_updated - tf.reduce_max(log_weights_updated)

        # Convert to linear weights using particle_estimate
        # This normalizes the weights properly
        _, weights = particle_estimate(log_weights_updated, particles_flowed)

        return weights, log_weights_updated

    @tf.function
    def _compute_effective_sample_size(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Compute effective sample size N_eff = 1 / Σ(w_i^2).

        Args:
            weights: Normalized particle weights

        Returns:
            N_eff: Effective sample size
        """
        return 1.0 / tf.reduce_sum(weights ** 2)

    def _resample(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, float]:
        """
        Resample particles based on effective sample size.

        Args:
            particles: Current particles
            weights: Current weights

        Returns:
            particles: Resampled particles (or original if no resampling)
            weights: Updated weights (uniform if resampled)
            log_weights: Updated log weights
            N_eff: Effective sample size
        """
        N_eff = self._compute_effective_sample_size(weights)
        threshold = self.resample_threshold * self.n_particle

        if N_eff < threshold:
            # Resample
            indices = multinomial_resample(weights)
            particles = tf.gather(particles, indices, axis=1)

            # Reset to uniform weights
            weights = tf.ones(self.n_particle, dtype=tf.float32) / self.n_particle
            log_weights = tf.math.log(weights)

            if self.verbose:
                print(f"    Resampled (N_eff={N_eff.numpy():.1f})")
        else:
            log_weights = tf.math.log(weights)

        return particles, weights, log_weights, N_eff.numpy()

    def step(
        self,
        measurement: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Perform one PFPF filter step.

        Follows the workflow from PFPF_function_clean_EDH.ipynb using self.* state.
        Extends EDH step with weight updates and resampling.

        Args:
            measurement: Current measurement, shape (n_sensor, 1) or (n_sensor,)
            model_params: Dictionary with model parameters

        Returns:
            particles_updated: Updated particles
            mean_estimate: Weighted state estimate
            P_updated: Posterior covariance
        """
        # Ensure measurement is (n_sensor, 1)
        if len(measurement.shape) == 1:
            measurement = tf.expand_dims(measurement, 1)

        # Store previous estimate (like x_est_prev in notebook)
        x_est_prev = tf.expand_dims(self.particles_mean, 1) if len(self.particles_mean.shape) == 1 else self.particles_mean

        # Step 1: Propagate particles (with and without noise)
        # This corresponds to propagateAndEstimatePriorCovariance in notebook
        self.particles_pred = self._propagate_particles(self.particles, model_params)
        self.particles_pred_deterministic = self._propagate_particles_deterministic(
            self.particles, model_params
        )

        # Step 2: Estimate prior covariance (vg['PP'] in notebook)
        if self.use_ekf:
            x_ekf_pred, self.P_pred = self._ekf_predict(self.x_ekf, self.P)
        else:
            self.P_pred = self._estimate_covariance(self.particles_pred, model_params)

        # Step 3: Update mean trajectory mu_0 (vg['mu_0'] in notebook)
        # This is the deterministic propagation of the previous estimate
        Phi = model_params['Phi']
        self.mu_0 = tf.matmul(Phi, x_est_prev)

        # Step 4: Update auxiliary trajectory for linearization (vg['xp_auxiliary_individual'])
        # For EDH, this is the same as mu_0 (global linearization point)
        self.auxiliary_trajectory = self.mu_0

        # Step 5: Particle flow
        # Uses self.P_pred and self.auxiliary_trajectory for linearization
        particles_flowed = self._particle_flow(
            self.particles_pred, measurement, self.P_pred, model_params
        )

        # Step 6: PFPF weight update
        log_jacobian_det_sum = 0.0  # Can be computed for local linearization
        weights, log_weights = self._update_weights(
            particles_flowed,
            self.particles_pred,
            self.particles_pred_deterministic,
            measurement,
            log_jacobian_det_sum,
            model_params
        )

        # Step 7: Weighted estimate using particle_estimate from prove_function.py
        mean_estimate, _ = particle_estimate(log_weights, particles_flowed)

        # Step 8: Covariance update (vg['PU'] in notebook)
        if self.use_ekf:
            x_ekf_updated, P_updated = self._ekf_update(x_ekf_pred, self.P_pred, measurement)
            self.x_ekf = x_ekf_updated
        else:
            P_updated = self._estimate_covariance(particles_flowed, model_params)

        # Step 9: Resample if needed
        particles_resampled, weights_resampled, log_weights_resampled, N_eff = self._resample(
            particles_flowed, weights
        )

        # Step 10: Update all internal state (replaces vg dictionary updates)
        self.particles = particles_resampled
        self.weights = weights_resampled
        self.log_weights = log_weights_resampled
        self.particles_mean = mean_estimate
        self.P = P_updated

        return particles_resampled, mean_estimate, P_updated

    def run(
        self,
        measurements: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Run PFPF filter on measurement sequence.

        Overrides EDHFilter.run with PFPF-specific processing.

        Args:
            measurements: Measurements, shape (n_sensor, T)
            model_params: Dictionary with model parameters

        Returns:
            estimates: State estimates, shape (state_dim, T)
            particles_all: All particles, shape (state_dim, n_particle, T)
            covariances_all: All covariances, shape (state_dim, state_dim, T)
        """
        T = tf.shape(measurements)[1].numpy()

        if self.verbose:
            print(f"\nRunning PFPF Filter:")
            print(f"  Particles: {self.n_particle}")
            print(f"  Lambda steps: {self.n_lambda}")
            print(f"  Lambda ratio: {self.lambda_ratio}")
            print(f"  Linearization: {'Local' if self.use_local else 'Global (EDH)'}")
            print(f"  EKF Covariance: {'Enabled' if self.use_ekf else 'Disabled'}")
            print(f"  Resample threshold: {self.resample_threshold}")
            print(f"  Time steps: {T}")

        # Initialize
        self.initialize(model_params)

        # Storage
        estimates_list = []
        particles_list = []
        covariances_list = []

        # Main loop
        if self.verbose:
            print("\nProcessing time steps...")

        for t in range(T):
            if self.verbose and (t + 1) % 10 == 0:
                print(f"  Step {t+1}/{T}")

            z_t = tf.expand_dims(measurements[:, t], 1)

            # Filter step
            particles, mean_estimate, P = self.step(z_t, model_params)

            # Store results
            estimates_list.append(tf.expand_dims(mean_estimate, 1))
            particles_list.append(tf.expand_dims(particles, 2))
            covariances_list.append(tf.expand_dims(P, 2))

        # Concatenate results
        estimates = tf.concat(estimates_list, axis=1)
        particles_all = tf.concat(particles_list, axis=2)
        covariances_all = tf.concat(covariances_list, axis=2)

        if self.verbose:
            print("\nPFPF filter completed successfully!")
            print(f"  Estimates shape: {estimates.shape}")
            print(f"  Particles shape: {particles_all.shape}")
            print(f"  Covariances shape: {covariances_all.shape}")

        return estimates, particles_all, covariances_all


