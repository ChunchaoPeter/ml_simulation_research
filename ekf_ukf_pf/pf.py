"""
Particle Filter Implementation 
-------------------------------------------------------------
Pseudocode

At time t = 0:
  • Sample X_0^i ~ q_1(·) = μ(·)
  • Set uniform weights: w_0^i = 1/N

At time t ≥ 1:
  • Predict: X_t^i ~ q_{t|t-1}(·|X_{t-1}^i) [prior proposal]
  • Update: w_t^i ∝ g(y_t|X_t^i) [observation likelihood]
  • Estimate: x̂_t = Σ_{i=1}^N w_t^i X_t^i
  • Resample: Draw ancestry A^i ~ Cat(w_1,...,w_N)

PRIOR PROPOSAL:
  q_{t|t-1}(x_t|x_{t-1}) = f(x_t|x_{t-1})
  Weight simplifies to: w_t^i ∝ g(y_t|X_t^i)

EXAMPLE USAGE:
--------------
For custom models:
    pf = ParticleFilter(
        state_transition_fn=my_f,           # f(x, u) -> x_next
        observation_fn=my_h,                # h(x) -> z
        process_noise_sampler=my_noise,    # (N) -> noise samples
        observation_likelihood_fn=my_lik,  # (z, x) -> likelihood
        x0_sampler=my_initial,              # (N) -> initial states
        num_particles=100
    )
    filtered_states = pf.filter(observations)

For specific applications (e.g., range-bearing tracking), see pf_demo_code.ipynb
which shows how to create specialized wrapper classes.

REFERENCES:
-----------
1. Doucet, A., & Johansen, A. M. (2009). A tutorial on particle filtering
   and smoothing: Fifteen years later.
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class ParticleFilter:
    """
    General Particle Filter

    This implementation uses PRIOR PROPOSAL:
    - Particles sampled from state transition: q_{t|t-1} = f
    - Weights computed from observation likelihood: w_t^i ∝ g(y_t|X_t^i)

    Parameters:
    -----------
    state_transition_fn : callable
        Non-linear state transition function f(x, u=None)
        Input: x (state_dim, 1), u (control_dim, 1) or None
        Output: (state_dim, 1)
    observation_fn : callable
        Non-linear observation function h(x)
        Input: x (state_dim, 1)
        Output: (obs_dim, 1)
    process_noise_sampler : callable
        Function to sample process noise
        Input: num_samples (int)
        Output: (state_dim, num_samples) noise samples
    observation_likelihood_fn : callable
        Function to compute p(z|x)
        Input: z (obs_dim, 1), x (state_dim, 1)
        Output: scalar likelihood
    x0_sampler : callable
        Function to sample initial states
        Input: num_samples (int)
        Output: (state_dim, num_samples) initial state samples
    num_particles : int
        Number of particles N
    resample_method : str
        Resampling method: 'multinomial'
    dtype : tf.DType
        Data type for computations
    """

    def __init__(self, state_transition_fn, observation_fn,
                 process_noise_sampler, observation_likelihood_fn, x0_sampler,
                 num_particles=100, resample_method='multinomial',
                 dtype=tf.float64):
        # Validate required functions
        if state_transition_fn is None:
            raise ValueError(
                "state_transition_fn is required and cannot be None. "
                "Please provide a function f(x, u=None) that computes the state transition."
            )
        if observation_fn is None:
            raise ValueError(
                "observation_fn is required and cannot be None. "
                "Please provide a function h(x) that computes the observation."
            )
        if process_noise_sampler is None:
            raise ValueError(
                "process_noise_sampler is required and cannot be None. "
                "Please provide a function that samples process noise: (num_samples) -> (state_dim, num_samples)."
            )
        if observation_likelihood_fn is None:
            raise ValueError(
                "observation_likelihood_fn is required and cannot be None. "
                "Please provide a function p(z|x) that computes observation likelihood."
            )
        if x0_sampler is None:
            raise ValueError(
                "x0_sampler is required and cannot be None. "
                "Please provide a function that samples initial states: (num_samples) -> (state_dim, num_samples)."
            )

        self.f = state_transition_fn
        self.h = observation_fn
        self.sample_process_noise = process_noise_sampler
        self.obs_likelihood = observation_likelihood_fn
        self.sample_initial_states = x0_sampler
        self.num_particles = num_particles
        self.resample_method = resample_method
        self.dtype = dtype


    @tf.function
    def multinomial_resample(self, weights):
        """
        Multinomial resampling.

        Draws ancestry vector A^{1:N} where A^i ~ Cat(w^1,...,w^N)

        Args:
            weights: Normalized particle weights (N,)

        Returns:
            indices: Resampled particle indices (N,)
        """
        # Sample from categorical distribution using TensorFlow Probability
        # This implements: A^i ~ Cat(w^1,...,w^N)
        categorical = tfd.Categorical(probs=weights)
        indices = categorical.sample(self.num_particles)

        # Cast to int32 for consistency
        return tf.cast(indices, tf.int32)

    def resample(self, weights):
        """
        Resample particles according to weights using specified method.

        Args:
            weights: Normalized particle weights (N,)

        Returns:
            indices: Resampled particle indices (N,)
        """
        if self.resample_method == 'multinomial':
            return self.multinomial_resample(weights)
        else:
            raise ValueError(f"Unknown resample method: {self.resample_method}")

    @tf.function
    def effective_sample_size(self, weights):
        """
        Compute effective sample size N_eff = 1 / sum(w_i^2).

        Args:
            weights: Normalized particle weights (N,)

        Returns:
            n_eff: Effective sample size
        """
        return 1.0 / tf.reduce_sum(weights ** 2)

    def force_resample(self, particles, weights):
        """
        Force resampling of all particles, ignoring the effective sample size.

        Args:
            particles: Current particles (state_dim, num_particles)
            weights: Current weights (num_particles,)

        Returns:
            particles: Resampled particles (state_dim, num_particles)
            weights: Uniform weights after resampling
            ancestry: Ancestry indices used for resampling
        """

        # Always resample
        indices = self.resample(weights)

        # Resample particles
        particles = tf.gather(particles, indices, axis=1)

        # Reset weights to uniform
        weights = tf.ones(self.num_particles, dtype=self.dtype) / self.num_particles

        ancestry = indices

        return particles, weights, ancestry


    @tf.function
    def predict(self, particles, u=None):
        """
        Prediction step: propagate particles through state transition.

        x_t^(i) = f(x_{t-1}^(i), u_t) + w_t^(i)

        Args:
            particles: Previous particles (state_dim, num_particles)
            u: Control input (control_dim, 1) or None

        Returns:
            particles_pred: Predicted particles (state_dim, num_particles)
        """
        particles_pred = []

        # Propagate each particle
        for i in range(self.num_particles):
            particle = particles[:, i:i+1]

            # Apply state transition
            if u is not None:
                particle_next = self.f(particle, u)
            else:
                particle_next = self.f(particle)

            particles_pred.append(particle_next)

        particles_pred = tf.concat(particles_pred, axis=1)

        # Add process noise
        noise = self.sample_process_noise(self.num_particles)
        particles_pred = particles_pred + noise

        return particles_pred

    @tf.function
    def update(self, z, particles):
        """
        Update step: compute particle weights based on observation likelihood.

        w_t^(i) ∝ p(z_t | x_t^(i))

        Args:
            z: Observation (obs_dim, 1)
            particles: Predicted particles (state_dim, num_particles)

        Returns:
            weights: Normalized particle weights (num_particles,)
        """
        weights = []

        # Compute likelihood for each particle
        for i in range(self.num_particles):
            particle = particles[:, i:i+1]
            likelihood = self.obs_likelihood(z, particle)
            weights.append(likelihood)

        weights = tf.stack(weights)

        # Normalize weights
        weights = weights / (tf.reduce_sum(weights) + 1e-10)

        return weights

    @tf.function
    def estimate_state(self, particles, weights):
        """
        Estimate state from weighted particles (weighted mean).

        Args:
            particles: Particles (state_dim, num_particles)
            weights: Normalized weights (num_particles,)

        Returns:
            x_est: Estimated state (state_dim, 1)
        """
        x_est = tf.reduce_sum(
            particles * weights[None, :],
            axis=1, keepdims=True
        )
        return x_est

    @tf.function
    def filter(self, observations, controls=None, return_details=False):
        """
        Run particle filter on observation sequence (Algorithm 3 from Notes 9).

        Args:
            observations: Observation sequence (obs_dim, T)
            controls: Control input sequence (control_dim, T) or None
            return_details: If True, return additional details for analysis

        Returns:
            filtered_states: Filtered state estimates (state_dim, T+1)
            predicted_states: Predicted state estimates (state_dim, T) [if return_details]
            particles_history: Particle history (state_dim, num_particles, T+1) [if return_details]
            weights_history: Weight history (num_particles, T+1) [if return_details]
            ess_history: Effective sample size history (T+1,) [if return_details]
            ancestry_history: Ancestry indices (num_particles, T+1) [if return_details]
        """
        observations = tf.convert_to_tensor(observations, dtype=self.dtype)
        T = observations.shape[1]

        # ============================================================
        # TIME t = 0: INITIALIZATION 
        # ============================================================
        # Sample initial particles: X_0^i ~ q_1(·) = μ(·)
        particles = self.sample_initial_states(self.num_particles)

        # Initialize uniform weights: w_0^i = 1/N
        weights = tf.ones(self.num_particles, dtype=self.dtype) / self.num_particles

        # Storage for results
        filtered_states = []
        predicted_states = []
        particles_history = []
        weights_history = []
        ess_history = []
        ancestry_history = []

        # Compute and store initial state estimate (t=0)
        x_est = self.estimate_state(particles, weights)
        filtered_states.append(x_est)

        if return_details:
            particles_history.append(particles)
            weights_history.append(weights)
            ess_history.append(self.effective_sample_size(weights))
            ancestry_history.append(tf.range(self.num_particles, dtype=tf.int32))

        # ============================================================
        # TIME t = 1 to T: SEQUENTIAL UPDATES (Algorithm 3 main loop)
        # ============================================================
        for t in range(T):
            u = controls[:, t:t+1] if controls is not None else None
            z = observations[:, t:t+1]

            # --------------------------------------------------------
            # STEP 1: PREDICTION (sample from prior proposal)
            # --------------------------------------------------------
            particles = self.predict(particles, u)

            # Compute prediction estimate (before measurement update)
            # Use current weights (uniform after resampling, or previous weights)
            x_pred = self.estimate_state(particles, weights)
            predicted_states.append(x_pred)

            # --------------------------------------------------------
            # STEP 2: UPDATE (compute weights)
            # --------------------------------------------------------
            weights = self.update(z, particles)

            # --------------------------------------------------------
            # STEP 3: STATE ESTIMATION
            # --------------------------------------------------------
            x_est = self.estimate_state(particles, weights)

            # Store
            filtered_states.append(x_est)
            if return_details:
                particles_history.append(particles)
                weights_history.append(weights)
                ess_history.append(self.effective_sample_size(weights))

            # --------------------------------------------------------
            # STEP 4: RESAMPLING (at the end, for next iteration)
            # --------------------------------------------------------
            particles, weights, ancestry = self.force_resample(particles, weights)

            if return_details:
                ancestry_history.append(ancestry)

        # Convert to tensors
        filtered_states = tf.concat(filtered_states, axis=1)  # (state_dim, T+1)
        predicted_states = tf.concat(predicted_states, axis=1)  # (state_dim, T)

        if return_details:
            particles_history = tf.stack(particles_history, axis=2)  # (state_dim, num_particles, T+1)
            weights_history = tf.stack(weights_history, axis=1)  # (num_particles, T+1)
            ess_history = tf.stack(ess_history)  # (T+1,)
            ancestry_history = tf.stack(ancestry_history, axis=1)  # (num_particles, T+1)

            return (filtered_states, predicted_states, particles_history, weights_history,
                    ess_history, ancestry_history)
        else:
            return filtered_states
