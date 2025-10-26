"""
Unscented Kalman Filter for Non-Linear Gaussian State Space Models

This module provides an Unscented Kalman Filter implementation using the
unscented transform with sigma points.

The UKF handles non-linear functions by propagating carefully chosen sigma points
through the non-linear transformations, providing better approximations than EKF
for highly non-linear systems.

Reference:
Wan, E. A., & Van Der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation.
In Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium.
"""

import tensorflow as tf


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for Non-Linear Gaussian State Space Models.

    State equation: x_t = f(x_{t-1}, u_t) + w_t, w_t ~ N(0, Q_t)
    Observation equation: z_t = h(x_t) + v_t, v_t ~ N(0, R_t)

    The UKF uses the unscented transform to handle non-linearities:
    1. Generate sigma points around the mean
    2. Propagate sigma points through non-linear function
    3. Compute statistics from transformed sigma points

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
    Q : tf.Tensor, shape (state_dim, state_dim)
        Process noise covariance
    R : tf.Tensor, shape (obs_dim, obs_dim)
        Observation noise covariance
    x0 : tf.Tensor, shape (state_dim, 1) or (state_dim,)
        Initial state mean (column vector)
    Sigma0 : tf.Tensor, shape (state_dim, state_dim)
        Initial state covariance
    alpha : float, optional (default: 1.0 for our 4-dimensional example)
        Controls the spread of the sigma points. 
        Be careful with very small values (e.g., 1e-3), as they can cause numerical instability in this case.
    beta : float, optional (default: 2.0)
        Incorporates prior knowledge (beta=2 optimal for Gaussian)
    kappa : float, optional (default: 0.0)
        Secondary scaling parameter (typically 0 or 3-n)
    """

    def __init__(self, state_transition_fn, observation_fn, Q, R, x0, Sigma0,
                 alpha=1, beta=2.0, kappa=0.0):
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

        self.f = state_transition_fn
        self.h = observation_fn
        self.Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        self.R = tf.convert_to_tensor(R, dtype=tf.float32)
        self.x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        self.Sigma0 = tf.convert_to_tensor(Sigma0, dtype=tf.float32)

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Ensure x0 is a column vector
        if len(self.x0.shape) == 1:
            self.x0 = tf.reshape(self.x0, [-1, 1])

        # Infer dimensions
        self.state_dim = self.x0.shape[0]
        self.obs_dim = self.R.shape[0]

        # Compute scaling parameters
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.gamma = tf.sqrt(self.state_dim + self.lambda_)

        # Compute weights for mean and covariance
        self._compute_weights()

    def _compute_weights(self):
        """Compute weights for sigma points."""
        n = self.state_dim

        # Weights for mean
        self.Wm = tf.concat([
            [self.lambda_ / (n + self.lambda_)],
            tf.ones(2 * n, dtype=tf.float32) / (2 * (n + self.lambda_))
        ], axis=0)

        # Weights for covariance
        self.Wc = tf.concat([
            [self.lambda_ / (n + self.lambda_) + (1 - self.alpha**2 + self.beta)],
            tf.ones(2 * n, dtype=tf.float32) / (2 * (n + self.lambda_))
        ], axis=0)

    def generate_sigma_points(self, x, Sigma):
        """
        Generate sigma points using the unscented transform.

        For state dimension n, generates 2n+1 sigma points.

        Parameters:
        -----------
        x : tf.Tensor, shape (state_dim, 1)
            Mean state
        Sigma : tf.Tensor, shape (state_dim, state_dim)
            Covariance

        Returns:
        --------
        sigma_points : tf.Tensor, shape (state_dim, 2*state_dim+1)
            Sigma points (each column is a sigma point)
        """
        n = self.state_dim

        # Compute matrix square root using Cholesky decomposition
        # Sigma = L @ L^T
        L = tf.linalg.cholesky(Sigma)

        # Scale by gamma
        scaled_L = self.gamma * L

        # Generate sigma points
        # First sigma point: mean
        sigma_points = [x]

        # Next n sigma points: mean + scaled columns of L
        for i in range(n):
            sigma_points.append(x + scaled_L[:, i:i+1])

        # Last n sigma points: mean - scaled columns of L
        for i in range(n):
            sigma_points.append(x - scaled_L[:, i:i+1])

        # Stack into matrix: (state_dim, 2n+1)
        sigma_points = tf.concat(sigma_points, axis=1)

        return sigma_points

    def unscented_transform_state(self, sigma_points, u=None):
        """
        Propagate sigma points through state transition function.

        Parameters:
        -----------
        sigma_points : tf.Tensor, shape (state_dim, 2*state_dim+1)
            Input sigma points
        u : tf.Tensor, optional, shape (control_dim, 1)
            Control input

        Returns:
        --------
        transformed_points : tf.Tensor, shape (state_dim, 2*state_dim+1)
            Transformed sigma points
        """
        n_sigma = 2 * self.state_dim + 1
        transformed = []

        for i in range(n_sigma):
            sigma_i = sigma_points[:, i:i+1]
            if u is not None:
                f_sigma_i = self.f(sigma_i, u)
            else:
                f_sigma_i = self.f(sigma_i)
            transformed.append(f_sigma_i)

        return tf.concat(transformed, axis=1)

    def unscented_transform_observation(self, sigma_points):
        """
        Propagate sigma points through observation function.

        Parameters:
        -----------
        sigma_points : tf.Tensor, shape (state_dim, 2*state_dim+1)
            Input sigma points

        Returns:
        --------
        transformed_points : tf.Tensor, shape (obs_dim, 2*state_dim+1)
            Transformed sigma points
        """
        n_sigma = 2 * self.state_dim + 1
        transformed = []

        for i in range(n_sigma):
            sigma_i = sigma_points[:, i:i+1]
            h_sigma_i = self.h(sigma_i)
            transformed.append(h_sigma_i)

        return tf.concat(transformed, axis=1)

    def compute_mean_and_covariance(self, sigma_points, weights_mean, weights_cov, noise_cov):
        """
        Compute mean and covariance from sigma points.

        Parameters:
        -----------
        sigma_points : tf.Tensor, shape (dim, 2*state_dim+1)
            Sigma points
        weights_mean : tf.Tensor, shape (2*state_dim+1,)
            Weights for computing mean
        weights_cov : tf.Tensor, shape (2*state_dim+1,)
            Weights for computing covariance
        noise_cov : tf.Tensor, shape (dim, dim)
            Additional noise covariance to add

        Returns:
        --------
        mean : tf.Tensor, shape (dim, 1)
            Weighted mean
        cov : tf.Tensor, shape (dim, dim)
            Weighted covariance
        """
        # Compute mean: weighted sum of sigma points
        mean = tf.reduce_sum(
            sigma_points * weights_mean[tf.newaxis, :],
            axis=1, keepdims=True
        )

        # Compute covariance using vectorized operations
        diff = sigma_points - mean  # (dim, 2n+1)
        # Weighted covariance: sum_i w_i * diff_i @ diff_i^T
        weighted_diff = diff * weights_cov[tf.newaxis, :]  # (dim, 2n+1)
        cov = tf.linalg.matmul(weighted_diff, diff, transpose_b=True) + noise_cov

        return mean, cov

    @tf.function
    def predict(self, x, Sigma_post, u=None):
        """
        Prediction step using unscented transform.

        1. Generate sigma points around x_{t-1|t-1}
        2. Propagate through f()
        3. Compute predicted mean and covariance

        Parameters:
        -----------
        x : tf.Tensor, shape (state_dim, 1)
            Previous state x_{t-1|t-1}
        Sigma_post : tf.Tensor, shape (state_dim, state_dim)
            Previous covariance Σ_{t-1|t-1}
        u : tf.Tensor, optional, shape (control_dim, 1)
            Control input u_t

        Returns:
        --------
        x_pred : tf.Tensor
            Predicted state x_{t|t-1}
        Sigma_pred : tf.Tensor
            Predicted covariance Σ_{t|t-1}
        sigma_points_pred : tf.Tensor
            Predicted sigma points (for update step)
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(x, Sigma_post)

        # Propagate through state transition
        sigma_points_pred = self.unscented_transform_state(sigma_points, u)

        # Compute predicted mean and covariance
        x_pred, Sigma_pred = self.compute_mean_and_covariance(
            sigma_points_pred, self.Wm, self.Wc, self.Q
        )

        return x_pred, Sigma_pred, sigma_points_pred

    @tf.function
    def update(self, z, x_pred, Sigma_pred, sigma_points_pred):
        """
        Update step using unscented transform.

        1. Propagate predicted sigma points through h()
        2. Compute predicted observation and innovation covariance
        3. Compute cross-covariance
        4. Compute Kalman gain and update

        Parameters:
        -----------
        z : tf.Tensor, shape (obs_dim, 1)
            Observation z_t
        x_pred : tf.Tensor, shape (state_dim, 1)
            Predicted state x_{t|t-1}
        Sigma_pred : tf.Tensor, shape (state_dim, state_dim)
            Predicted covariance Σ_{t|t-1}
        sigma_points_pred : tf.Tensor, shape (state_dim, 2*state_dim+1)
            Predicted sigma points

        Returns:
        --------
        x : tf.Tensor
            Updated state x_{t|t}
        Sigma_post : tf.Tensor
            Updated covariance Σ_{t|t}
        """
        # Propagate sigma points through observation function
        sigma_points_obs = self.unscented_transform_observation(sigma_points_pred)

        # Compute predicted observation mean and covariance
        z_pred, S = self.compute_mean_and_covariance(
            sigma_points_obs, self.Wm, self.Wc, self.R
        )

        # Compute cross-covariance between state and observation using vectorized operations
        diff_x = sigma_points_pred - x_pred  # (state_dim, 2n+1)
        diff_z = sigma_points_obs - z_pred   # (obs_dim, 2n+1)

        # Sigma_xz (t|t-1) = sum_i w_i * diff_x_i @ diff_z_i^T = diff_x @ diag(Wc) @ diff_z^T
        weighted_diff_x = diff_x * self.Wc[None, :]  # (state_dim, 2n+1)
        Sigma_xz = tf.linalg.matmul(weighted_diff_x, diff_z, transpose_b=True)  # (state_dim, obs_dim)

        # Kalman gain: K = Sigma_xz @ S^{-1}
        K = tf.linalg.matmul(Sigma_xz, tf.linalg.inv(S))

        # Update state
        innovation = z - z_pred
        x = x_pred + tf.linalg.matmul(K, innovation)

        # Update covariance
        Sigma_post = Sigma_pred - tf.linalg.matmul(tf.linalg.matmul(K, S), K, transpose_b=True)

        return x, Sigma_post

    @tf.function
    def filter(self, observations, controls=None):
        """
        Run Unscented Kalman filter on observation sequence.

        Parameters:
        -----------
        observations : tf.Tensor, shape (obs_dim, T)
            Observation sequence z_{1:T} (each column is an observation)
        controls : tf.Tensor, optional, shape (control_dim, T)
            Control input sequence u_{1:T} (each column is a control)

        Returns:
        --------
        filtered_states : tf.Tensor, shape (state_dim, T+1)
            Filtered states x_{t|t} (includes initial state at t=0)
        predicted_states : tf.Tensor, shape (state_dim, T)
            Predicted states x_{t|t-1} (for t=1:T)
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        T = tf.shape(observations)[1]

        # Initialize
        x = self.x0
        Sigma_post = self.Sigma0

        # Storage using TensorArray
        filtered_states_array = tf.TensorArray(dtype=tf.float32, size=T+1, dynamic_size=False)
        predicted_states_array = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)

        # Store initial filtered state
        filtered_states_array = filtered_states_array.write(0, tf.squeeze(self.x0, axis=1))

        for t in tf.range(T):
            u = tf.reshape(controls[:, t], [-1, 1]) if controls is not None else None
            z = tf.reshape(observations[:, t], [-1, 1])

            # Prediction
            x_pred, Sigma_pred, sigma_points_pred = self.predict(x, Sigma_post, u)
            predicted_states_array = predicted_states_array.write(t, tf.squeeze(x_pred, axis=1))

            # Update
            x, Sigma_post = self.update(z, x_pred, Sigma_pred, sigma_points_pred)
            filtered_states_array = filtered_states_array.write(t+1, tf.squeeze(x, axis=1))

        # Stack and transpose
        filtered_states = tf.transpose(filtered_states_array.stack())
        predicted_states = tf.transpose(predicted_states_array.stack())

        return filtered_states, predicted_states
