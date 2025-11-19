"""
Kalman Filter for Linear Gaussian State Space Models

This module provides a simple Kalman Filter implementation following the formulas in kf.md.

Reference:
Pei Y, Biswas S, Fussell D S, et al. An elementary introduction to Kalman filtering[J].
Communications of the ACM, 2019, 62(11): 122-133.
"""

import tensorflow as tf


class KalmanFilter:
    """
    Kalman Filter for Linear Gaussian State Space Models.

    State equation: x_t = F_t @ x_{t-1} + B_t @ u_t + w_t, w_t ~ N(0, Q_t)
    Observation equation: z_t = H_t @ x_t + v_t, v_t ~ N(0, R_t)

    Parameters:
    -----------
    F : tf.Tensor, shape (state_dim, state_dim)
        State transition matrix
    H : tf.Tensor, shape (obs_dim, state_dim)
        Observation matrix
    Q : tf.Tensor, shape (state_dim, state_dim)
        Process noise covariance
    R : tf.Tensor, shape (obs_dim, obs_dim)
        Observation noise covariance
    x0 : tf.Tensor, shape (state_dim, 1)
        Initial state mean (column vector)
    Sigma0 : tf.Tensor, shape (state_dim, state_dim)
        Initial state covariance
    B : tf.Tensor, optional, shape (state_dim, control_dim)
        Control input matrix
    use_joseph_form : bool, optional (default: True)
        If True, use Joseph stabilized covariance update for numerical stability.
        If False, use standard covariance update form.
    """

    def __init__(self, F, H, Q, R, x0, Sigma0, B=None, use_joseph_form=True):
        self.F = tf.convert_to_tensor(F, dtype=tf.float32)
        self.H = tf.convert_to_tensor(H, dtype=tf.float32)
        self.Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        self.R = tf.convert_to_tensor(R, dtype=tf.float32)
        self.x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        self.Sigma0 = tf.convert_to_tensor(Sigma0, dtype=tf.float32)
        self.B = tf.convert_to_tensor(B, dtype=tf.float32) if B is not None else None
        self.use_joseph_form = use_joseph_form

        # Ensure x0 is a column vector
        if len(self.x0.shape) == 1:
            self.x0 = tf.reshape(self.x0, [-1, 1])

        # Infer dimensions
        self.state_dim = self.x0.shape[0]
        self.obs_dim = self.H.shape[0]

    @tf.function
    def predict(self, x, Sigma_post, u=None):
        """
        Prediction step.

        x_{t|t-1} = F_t @ x_{t-1|t-1} + B_t @ u_t
        Σ_{t|t-1} = F_t @ Σ_{t-1|t-1} @ F_t^T + Q_t

        Parameters:
        -----------
        x : tf.Tensor, shape (state_dim, 1)
            Previous state x_{t-1|t-1} (column vector)
        Sigma_post : tf.Tensor, shape (state_dim, state_dim)
            Previous covariance Σ_{t-1|t-1}
        u : tf.Tensor, optional, shape (control_dim, 1)
            Control input u_t (column vector)

        Returns:
        --------
        x_pred : tf.Tensor
            Predicted state x_{t|t-1}
        Sigma_pred : tf.Tensor
            Predicted covariance Σ_{t|t-1}
        """
        # Predicted state
        x_pred = tf.linalg.matmul(self.F, x)
        if self.B is not None and u is not None:
            x_pred = x_pred + tf.linalg.matmul(self.B, u)

        # Predicted covariance
        Sigma_pred = tf.linalg.matmul(tf.linalg.matmul(self.F, Sigma_post), self.F, transpose_b=True) + self.Q

        return x_pred, Sigma_pred

    @tf.function
    def update(self, z, x_pred, Sigma_pred):
        """
        Update step.

        Posterior mean and Kalman gain:
        r_t = z_t - H_t @ x_{t|t-1}
        S_t = H_t @ Σ_{t|t-1} @ H_t^T + R_t
        K_t = Σ_{t|t-1} @ H_t^T @ S_t^{-1}
        x_{t|t} = x_{t|t-1} + K_t @ r_t

        Posterior covariance (two forms available):

        Standard form (kf.md):
        Σ_{t|t} = (I - K_t @ H_t) @ Σ_{t|t-1}

        Joseph stabilized form (recommended for numerical stability):
        Σ_{t|t} = (I - K_t @ H_t) @ Σ_{t|t-1} @ (I - K_t @ H_t)^T + K_t @ R_t @ K_t^T

        Parameters:
        -----------
        z : tf.Tensor, shape (obs_dim, 1)
            Observation z_t (column vector)
        x_pred : tf.Tensor, shape (state_dim, 1)
            Predicted state x_{t|t-1} (column vector)
        Sigma_pred : tf.Tensor, shape (state_dim, state_dim)
            Predicted covariance Σ_{t|t-1}

        Returns:
        --------
        x : tf.Tensor
            Updated state x_{t|t}
        Sigma_post : tf.Tensor
            Updated covariance Σ_{t|t}
        """
        # Innovation: r_t = z_t - H_t @ x_{t|t-1}
        r = z - tf.linalg.matmul(self.H, x_pred)

        # Innovation covariance: S_t = H_t @ Σ_{t|t-1} @ H_t^T + R_t
        S = tf.linalg.matmul(tf.linalg.matmul(self.H, Sigma_pred), self.H, transpose_b=True) + self.R

        # Kalman gain: K_t = Σ_{t|t-1} @ H_t^T @ S_t^{-1}
        K = tf.linalg.matmul(
            tf.linalg.matmul(Sigma_pred, self.H, transpose_b=True),
            tf.linalg.inv(S)
        )

        # Posterior state: x_{t|t} = x_{t|t-1} + K_t @ r_t
        x = x_pred + tf.linalg.matmul(K, r)

        # Posterior covariance
        I_KH = tf.eye(self.state_dim, dtype=tf.float32) - tf.linalg.matmul(K, self.H)

        if self.use_joseph_form:
            # Joseph stabilized form (guarantees numerical stability and symmetry)
            # Σ_{t|t} = (I - K_t @ H_t) @ Σ_{t|t-1} @ (I - K_t @ H_t)^T + K_t @ R_t @ K_t^T
            Sigma_post = (
                tf.linalg.matmul(tf.linalg.matmul(I_KH, Sigma_pred), I_KH, transpose_b=True) +
                tf.linalg.matmul(tf.linalg.matmul(K, self.R), K, transpose_b=True)
            )
        else:
            # Standard form (simpler but may lose numerical stability)
            # Σ_{t|t} = (I - K_t @ H_t) @ Σ_{t|t-1}
            Sigma_post = tf.linalg.matmul(I_KH, Sigma_pred)

        return x, Sigma_post

    @tf.function
    def filter(self, observations, controls=None):
        """
        Run Kalman filter on observation sequence.

        Parameters:
        -----------
        observations : tf.Tensor, shape (obs_dim, T)
            Observation sequence z_{1:T} (each column is an observation)
        controls : tf.Tensor, optional, shape (control_dim, T)
            Control input sequence u_{1:T} (each column is a control)

        Returns:
        --------
        filtered_states : tf.Tensor, shape (state_dim, T+1)
            Filtered states x_{t|t} (each column is a state, includes initial state at t=0)
        predicted_states : tf.Tensor, shape (state_dim, T)
            Predicted states x_{t|t-1} (each column is a state, for t=1:T)
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        T = tf.shape(observations)[1]

        # Initialize
        x = self.x0
        Sigma_post = self.Sigma0

        # Storage using TensorArray for dynamic graph compilation
        # We'll store T+1 filtered states (including initial) and T predicted states
        filtered_states_array = tf.TensorArray(dtype=tf.float32, size=T+1, dynamic_size=False)
        predicted_states_array = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)

        # Store initial filtered state
        filtered_states_array = filtered_states_array.write(0, tf.squeeze(self.x0, axis=1))

        for t in tf.range(T):
            # we change the dimension here
            u = tf.reshape(controls[:, t], [-1, 1]) if controls is not None else None
            z = tf.reshape(observations[:, t], [-1, 1])

            # Prediction
            x_pred, Sigma_pred = self.predict(x, Sigma_post, u)
            predicted_states_array = predicted_states_array.write(t, tf.squeeze(x_pred, axis=1))

            # Update
            x, Sigma_post = self.update(z, x_pred, Sigma_pred)
            filtered_states_array = filtered_states_array.write(t+1, tf.squeeze(x, axis=1))
        
        # Stack and transpose to get (state_dim, T+1) and (state_dim, T)
        filtered_states = tf.transpose(filtered_states_array.stack())  # (T+1, state_dim) -> (state_dim, T+1)
        predicted_states = tf.transpose(predicted_states_array.stack())  # (T, state_dim) -> (state_dim, T)

        return filtered_states, predicted_states
