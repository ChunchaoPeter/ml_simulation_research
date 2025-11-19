"""
Extended Kalman Filter for Non-Linear Gaussian State Space Models

This module provides an Extended Kalman Filter implementation

The EKF linearizes non-linear functions using first-order Taylor expansion.
"""

import tensorflow as tf

# tf.config.run_functions_eagerly(True)  # turn off graph mode

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for Non-Linear Gaussian State Space Models.

    State equation: x_t = f(x_{t-1}, u_t) + w_t, w_t ~ N(0, Q_t)
    Observation equation: z_t = h(x_t) + v_t, v_t ~ N(0, R_t)

    The EKF linearizes these non-linear functions using Jacobians:
    - F_t = ∂f/∂x evaluated at x_{t-1|t-1}
    - H_t = ∂h/∂x evaluated at x_{t|t-1}

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
    state_transition_jacobian_fn : callable, required
        Jacobian of state transition ∂f/∂x
        Must be provided by user
    observation_jacobian_fn : callable, required
        Jacobian of observation ∂h/∂x
        Must be provided by user
    use_joseph_form : bool, optional (default: True)
        If True, use Joseph stabilized covariance update for numerical stability
    """

    def __init__(self, state_transition_fn, observation_fn, Q, R, x0, Sigma0,
                 state_transition_jacobian_fn, observation_jacobian_fn,
                 use_joseph_form=True):
        self.f = state_transition_fn
        self.h = observation_fn
        self.Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        self.R = tf.convert_to_tensor(R, dtype=tf.float32)
        self.x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        self.Sigma0 = tf.convert_to_tensor(Sigma0, dtype=tf.float32)
        self.use_joseph_form = use_joseph_form

        # Check that Jacobian functions are provided
        if state_transition_jacobian_fn is None:
            raise ValueError(
                "state_transition_jacobian_fn is required and cannot be None. "
                "Please provide a function that computes the Jacobian F = ∂f/∂x."
            )
        if observation_jacobian_fn is None:
            raise ValueError(
                "observation_jacobian_fn is required and cannot be None. "
                "Please provide a function that computes the Jacobian H = ∂h/∂x."
            )

        # Jacobian functions (must be provided by user)
        self.F_jacobian_fn = state_transition_jacobian_fn
        self.H_jacobian_fn = observation_jacobian_fn

        # Ensure x0 is a column vector
        if len(self.x0.shape) == 1:
            self.x0 = tf.reshape(self.x0, [-1, 1])

        # Infer dimensions
        self.state_dim = self.x0.shape[0]
        self.obs_dim = self.R.shape[0]

    def compute_state_jacobian(self, x, u=None):
        """
        Compute Jacobian F_t = ∂f/∂x at state x using user-provided function.

        Parameters:
        -----------
        x : tf.Tensor, shape (state_dim, 1)
            State at which to compute Jacobian
        u : tf.Tensor, optional, shape (control_dim, 1)
            Control input (if applicable)

        Returns:
        --------
        F : tf.Tensor, shape (state_dim, state_dim)
            Jacobian matrix ∂f/∂x
        """
        try:
            F = self.F_jacobian_fn(x, u)
            # Verify shape
            if F.shape != (self.state_dim, self.state_dim):
                raise ValueError(
                    f"State Jacobian shape mismatch: expected ({self.state_dim}, {self.state_dim}), "
                    f"got {F.shape}. Check your state_transition_jacobian_fn implementation."
                )
            return F
        except Exception as e:
            raise RuntimeError(f"Error in state_transition_jacobian_fn: {e}")

    def compute_observation_jacobian(self, x):
        """
        Compute Jacobian H_t = ∂h/∂x at state x using user-provided function.

        Parameters:
        -----------
        x : tf.Tensor, shape (state_dim, 1)
            State at which to compute Jacobian

        Returns:
        --------
        H : tf.Tensor, shape (obs_dim, state_dim)
            Jacobian matrix ∂h/∂x
        """
        try:
            H = self.H_jacobian_fn(x)
            # Verify shape
            if H.shape != (self.obs_dim, self.state_dim):
                raise ValueError(
                    f"Observation Jacobian shape mismatch: expected ({self.obs_dim}, {self.state_dim}), "
                    f"got {H.shape}. Check your observation_jacobian_fn implementation."
                )
            return H
        except Exception as e:
            raise RuntimeError(f"Error in observation_jacobian_fn: {e}")

    @tf.function
    def predict(self, x, Sigma_post, u=None):
        """
        Prediction step.

        Linearize around current estimate:
        F_t = ∂f/∂x|_{x_{t-1|t-1}}

        Then:
        x_{t|t-1} = f(x_{t-1|t-1}, u_t)
        Σ_{t|t-1} = F_t @ Σ_{t-1|t-1} @ F_t^T + Q_t

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
        """
        # Compute Jacobian at current state
        F = self.compute_state_jacobian(x, u)

        # Predicted state (non-linear)
        if u is not None:
            x_pred = self.f(x, u)
        else:
            x_pred = self.f(x)

        # Predicted covariance (linearized)
        Sigma_pred = tf.linalg.matmul(tf.linalg.matmul(F, Sigma_post), F, transpose_b=True) + self.Q

        return x_pred, Sigma_pred

    @tf.function
    def update(self, z, x_pred, Sigma_pred):
        """
        Update step.

        Linearize around predicted state:
        H_t = ∂h/∂x|_{x_{t|t-1}}

        Then:
        r_t = z_t - h(x_{t|t-1})
        S_t = H_t @ Σ_{t|t-1} @ H_t^T + R_t
        K_t = Σ_{t|t-1} @ H_t^T @ S_t^{-1}
        x_{t|t} = x_{t|t-1} + K_t @ r_t
        Σ_{t|t} = (I - K_t @ H_t) @ Σ_{t|t-1}  [or Joseph form]

        Parameters:
        -----------
        z : tf.Tensor, shape (obs_dim, 1)
            Observation z_t
        x_pred : tf.Tensor, shape (state_dim, 1)
            Predicted state x_{t|t-1}
        Sigma_pred : tf.Tensor, shape (state_dim, state_dim)
            Predicted covariance Σ_{t|t-1}

        Returns:
        --------
        x : tf.Tensor
            Updated state x_{t|t}
        Sigma_post : tf.Tensor
            Updated covariance Σ_{t|t}
        """
        # Compute Jacobian at predicted state
        H = self.compute_observation_jacobian(x_pred)

        # Innovation: r_t = z_t - h(x_{t|t-1})
        r = z - self.h(x_pred)

        # Innovation covariance: S_t = H_t @ Σ_{t|t-1} @ H_t^T + R_t + eps (make it invertable)
        S = tf.linalg.matmul(tf.linalg.matmul(H, Sigma_pred), H, transpose_b=True) + self.R 

        # Kalman gain: K_t = Σ_{t|t-1} @ H_t^T @ S_t^{-1}
        K = tf.linalg.matmul(
            tf.linalg.matmul(Sigma_pred, H, transpose_b=True),
            tf.linalg.inv(S)
        )

        # Posterior state: x_{t|t} = x_{t|t-1} + K_t @ r_t
        x = x_pred + tf.linalg.matmul(K, r)

        # Posterior covariance
        I_KH = tf.eye(self.state_dim, dtype=tf.float32) - tf.linalg.matmul(K, H)

        if self.use_joseph_form:
            # Joseph stabilized form
            # Σ_{t|t} = (I - K_t @ H_t) @ Σ_{t|t-1} @ (I - K_t @ H_t)^T + K_t @ R_t @ K_t^T
            Sigma_post = (
                tf.linalg.matmul(tf.linalg.matmul(I_KH, Sigma_pred), I_KH, transpose_b=True) +
                tf.linalg.matmul(tf.linalg.matmul(K, self.R), K, transpose_b=True)
            )
        else:
            # Standard form
            # Σ_{t|t} = (I - K_t @ H_t) @ Σ_{t|t-1} 
            Sigma_post = tf.linalg.matmul(I_KH, Sigma_pred)

        return x, Sigma_post

    @tf.function
    def filter(self, observations, controls=None):
        """
        Run Extended Kalman filter on observation sequence.

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
            Predicted states x_{t|t-1} (each column is a state, for t=1:T)
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
