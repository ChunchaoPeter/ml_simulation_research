"""
Sampling functions for Linear Gaussian State Space Models for testing purposes
"""

import tensorflow as tf


def sample(F, H, Q, R, x0, P0, T, B=None, controls=None, seed=None):
    """
    Simulate data from a Linear Gaussian State Space Model.

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
    x0 : tf.Tensor, shape (state_dim, 1) or (state_dim,)
        Initial state mean (column vector or 1D vector)
    P0 : tf.Tensor, shape (state_dim, state_dim)
        Initial state covariance
    T : int
        Number of time steps
    B : tf.Tensor, optional, shape (state_dim, control_dim)
        Control input matrix
    controls : tf.Tensor, optional, shape (control_dim, T)
        Control inputs (each column is a control)
    seed : int, optional
        Random seed

    Returns:
    --------
    states : tf.Tensor, shape (state_dim, T+1)
        Simulated states x_{0:T} (each column is a state, includes initial state)
    observations : tf.Tensor, shape (obs_dim, T)
        Simulated observations z_{1:T} (each column is an observation)
    """
    F = tf.convert_to_tensor(F, dtype=tf.float32)
    H = tf.convert_to_tensor(H, dtype=tf.float32)
    Q = tf.convert_to_tensor(Q, dtype=tf.float32)
    R = tf.convert_to_tensor(R, dtype=tf.float32)
    x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
    P0 = tf.convert_to_tensor(P0, dtype=tf.float32)
    B = tf.convert_to_tensor(B, dtype=tf.float32) if B is not None else None

    # Ensure x0 is a column vector
    if len(x0.shape) == 1:
        x0 = tf.reshape(x0, [-1, 1])

    if seed is not None:
        tf.random.set_seed(seed)

    state_dim = x0.shape[0]
    obs_dim = H.shape[0]

    return _sample_impl(F, H, Q, R, x0, P0, T, B, controls, state_dim, obs_dim)


@tf.function
def _sample_impl(F, H, Q, R, x0, P0, T, B, controls, state_dim, obs_dim):
    """Internal implementation of sampling with tf.function."""
    # Sample initial state: x_0 ~ N(x0, P0)
    x = x0 + tf.matmul(
        tf.linalg.cholesky(P0),
        tf.random.normal([state_dim, 1], dtype=tf.float32)
    )


    # We'll store T+1 states (including initial) and T observations in order for @tf.function
    states_array = tf.TensorArray(dtype=tf.float32, size=T+1, dynamic_size=False)
    observations_array = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)

    # Store initial state by using 1-D tensor
    states_array = states_array.write(0, tf.squeeze(x, axis=1))

    for t in tf.range(T):
        # State transition: x_t = F @ x_{t-1} + B @ u_t + w_t, w_t ~ N(0, Q)
        w = tf.matmul(
            tf.linalg.cholesky(Q),
            tf.random.normal([state_dim, 1], dtype=tf.float32)
        )
        x = tf.matmul(F, x) + w

        if B is not None and controls is not None:
            ## controls is u_t
            u = tf.reshape(controls[:, t], [-1, 1])
            x = x + tf.matmul(B, u)

        # Observation: z_t = H @ x_t + v_t, v_t ~ N(0, R)
        v = tf.matmul(
            tf.linalg.cholesky(R),
            tf.random.normal([obs_dim, 1], dtype=tf.float32)
        )
        z = tf.matmul(H, x) + v

        states_array = states_array.write(t+1, tf.squeeze(x, axis=1))
        observations_array = observations_array.write(t, tf.squeeze(z, axis=1))

    # Stack and transpose to get (state_dim, T+1) and (obs_dim, T)
    states = tf.transpose(states_array.stack())  # (T+1, state_dim) -> (state_dim, T+1)
    observations = tf.transpose(observations_array.stack())  # (T, obs_dim) -> (obs_dim, T)

    return states, observations
