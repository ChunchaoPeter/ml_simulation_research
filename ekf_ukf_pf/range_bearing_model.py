"""
Range-Bearing Non-Linear State-Space Model
Based on Extended Kalman Filter tutorial (CMU Lecture note 16-385)

This is a non-linear and non-Gaussian State-Space Model implemented in TensorFlow.

State-Space Model:
    State: x = [x, x_dot, y, y_dot]^T (2D position and velocity)

    Motion Model (Linear - Constant Velocity):
        x_t = A * x_{t-1} + w_t
        where w_t ~ N(0, Q) is process noise

        A = [[1, dt, 0,  0 ],
             [0, 1,  0,  0 ],
             [0, 0,  1,  dt],
             [0, 0,  0,  1 ]]

    Observation Model (Non-Linear - Range-Bearing):
        z_t = h(x_t) + v_t
        where v_t ~ N(0, R) is measurement noise

        h(x) = [r  ] = [sqrt(x^2 + y^2)    ]
               [θ  ]   [arctan2(y, x)      ]

        r: range (distance from origin, r >= 0)
        θ: bearing (angle from x-axis, θ ∈ [-π, π])

This model is non-linear because:
    - Range involves square root: r = sqrt(x^2 + y^2)
    - Bearing involves arctangent: θ = arctan2(y, x)
    - Bearing measurements are wrapped to [-π, π] after adding noise

The non-linearity makes the exact posterior non-Gaussian, even with Gaussian noise.
"""

import tensorflow as tf


class RangeBearingModel:
    """
    Range-Bearing Tracking Model for 2D motion.

    This model tracks an object moving in 2D space with constant velocity,
    but observations are in polar coordinates (range and bearing).
    """

    def __init__(self, dt=1.0, process_noise_std_pos=0.1, process_noise_std_vel=0.1,
                 range_noise_std=50.0, bearing_noise_std=0.005, seed=None):
        """
        Initialize the Range-Bearing model.

        Args:
            dt: Time step (delta t)
            process_noise_std_pos: Standard deviation of process noise for position
            process_noise_std_vel: Standard deviation of process noise for velocity
            range_noise_std: Standard deviation of range measurement noise
            bearing_noise_std: Standard deviation of bearing measurement noise (in radians)
            seed: Random seed for reproducibility
        """
        self.dt = tf.constant(dt, dtype=tf.float32)

        # State transition matrix (constant velocity model)
        self.A = tf.constant([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=tf.float32)

        # Process noise covariance Q (affects both position and velocity)
        q_pos = process_noise_std_pos ** 2
        q_vel = process_noise_std_vel ** 2
        self.Q = tf.constant([
            [q_pos, 0.0,   0.0,   0.0  ],
            [0.0,   q_vel, 0.0,   0.0  ],
            [0.0,   0.0,   q_pos, 0.0  ],
            [0.0,   0.0,   0.0,   q_vel]
        ], dtype=tf.float32)

        # Measurement noise covariance R (defined for model completeness; not used in this implementation)
        self.R = tf.constant([
            [range_noise_std**2, 0.0],
            [0.0, bearing_noise_std**2]
        ], dtype=tf.float32)

        self.range_noise_std = range_noise_std
        self.bearing_noise_std = bearing_noise_std

        if seed is not None:
            tf.random.set_seed(seed)

    def sample_initial_state(self, initial_pos_std=100.0, initial_vel_std=10.0):
        """
        Sample initial state x_0 ~ N(0, P_0)

        Args:
            initial_pos_std: Standard deviation for initial position
            initial_vel_std: Standard deviation for initial velocity

        Returns:
            Initial state tensor of shape (4, 1) - column vector [x, x_dot, y, y_dot]^T
        """
        # Position
        x = tf.random.normal((1,), mean=0.0, stddev=initial_pos_std)
        y = tf.random.normal((1,), mean=0.0, stddev=initial_pos_std)

        # Velocity
        x_dot = tf.random.normal((1,), mean=0.0, stddev=initial_vel_std)
        y_dot = tf.random.normal((1,), mean=0.0, stddev=initial_vel_std)

        # Stack as column vector (4, 1)
        return tf.concat([x, x_dot, y, y_dot], axis=0)[:, tf.newaxis]

    def state_transition(self, x_prev):
        """
        State transition: x_t = A * x_{t-1} + w_t

        This is a LINEAR motion model (constant velocity).

        Args:
            x_prev: Previous state, shape (4, 1) - column vector [x, x_dot, y, y_dot]^T

        Returns:
            Next state x_t, shape (4, 1) - column vector [x, x_dot, y, y_dot]^T
        """
        # Linear state transition: A @ x_prev
        x_next = tf.linalg.matmul(self.A, x_prev)

        # Add Gaussian process noise
        w = tf.linalg.matmul(
            tf.linalg.cholesky(self.Q),
            tf.random.normal([4, 1], dtype=tf.float32)
        )
        x_next = x_next + w

        return x_next

    def observation_model(self, x):
        """
        NON-LINEAR observation model: z_t = h(x_t) + v_t

        h(x) = [r  ] = [sqrt(x^2 + y^2)  ]
               [θ  ]   [arctan2(y, x)    ]

        Args:
            x: Current state, shape (4, 1) - column vector [x, x_dot, y, y_dot]^T

        Returns:
            Observation z, shape (2, 1) - column vector [range, bearing]^T
            Note: Bearing is wrapped to [-π, π] range after adding noise
        """
        # Extract position
        x_pos = x[0, 0]  # x position
        y_pos = x[2, 0]  # y position

        # Compute range: r = sqrt(x^2 + y^2)
        range_true = tf.sqrt(x_pos**2 + y_pos**2)  # Add small constant for stability

        # Compute bearing: θ = arctan2(y, x)
        bearing_true = tf.atan2(y_pos, x_pos)

        # Add measurement noise
        range_noise = tf.random.normal(
            shape=(),
            mean=0.0,
            stddev=self.range_noise_std
        )

        bearing_noise = tf.random.normal(
            shape=(),
            mean=0.0,
            stddev=self.bearing_noise_std
        )

        range_measured = range_true + range_noise
        bearing_measured = bearing_true + bearing_noise

        # Wrap bearing to [-π, π] range
        # After adding noise, bearing can go outside [-π, π]
        # Use atan2(sin(θ), cos(θ)) to wrap angle back to [-π, π]
        bearing_measured = tf.atan2(tf.sin(bearing_measured), tf.cos(bearing_measured))

        # Stack into observation column vector (2, 1)
        z = tf.stack([range_measured, bearing_measured])[:, tf.newaxis]

        return z

    def compute_observation_jacobian(self, x):
        """
        Compute the Jacobian matrix H of the observation model.

        H = ∂h/∂x = [[∂r/∂x,  ∂r/∂ẋ,  ∂r/∂y,  ∂r/∂ẏ ],
                      [∂θ/∂x,  ∂θ/∂ẋ,  ∂θ/∂y,  ∂θ/∂ẏ ]]

        For range r = sqrt(x^2 + y^2):
            ∂r/∂x = x/r = cos(θ)
            ∂r/∂y = y/r = sin(θ)
            ∂r/∂ẋ = 0
            ∂r/∂ẏ = 0

        For bearing θ = arctan2(y, x):
            ∂θ/∂x = -y/r^2 = -sin(θ)/r
            ∂θ/∂y = x/r^2 = cos(θ)/r
            ∂θ/∂ẋ = 0
            ∂θ/∂ẏ = 0

        Args:
            x: State, shape (4, 1) - column vector [x, x_dot, y, y_dot]^T

        Returns:
            Jacobian H, shape (2, 4)
        """
        # Extract position
        x_pos = x[0, 0]
        y_pos = x[2, 0]

        # Compute range and bearing
        r = tf.sqrt(x_pos**2 + y_pos**2)
        theta = tf.atan2(y_pos, x_pos)

        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)

        # Build Jacobian matrix
        # H = [[cos(θ),      0,  sin(θ),     0],
        #      [-sin(θ)/r,   0,  cos(θ)/r,   0]]

        H = tf.stack([
            tf.stack([cos_theta, 0.0, sin_theta, 0.0]),
            tf.stack([-sin_theta/r, 0.0, cos_theta/r, 0.0])
        ])

        return H

    def simulate_trajectory(self, T, initial_pos_std=100.0, initial_vel_std=10.0):
        """
        Simulate a complete trajectory of states and observations.

        Following the convention from lgss_sample.py:
        - States: x_0, x_1, ..., x_T (T+1 states total)
        - Observations: z_1, z_2, ..., z_T (T observations total)

        The initial state x_0 has no corresponding observation.
        Each observation z_t is generated from state x_t.

        Args:
            T: Number of time steps (number of observations)
            initial_pos_std: Standard deviation for initial position
            initial_vel_std: Standard deviation for initial velocity

        Returns:
            states: Tensor of shape (state_dim, T+1) = (4, T+1) - includes x_0, x_1, ..., x_T
            observations: Tensor of shape (obs_dim, T) = (2, T) - includes z_1, ..., z_T
        """
        # Sample initial state x_0 (shape: (4, 1) - column vector)
        x = self.sample_initial_state(initial_pos_std, initial_vel_std)

        # TensorArrays to store results
        states_array = tf.TensorArray(dtype=tf.float32, size=T+1)
        observations_array = tf.TensorArray(dtype=tf.float32, size=T)

        # Store initial state x_0, squeeze to 1D vector (4,)
        states_array = states_array.write(0, tf.squeeze(x, axis=1))

        # Generate trajectory
        for t in range(T):
            # State transition: x_t = A * x_{t-1} + w_t
            x = self.state_transition(x)

            # Generate observation: z_t = h(x_t) + v_t
            z = self.observation_model(x)

            # Store x_t and z_t, squeeze column vectors to 1D
            states_array = states_array.write(t + 1, tf.squeeze(x, axis=1))
            observations_array = observations_array.write(t, tf.squeeze(z, axis=1))

        # Stack results: (T+1, state_dim) -> (state_dim, T+1)
        states = tf.transpose(states_array.stack())
        # Stack results: (T, obs_dim) -> (obs_dim, T)
        observations = tf.transpose(observations_array.stack())

        return states, observations
