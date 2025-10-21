"""
Range-Bearing Non-Linear State-Space Model
Based on Extended Kalman Filter tutorial (CMU 16-385)

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

        r: range (distance from origin)
        θ: bearing (angle from x-axis)

This model is non-linear because:
    - Range involves square root: r = sqrt(x^2 + y^2)
    - Bearing involves arctangent: θ = arctan2(y, x)

Although the process and measurement noises are Gaussian,the posterior distribution 
becomes non-Gaussian due to the non-linear observation function.
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

        # Measurement noise covariance R
        self.R = tf.constant([
            [range_noise_std**2, 0.0],
            [0.0, bearing_noise_std**2]
        ], dtype=tf.float32)

        self.range_noise_std = range_noise_std
        self.bearing_noise_std = bearing_noise_std

        if seed is not None:
            tf.random.set_seed(seed)

    def sample_initial_state(self, num_samples=1, initial_pos_std=100.0, initial_vel_std=10.0):
        """
        Sample initial state x_0 ~ N(0, P_0)

        Args:
            num_samples: Number of independent samples/trajectories
            initial_pos_std: Standard deviation for initial position
            initial_vel_std: Standard deviation for initial velocity

        Returns:
            Initial state tensor of shape (num_samples, 4)
        """
        # Position
        x = tf.random.normal((num_samples, 1), mean=0.0, stddev=initial_pos_std)
        y = tf.random.normal((num_samples, 1), mean=0.0, stddev=initial_pos_std)

        # Velocity
        x_dot = tf.random.normal((num_samples, 1), mean=0.0, stddev=initial_vel_std)
        y_dot = tf.random.normal((num_samples, 1), mean=0.0, stddev=initial_vel_std)

        return tf.concat([x, x_dot, y, y_dot], axis=1)

    def state_transition(self, x_prev):
        """
        State transition: x_t = A * x_{t-1} + w_t

        This is a LINEAR motion model (constant velocity).

        Args:
            x_prev: Previous state, shape (num_samples, 4)

        Returns:
            Next state x_t, shape (num_samples, 4)
        """
        num_samples = tf.shape(x_prev)[0]

        # Linear state transition
        x_next = tf.linalg.matvec(self.A, x_prev)

        # Add Gaussian process noise
        process_noise = tf.random.normal(
            shape=(num_samples, 4),
            mean=0.0,
            stddev=1.0
        )

        # Scale noise by Q
        Q_chol = tf.linalg.cholesky(self.Q + 1e-6 * tf.eye(4))
        scaled_noise = tf.linalg.matvec(Q_chol, process_noise)

        x_next = x_next + scaled_noise

        return x_next

    def observation_model(self, x):
        """
        NON-LINEAR observation model: z_t = h(x_t) + v_t

        h(x) = [r  ] = [sqrt(x^2 + y^2)  ]
               [θ  ]   [arctan2(y, x)    ]

        Args:
            x: Current state, shape (num_samples, 4) [x, x_dot, y, y_dot]

        Returns:
            Observation z, shape (num_samples, 2) [range, bearing]
        """
        num_samples = tf.shape(x)[0]

        # Extract position
        x_pos = x[:, 0]  # x position
        y_pos = x[:, 2]  # y position

        # Compute range: r = sqrt(x^2 + y^2)
        range_true = tf.sqrt(x_pos**2 + y_pos**2 + 1e-8)  # Add small constant for stability

        # Compute bearing: θ = arctan2(y, x)
        bearing_true = tf.atan2(y_pos, x_pos)

        # Add measurement noise
        range_noise = tf.random.normal(
            shape=(num_samples,),
            mean=0.0,
            stddev=self.range_noise_std
        )

        bearing_noise = tf.random.normal(
            shape=(num_samples,),
            mean=0.0,
            stddev=self.bearing_noise_std
        )

        range_measured = range_true + range_noise
        bearing_measured = bearing_true + bearing_noise

        # Stack into observation vector
        z = tf.stack([range_measured, bearing_measured], axis=1)

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
            x: State, shape (num_samples, 4)

        Returns:
            Jacobian H, shape (num_samples, 2, 4)
        """
        num_samples = tf.shape(x)[0]

        # Extract position
        x_pos = x[:, 0]
        y_pos = x[:, 2]

        # Compute range and bearing
        r = tf.sqrt(x_pos**2 + y_pos**2 + 1e-8)
        theta = tf.atan2(y_pos, x_pos)

        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)

        # Build Jacobian matrix
        # H = [[cos(θ),      0,  sin(θ),     0],
        #      [-sin(θ)/r,   0,  cos(θ)/r,   0]]

        H = tf.stack([
            tf.stack([cos_theta, tf.zeros_like(cos_theta), sin_theta, tf.zeros_like(sin_theta)], axis=1),
            tf.stack([-sin_theta/r, tf.zeros_like(sin_theta), cos_theta/r, tf.zeros_like(cos_theta)], axis=1)
        ], axis=1)

        return H

    def simulate_trajectory(self, T, num_samples=1,
                          initial_pos_std=100.0, initial_vel_std=10.0):
        """
        Simulate a complete trajectory of states and observations.

        Following the convention from lgss_sample.py:
        - States: x_0, x_1, ..., x_T (T+1 states total)
        - Observations: z_1, z_2, ..., z_T (T observations total)

        The initial state x_0 has no corresponding observation.
        Each observation z_t is generated from state x_t.

        Args:
            T: Number of time steps (number of observations)
            num_samples: Number of independent trajectories
            initial_pos_std: Standard deviation for initial position
            initial_vel_std: Standard deviation for initial velocity

        Returns:
            states: Tensor of shape (4, T+1, num_samples) - includes x_0, x_1, ..., x_T
            observations: Tensor of shape (2, T, num_samples) - includes z_1, ..., z_T
        """
        # Sample initial state x_0
        x = self.sample_initial_state(num_samples, initial_pos_std, initial_vel_std)

        # TensorArrays to store results
        states_array = tf.TensorArray(dtype=tf.float32, size=T+1)
        observations_array = tf.TensorArray(dtype=tf.float32, size=T)

        # Store initial state x_0
        states_array = states_array.write(0, x)

        # Generate trajectory
        for t in range(T):
            # State transition: x_t = A * x_{t-1} + w_t
            x = self.state_transition(x)

            # Generate observation: z_t = h(x_t) + v_t
            z = self.observation_model(x)

            # Store x_t and z_t
            states_array = states_array.write(t + 1, x)
            observations_array = observations_array.write(t, z)

        # Stack results: (T+1, num_samples, 4) -> (4, T+1, num_samples)
        states = tf.transpose(states_array.stack(), [2, 0, 1])
        # Stack results: (T, num_samples, 2) -> (2, T, num_samples)
        observations = tf.transpose(observations_array.stack(), [2, 0, 1])

        return states, observations

    def cartesian_to_polar(self, x, y):
        """
        Convert Cartesian coordinates to polar (range-bearing).

        Args:
            x: x-coordinate
            y: y-coordinate

        Returns:
            range, bearing
        """
        r = tf.sqrt(x**2 + y**2)
        theta = tf.atan2(y, x)
        return r, theta

    def polar_to_cartesian(self, r, theta):
        """
        Convert polar coordinates to Cartesian.

        Args:
            r: range
            theta: bearing (radians)

        Returns:
            x, y coordinates
        """
        x = r * tf.cos(theta)
        y = r * tf.sin(theta)
        return x, y
