"""
Shared constants for particle filter.
"""

import tensorflow as tf

DEFAULT_DTYPE = tf.float64      # Use float64 for numerical precision
DEFAULT_ESS_THRESHOLD = 0.5     # Resample when ESS < threshold * N
DEFAULT_SEED = 0                # Default random seed for reproducibility
