"""
Sequential Monte Carlo (Standard Particle Filter) — Algorithm 1.

Given a state-space model:
    X_1 ~ mu(.)                          (initial distribution)
    X_{t+1} | X_t = x  ~  f(. | x)      (state transition)
    Y_t     | X_t = x  ~  g(. | x)      (observation)

The particle filter approximates the filtering distribution p(x_t | y_{1:t})
using a weighted set of N particles {X_t^i, w_t^i}_{i=1}^{N}. At each step:
    - Particles are propagated through a proposal q (here q = f).
    - Weights are updated using the observation likelihood g(y_t | x_t).
    - Resampling eliminates low-weight particles to combat weight degeneracy.

The filter also estimates the log marginal likelihood:
    log p(y_{1:T}) = sum_{t=1}^{T} log p(y_t | y_{1:t-1})
which is differentiable w.r.t. model parameters theta.

Implements the following algorithm (Corenflos et al. 2021):

    Standard Particle Filter
    ──────────────────────────────────────
     1: Sample X_1^i  ~iid  q_phi(. | y_1)              for i in [N]
     2: Compute omega_1^i = p_theta(X_1^i, y_1)
                            / q_phi(X_1^i | y_1)         for i in [N]
     3: ell(theta) <- (1/N) sum_{i=1}^{N} omega_1^i
     4: for t = 2, ..., T do
     5:     Normalize  w_{t-1}^i  propto  omega_{t-1}^i,
                       sum_{i=1}^{N} w_{t-1}^i = 1
     6:     Resample   X_tilde_{t-1}^i ~ sum_{i=1}^{N} w_{t-1}^i delta_{X_{t-1}^i}
                                                         for i in [N]
     7:     Sample     X_t^i ~ q_phi(. | X_tilde_{t-1}^i, y_t)
                                                         for i in [N]
     8:     Compute    omega_t^i = p_theta(X_t^i, y_t | X_tilde_{t-1}^i)
                                   / q_phi(X_t^i | X_tilde_{t-1}^i, y_t)
                                                         for i in [N]
     9:     Compute    p_hat_theta(y_t | y_{1:t-1}) = (1/N) sum_{i=1}^{N} omega_t^i
    10:     ell(theta) <- ell(theta) + log p_hat_theta(y_t | y_{1:t-1})
    11: end for
    12: Return: log-likelihood estimate ell(theta) = log p_hat_theta(y_{1:T})

With the bootstrap proposal q = f (transition prior), the importance
weights simplify to:  omega_t^i = g(y_t | X_t^i)  (observation likelihood).

Implementation maps:
    Step 5-6 -> resampling_criterion.apply() + resampling_method.apply()
    Step 7   -> proposal_model.propose()
    Step 8-9 -> observation_model.loglikelihood()
    Step 10  -> accumulated inside observation_model.loglikelihood()

Components are composed via dependency injection:
    - observation_model: p(y_t | x_t)       — e.g. LinearObservationModel
    - transition_model:  p(x_t | x_{t-1})   — e.g. LinearGaussianTransition
    - proposal_model:    q(x_t | x_{t-1}, y_t) — e.g. BootstrapProposal
    - resampling_criterion: when to resample — e.g. NeffCriterion
    - resampling_method: how to resample     — e.g. MultinomialResampler
"""

import attr
import tensorflow as tf
from dpf.base import State, StateSeries, Module
from dpf.observation.base import ObservationModelBase
from dpf.transition.base import TransitionModelBase
from dpf.proposal.base import ProposalModelBase
from dpf.resampling.base import ResamplerBase
from dpf.resampling.criterion import ResamplingCriterionBase

# TODO: extend StateSeries to also store ancestor_indices, t, and ess

class SMC(Module):
    """
    Sequential Monte Carlo (particle filter) orchestrator.

    Usage:
        # Build components
        obs_model = LinearObservationModel(H, noise)
        trans_model = RandomWalkModel(F, noise)
        proposal = BootstrapProposal(trans_model)
        criterion = NeffCriterion(threshold=0.5)
        resampler = MultinomialResampler()

        # Create filter
        smc = SMC(obs_model, trans_model, proposal, criterion, resampler)

        # Run filtering
        final_state = smc(initial_state, observations)
        # or
        final_state, state_series = smc(initial_state, observations,
                                         return_series=True)

    Attributes:
        observation_model: p(y_t | x_t).
        transition_model: p(x_t | x_{t-1}).
        proposal_model: q(x_t | x_{t-1}, y_t).
        resampling_criterion: When to resample.
        resampling_method: How to resample.
    """

    def __init__(
        self,
        observation_model: ObservationModelBase,
        transition_model: TransitionModelBase,
        proposal_model: ProposalModelBase,
        resampling_criterion: ResamplingCriterionBase,
        resampling_method: ResamplerBase,
        name: str = 'SMC',
    ):
        """
        Args:
            observation_model: Observation likelihood model.
            transition_model: State transition model.
            proposal_model: Proposal distribution.
            resampling_criterion: When-to-resample criterion.
            resampling_method: Resampling algorithm.
        """
        super().__init__(name=name)
        self._observation_model = observation_model
        self._transition_model = transition_model
        self._proposal_model = proposal_model
        self._resampling_criterion = resampling_criterion
        self._resampling_method = resampling_method

    def one_step(self, state: State, observation: tf.Tensor,
                 inputs: tf.Tensor = None, seed=None) -> State:
        """
        One filtering step (steps 5-10 for a single t).

        Step 5-6: Normalize weights, resample if ESS < threshold.
                  X_tilde_{t-1}^i ~ sum w_{t-1}^i delta_{X_{t-1}^i}
        Step 7:   Propose new particles.
                  X_t^i ~ q_phi(. | X_tilde_{t-1}^i, y_t)
        Step 8:   Compute importance weights.
                  omega_t^i = p_theta(X_t^i, y_t | X_tilde_{t-1}^i)
                              / q_phi(X_t^i | X_tilde_{t-1}^i, y_t)
        Step 9-10: Estimate marginal likelihood and accumulate.
                  ell(theta) <- ell(theta) + log p_hat(y_t | y_{1:t-1})

        Args:
            state: Current state at time t-1.
            observation: Observation y_t, shape [batch, obs_dim].
            inputs: Optional controls, shape [batch, input_dim].
            seed: Random seed.

        Returns:
            Updated state at time t.
        """
        # Steps 5-6: Normalize weights and resample (conditional on criterion)
        resample_flags = self._resampling_criterion.apply(state)
        state = self._resampling_method.apply(state, resample_flags, seed)

        # Step 7: Propose new particles from q(x_t | x_tilde_{t-1}, y_t)
        state = self._proposal_model.propose(state, inputs, observation, seed)

        # Steps 8-10: Compute importance weights, estimate marginal likelihood,
        #             and accumulate log-likelihood
        state = self._observation_model.loglikelihood(state, observation)

        return state

    def __call__(self, initial_state: State,
                 observations,
                 n_steps: int = None,
                 inputs=None,
                 return_final: bool = False,
                 return_series: bool = False):
        """
        Run the full filtering loop (Algorithm 1, steps 4-11).

        For t = 2, ..., T:
            Steps 5-6:  Normalize weights and resample
            Step 7:     Propose X_t^i ~ q_phi(. | X_tilde_{t-1}^i, y_t)
            Step 8:     Compute omega_t^i (importance weights)
            Step 9-10:  Accumulate ell(theta) += log p_hat(y_t | y_{1:t-1})

        Returns the accumulated log-likelihood (step 12):
            ell(theta) = log p_hat_theta(y_{1:T})

        Args:
            initial_state: Initial State with prior particles.
                particles: [batch, N, d] sampled from the prior p(x_1).
                log_weights: [batch, N] typically uniform log(1/N).
            observations: Tensor of observations, shape [T, batch, obs_dim].
                Time dimension MUST be first (axis 0).
            n_steps: Number of time steps. Inferred from observations if None.
            inputs: Optional tensor of inputs, shape [T, batch, input_dim].
            return_final: If True, return only the final state (default).
            return_series: If True, return (final_state, StateSeries).

        Returns:
            If return_series=True:
                Tuple (final_state, series_dict) where series_dict has keys
                'particles', 'log_weights', 'log_likelihoods' with time axis 0.
            Otherwise:
                final_state: State at time T with accumulated log_likelihoods.
        """
        # Determine number of time steps
        if n_steps is None:
            n_steps = observations.shape[0]

        state = initial_state

        # Initialize StateSeries if returning trajectory
        series = StateSeries.create(n_steps) if return_series else None

        # ---- Main filtering loop ----
        for t in range(n_steps):
            # Extract observation at time t
            obs_t = observations[t]  # [batch, obs_dim]

            # Extract inputs at time t (if provided)
            inputs_t = inputs[t] if inputs is not None else None

            # One filtering step: resample -> propose -> reweight
            state = self.one_step(state, obs_t, inputs_t)

            # Update time step
            state = attr.evolve(state, t=t + 1)

            # Record state in series (immutable: write returns new series)
            if series is not None:
                series = series.write(t, state)

        if return_series:
            return state, series.stack()
        return state

    def log_marginal_likelihood(self, initial_state: State,
                                 observations: tf.Tensor,
                                 inputs: tf.Tensor = None) -> tf.Tensor:
        """
        Compute the log marginal likelihood estimate (step 12).

        Runs the full particle filter and returns the accumulated log-likelihood:

            ell_hat(theta) = log p_hat(y_{1:T})
                           = sum_{t=1}^{T} log p_hat(y_t | y_{1:t-1})

        where p_hat(y_t | y_{1:t-1}) = (1/N) * sum_i omega_t^i is the
        particle filter estimate of the marginal likelihood at each step,
        and omega_t^i are the unnormalized importance weights.

        Args:
            initial_state: Initial State with prior particles.
            observations: Observation sequence, shape [T, batch, obs_dim].
            inputs: Optional inputs per step, shape [T, batch, input_dim].

        Returns:
            Log marginal likelihood estimate, shape [batch].
        """
        final_state = self(initial_state, observations, inputs=inputs)
        return final_state.log_likelihoods
