# Kalman Filter Formulas

## System Model

### State Space Representation

**State equation:**

```math
\mathbf{x}_t = F_t \mathbf{x}_{t-1} + B_t \mathbf{u}_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q_t)
```

**Observation equation:**

```math
\mathbf{z}_t = H_t \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, R_t)
```

Where:
- $\mathbf{x}_t \in \mathbb{R}^{n_x}$ is the latent (hidden) state vector
- $\mathbf{z}_t \in \mathbb{R}^{n_z}$ is the observation vector
- $\mathbf{u}_t$ is an optional control input
- $\mathbf{w}_t, \mathbf{v}_t$ are mutually independent Gaussian noise terms
- $F_t, B_t, H_t$ are known model matrices
- $Q_t$ and $R_t$ are the process and observation covariance matrices

---

## Kalman Filter Algorithm

### 1. Initialization

```math
\hat{\mathbf{x}}_{0|0} = \mathbb{E}[\mathbf{x}_0], \qquad \Sigma_{0|0} = \mathrm{Cov}[\mathbf{x}_0]
```

### 2. Prediction Step

**Predicted state estimate:**

```math
\hat{\mathbf{x}}_{t|t-1} = F_t \hat{\mathbf{x}}_{t-1|t-1} + B_t \mathbf{u}_t
```

**Predicted error covariance:**

```math
\Sigma_{t|t-1} = F_t \Sigma_{t-1|t-1} F_t^\top + Q_t
```

### 3. Update Step

**Innovation (residual):**

```math
\mathbf{r}_t = \mathbf{z}_t - H_t \hat{\mathbf{x}}_{t|t-1}
```

**Innovation covariance:**

```math
S_t = H_t \Sigma_{t|t-1} H_t^\top + R_t
```

**Kalman gain:**

```math
K_t = \Sigma_{t|t-1} H_t^\top S_t^{-1}
```

**Posterior state estimate:**

```math
\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_t \mathbf{r}_t
```

**Posterior error covariance:**

```math
\Sigma_{t|t}  = (I - K_t H_t)\Sigma_{t|t-1} \text{ or } \Sigma_{t|t} = (I - K_t H_t)\Sigma_{t|t-1}(I - K_t H_t)^\top + K_t R_t K_t^\top \text{(Joseph stabilized form)}
```

---
The notation ${t \mid t-1}$ means “at time $t$ given information up to time $t-1$”. Thus, $\hat{\mathbf{x}}_{t|t-1}$ is the \emph{predicted} state at time $t$ before observing $\mathbf{z}_t$,  while $\hat{\mathbf{x}}_{t|t}$ is the \emph{updated} (filtered) state after incorporating $\mathbf{z}_t$. In our implementation, we replace $F_t, H_t, Q_t, R_t, B_t$ with $F, H, Q, R, B$, assuming they are time-invariant.

## Notes

- The **Joseph stabilized covariance update** guarantees numerical stability and symmetry of $\Sigma_{t|t}$
- The Kalman filter provides recursive minimum-variance estimates of $\mathbf{x}_t$ given observations $\mathbf{z}_{1:t}$
- This filter is optimal for linear-Gaussian state-space models

## Reference

Pei Y, Biswas S, Fussell D S, et al. An elementary introduction to Kalman filtering[J]. Communications of the ACM, 2019, 62(11): 122-133.
