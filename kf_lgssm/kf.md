# Kalman Filter Formulas

## System Model

### State Space Representation

**State equation:**
$$\mathbf{x}_t = F_t \mathbf{x}_{t-1} + B_t \mathbf{u}_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q_t)$$

**Observation equation:**
$$\mathbf{z}_t = H_t \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, R_t)$$

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
$$\hat{\mathbf{x}}_{0|0} = \mathbb{E}[\mathbf{x}_0], \qquad \Sigma_{0|0} = \mathrm{Cov}[\mathbf{x}_0]$$

### 2. Prediction Step
**Predicted state estimate:**
$$\hat{\mathbf{x}}_{t|t-1} = F_t \hat{\mathbf{x}}_{t-1|t-1} + B_t \mathbf{u}_t$$

**Predicted error covariance:**
$$\Sigma_{t|t-1} = F_t \Sigma_{t-1|t-1} F_t^\top + Q_t$$

### 3. Update Step

**Innovation (residual):**
$$\mathbf{r}_t = \mathbf{z}_t - H_t \hat{\mathbf{x}}_{t|t-1}$$

**Innovation covariance:**
$$S_t = H_t \Sigma_{t|t-1} H_t^\top + R_t$$

**Kalman gain:**
$$K_t = \Sigma_{t|t-1} H_t^\top S_t^{-1}$$

**Posterior state estimate:**
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_t \mathbf{r}_t$$

**Posterior error covariance (Joseph stabilized form):**
$$\Sigma_{t|t} = (I - K_t H_t)\Sigma_{t|t-1}(I - K_t H_t)^\top + K_t R_t K_t^\top$$

---

## Log-Likelihood (Optional)

The log-likelihood of the observation sequence can be computed incrementally as:

$$\log p(\mathbf{z}_{1:T}) = -\frac{1}{2} \sum_{t=1}^{T} \left[ \log|S_t| + \mathbf{r}_t^\top S_t^{-1}\mathbf{r}_t + n_z \log(2\pi) \right]$$

---

## Notes

- The **Joseph stabilized covariance update** guarantees numerical stability and symmetry of $\Sigma_{t|t}$
- The Kalman filter provides recursive minimum-variance estimates of $\mathbf{x}_t$ given observations $\mathbf{z}_{1:t}$
- This filter is optimal for linear-Gaussian state-space models

## Reference
Pei Y, Biswas S, Fussell D S, et al. An elementary introduction to Kalman filtering[J]. Communications of the ACM, 2019, 62(11): 122-133.