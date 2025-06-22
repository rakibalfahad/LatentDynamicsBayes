---
title: "Mathematical Foundations of HDP-HMM"
author: "Your Name"
date: "June 21, 2025"
---

# Mathematical Foundations of HDP-HMM

This document provides a detailed exposition of the mathematical foundations of the Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM). It's aimed at readers with a background in probability and statistics who want to understand the theoretical underpinnings of the model.

## Bayesian Nonparametrics

### The Dirichlet Process

The Dirichlet Process (DP) is a distribution over probability measures. For a measurable space $(\Theta, \mathcal{B})$, a base probability measure $H$ on $(\Theta, \mathcal{B})$, and a positive real concentration parameter $\alpha$, a random probability measure $G$ on $(\Theta, \mathcal{B})$ is distributed according to a Dirichlet process with parameters $\alpha$ and $H$, written $G \sim \text{DP}(\alpha, H)$, if for any finite measurable partition $(A_1, \ldots, A_r)$ of $\Theta$, the random vector $(G(A_1), \ldots, G(A_r))$ has a Dirichlet distribution with parameters $(\alpha H(A_1), \ldots, \alpha H(A_r))$:

$$(G(A_1), \ldots, G(A_r)) \sim \text{Dir}(\alpha H(A_1), \ldots, \alpha H(A_r))$$

The DP has several key properties:

1. **Discrete Realizations**: Although the base measure $H$ may be continuous, a draw $G \sim \text{DP}(\alpha, H)$ is almost surely discrete.

2. **Posterior Distribution**: Given observations $\theta_1, \ldots, \theta_n$ drawn from $G$, the posterior distribution of $G$ is another DP:

   $$G | \theta_1, \ldots, \theta_n \sim \text{DP}\left(\alpha + n, \frac{\alpha H + \sum_{i=1}^n \delta_{\theta_i}}{\alpha + n}\right)$$

   where $\delta_{\theta}$ is the Dirac measure centered at $\theta$.

3. **Predictive Distribution**: The predictive distribution of a new observation $\theta_{n+1}$ given $\theta_1, \ldots, \theta_n$ follows the Pólya urn scheme:

   $$\theta_{n+1} | \theta_1, \ldots, \theta_n \sim \frac{\alpha}{\alpha + n} H + \frac{1}{\alpha + n} \sum_{i=1}^n \delta_{\theta_i}$$

### Stick-Breaking Construction

The stick-breaking construction provides an explicit representation of draws from a DP. Let:

1. $v_k \sim \text{Beta}(1, \alpha)$ for $k = 1, 2, \ldots$
2. $\beta_k = v_k \prod_{i=1}^{k-1} (1 - v_i)$ for $k = 1, 2, \ldots$
3. $\phi_k \sim H$ independently for $k = 1, 2, \ldots$

Then:

$$G = \sum_{k=1}^{\infty} \beta_k \delta_{\phi_k} \sim \text{DP}(\alpha, H)$$

The weights $\beta_k$ can be visualized as breaking a stick of unit length: $v_k$ represents the proportion of the remaining stick that is broken off at step $k$. This construction is also known as the GEM (Griffiths, Engen, McCloskey) distribution.

## Hierarchical Dirichlet Process

The Hierarchical Dirichlet Process (HDP) extends the DP to model groups of data with shared mixture components. In an HDP:

1. A global measure $G_0 \sim \text{DP}(\gamma, H)$ is drawn from a DP.
2. For each group $j$, a local measure $G_j \sim \text{DP}(\alpha, G_0)$ is drawn from another DP, using $G_0$ as the base measure.

This hierarchical structure allows different groups to share the same set of atoms (the support points of $G_0$) while having different weights on these atoms.

Using the stick-breaking construction, we can represent the HDP as:

1. $G_0 = \sum_{k=1}^{\infty} \beta_k \delta_{\phi_k}$ where $\beta \sim \text{GEM}(\gamma)$ and $\phi_k \sim H$
2. $G_j = \sum_{k=1}^{\infty} \pi_{jk} \delta_{\phi_k}$ where $\pi_j \sim \text{DP}(\alpha, \beta)$

The local weights $\pi_{jk}$ can be constructed using another stick-breaking process:

1. $v_{jk} \sim \text{Beta}(\alpha \beta_k, \alpha (1 - \sum_{l=1}^k \beta_l))$
2. $\pi_{jk} = v_{jk} \prod_{l=1}^{k-1} (1 - v_{jl})$

This ensures that $\mathbb{E}[\pi_{jk}] = \beta_k$, meaning that on average, the local weights match the global weights.

## Hidden Markov Models

### Classical HMMs

A Hidden Markov Model (HMM) is a statistical model where the system being modeled is assumed to be a Markov process with unobservable (hidden) states. An HMM is characterized by:

1. A set of $N$ hidden states $\{1, 2, \ldots, N\}$
2. A transition matrix $A = \{a_{ij}\}$ where $a_{ij} = P(z_t = j | z_{t-1} = i)$
3. An emission model $\{b_i(o_t)\}$ where $b_i(o_t) = P(o_t | z_t = i)$
4. An initial state distribution $\pi = \{\pi_i\}$ where $\pi_i = P(z_1 = i)$

The joint distribution of a sequence of observations $o_{1:T}$ and hidden states $z_{1:T}$ is:

$$P(o_{1:T}, z_{1:T}) = P(z_1) P(o_1 | z_1) \prod_{t=2}^T P(z_t | z_{t-1}) P(o_t | z_t)$$

### Inference in HMMs

Inference in HMMs typically involves computing:

1. **Filtering**: $P(z_t | o_{1:t})$ - The probability of being in a certain state at time $t$ given observations up to time $t$.
2. **Smoothing**: $P(z_t | o_{1:T})$ - The probability of being in a certain state at time $t$ given all observations.
3. **Most likely state sequence**: $\arg\max_{z_{1:T}} P(z_{1:T} | o_{1:T})$ - The most probable sequence of hidden states given the observations.

The forward-backward algorithm is used for filtering and smoothing:

**Forward Recursion**:
Define $\alpha_t(i) = P(o_{1:t}, z_t = i)$, then:
- $\alpha_1(i) = \pi_i b_i(o_1)$
- $\alpha_{t+1}(j) = b_j(o_{t+1}) \sum_{i=1}^N \alpha_t(i) a_{ij}$ for $t = 1, \ldots, T-1$

**Backward Recursion**:
Define $\beta_t(i) = P(o_{t+1:T} | z_t = i)$, then:
- $\beta_T(i) = 1$
- $\beta_t(i) = \sum_{j=1}^N a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)$ for $t = T-1, \ldots, 1$

The smoothing probabilities are then:
$P(z_t = i | o_{1:T}) = \frac{\alpha_t(i) \beta_t(i)}{P(o_{1:T})}$

## The HDP-HMM

### Model Definition

The HDP-HMM uses the HDP to define the transition distributions of an HMM with a potentially infinite number of states. The generative process is:

1. Draw a global distribution over states: $\beta \sim \text{GEM}(\gamma)$
2. For each state $i$, draw a transition distribution: $\pi_i \sim \text{DP}(\alpha, \beta)$
3. For each time step $t$:
   - Draw state: $z_t \sim \pi_{z_{t-1}}$
   - Draw observation: $o_t \sim F(\theta_{z_t})$

where $F(\theta)$ is the emission distribution with parameters $\theta$.

In matrix form, the transition probability from state $i$ to state $j$ is:
$a_{ij} = P(z_t = j | z_{t-1} = i) = \pi_{ij}$

The key advantage of the HDP-HMM is that it allows for a potentially infinite number of states, with the actual number determined by the data.

### Sticky HDP-HMM

A limitation of the basic HDP-HMM is that it does not incorporate any bias towards self-transitions, which are common in many real-world processes. The sticky HDP-HMM addresses this by adding a self-transition bias:

For each state $i$, draw a transition distribution:
$\pi_i \sim \text{DP}\left(\alpha + \kappa, \frac{\alpha \beta + \kappa \delta_i}{\alpha + \kappa}\right)$

where $\kappa > 0$ is a parameter that controls the degree of self-transition bias, and $\delta_i$ is a Dirac measure centered at state $i$.

This modification increases the expected probability of self-transitions by an amount proportional to $\kappa/(\alpha + \kappa)$.

## Emission Models

The choice of emission model $F(\theta)$ depends on the nature of the observations:

### Gaussian Emissions

For continuous observations, a common choice is the multivariate Gaussian:
$o_t | z_t = k \sim \mathcal{N}(\mu_k, \Sigma_k)$

The log-likelihood of an observation $o_t$ given state $k$ is:
$\log P(o_t | z_t = k) = -\frac{D}{2} \log(2\pi) - \frac{1}{2} \log |\Sigma_k| - \frac{1}{2} (o_t - \mu_k)^T \Sigma_k^{-1} (o_t - \mu_k)$

where $D$ is the dimensionality of the observations.

In our implementation, we use a diagonal covariance matrix for simplicity:
$\Sigma_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kD}^2)$

This simplifies the log-likelihood to:
$\log P(o_t | z_t = k) = -\frac{D}{2} \log(2\pi) - \frac{1}{2} \sum_{d=1}^D \log \sigma_{kd}^2 - \frac{1}{2} \sum_{d=1}^D \frac{(o_{td} - \mu_{kd})^2}{\sigma_{kd}^2}$

### Discrete Emissions

For discrete observations, we can use a categorical distribution:
$o_t | z_t = k \sim \text{Categorical}(\phi_k)$

where $\phi_k$ is a probability vector over the possible observation values.

## Learning Algorithms

### Gibbs Sampling

Gibbs sampling for the HDP-HMM involves iteratively sampling:

1. The hidden states $z_{1:T}$ given the current parameters and observations.
2. The transition distributions $\pi_i$ for each state $i$.
3. The global distribution $\beta$.
4. The emission parameters $\theta_k$ for each state $k$.

The state sampling uses a variant of the forward-backward algorithm that accounts for the infinite state space.

### Variational Inference

Variational inference approximates the posterior distribution with a simpler distribution and optimizes it to be as close as possible to the true posterior. For the HDP-HMM, this involves:

1. Truncating the infinite state space to a finite but large number of states.
2. Defining variational distributions over all parameters.
3. Optimizing the evidence lower bound (ELBO).

### Beam Sampling

Beam sampling combines slice sampling with dynamic programming to efficiently sample from the posterior of an HDP-HMM. It introduces auxiliary variables that allow the forward-backward algorithm to be run on a finite subset of states at each iteration.

## Dynamic State Management

In our implementation, we use a combination of approaches inspired by variational inference and heuristic methods to dynamically manage the number of states:

### Birth Mechanism

New states are created when the current model fails to explain some observations well:

1. Compute the negative log-likelihood for each observation.
2. If the average negative log-likelihood exceeds a threshold, create a new state.
3. Initialize the new state based on the poorly explained observations.

### Merge Mechanism

Similar states are merged to prevent redundancy:

1. Compute the distance between each pair of state means.
2. If the distance is below a threshold, merge the states.
3. Update the parameters of the merged state as a weighted average of the original states.

### Delete Mechanism

States with negligible probability are removed:

1. Compute the global weight $\beta_k$ for each state $k$.
2. If the weight is below a threshold, mark the state for deletion.
3. Reorder the remaining states to maintain a compact representation.

These mechanisms work together to maintain an appropriate number of states that best explains the data.

## References

1. Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet Processes. Journal of the American Statistical Association, 101(476), 1566-1581.

2. Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2008). An HDP-HMM for Systems with State Persistence. In Proceedings of the 25th International Conference on Machine Learning (ICML).

3. Beal, M. J., Ghahramani, Z., & Rasmussen, C. E. (2002). The Infinite Hidden Markov Model. In Advances in Neural Information Processing Systems (NIPS).

4. Van Gael, J., Saatci, Y., Teh, Y. W., & Ghahramani, Z. (2008). Beam Sampling for the Infinite Hidden Markov Model. In Proceedings of the 25th International Conference on Machine Learning (ICML).

5. Wang, C., & Blei, D. M. (2012). Truncation-free Online Variational Inference for Bayesian Nonparametric Models. In Advances in Neural Information Processing Systems (NIPS).

6. Hughes, M. C., Stephenson, W. T., & Sudderth, E. (2015). Scalable Adaptation of State Complexity for Nonparametric Hidden Markov Models. In Advances in Neural Information Processing Systems (NIPS).
