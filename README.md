Non-official implementation of [sliced score matching and de-noising score matching](https://arxiv.org/abs/1907.05600) in TensorFlow v2. The basic idea of score matching is to estimate the gradient field of an un-normalized energy function to match the true gradient field of the normalized data generating density function.
The gradient near sampled points should be close to zero to indicate high probability of sampling. 
# Sliced Score Matching
The loss function takes the following form:
```math
\mathcal{L}(\theta) = \mathbb{E}_{\mathbb{P}(x)} [ Tr(\nabla_{x}\bf{s}_{\theta}(x)) + \frac{1}{2} \| \bf{s}_{\theta}(x) \|^{2}_{2} ]
```

Where $` Tr `$ is the trace operator and $` \bf{s}_{\theta}(x) `$ is the score (gradient) estimator of the energy function $` \bf E_{\theta}: \mathbb{R}^{d} \rightarrow \mathbb{R} `$  with parameters $` \theta `$ :

```math
\bf{s}_{\theta}(x) = \nabla_{x} \bf E_{\theta}(x)
```

The energy $` \bf E_{\theta} `$ estimates the un-normalized log-density function:

```math
log(\mathbb{P}_{\theta}(x)) = log(\frac{\bf e^{E_{\theta}(x)}}{\bf Z})
```

```math
\Rightarrow \bf E_{\theta}(x) \propto log(\mathbb{P}_{\theta}(x))  
```

where $` \bf Z `$ is the partition function (normalizing constant).
\
\
The trace of the hessian matrix $` Tr(\nabla_{x}\bf{s}_{\theta}(x)) `$ can be estimated by the Hutchinson trace estimator:

```math
Tr(\nabla_{x}\bf{s}_{\theta}(x)) = \mathbb{E}_{v \sim \mathbb{P}(v) }[v^\top \nabla_{x} \bf{s}_{\theta}(x) v]
```

where $` \mathbb{P}(v) `$ can be a normal distribution $` \mathcal{N}(0,1) `$ or a rademacher distribution. 

# De-noising Score Matching  
Another way to estimate the gradient is to add noise to the samples and trying to match the gradient of the noise distribution, which is chosen a priori to be gaussian:

```math
q_{\sigma}(\tilde x |x) = \mathcal N(x, \sigma)
```

```math
\nabla_{\tilde x} log \, q_{\sigma}(\tilde x |x) = - \frac{\tilde x -x}{\sigma^{2}}
```

where $` q_{\sigma}(\tilde x |x) `$ is the gaussian noise distribution. The gradient has close form, thereby we can take the difference between the score gradient estimated by the model and the true gradient:

```math
\mathcal L(\theta, \sigma) = \frac{1}{2} \mathbb E_{x \sim p_{data}(x)} \mathbb E_{\tilde x \sim \mathcal N(x, \sigma)}[ (\bf{s_{\theta}}(\tilde x, \sigma) -  \nabla_{\tilde x} log \, q_{\sigma}(\tilde x |x) )^2] 
```
```math
\Rightarrow \mathcal L(\theta, \sigma) = \frac{1}{2} \mathbb E_{p_{data}(x)} \mathbb E_{\mathcal N(x, \sigma)}[ (\bf{s_{\theta}}(\tilde x, \sigma) - (- \frac{\tilde x -x}{\sigma^{2}}))^2] 
```

To increase the accuracy of the model the final loss function is weighted in proportion to the variance of the noise distribution: 

```math
\mathcal L(\theta, \{ \sigma_{i} \}_{i=1}^L) = \frac{1}{L} \sum_{i}^{L} \sigma^2_{i} \mathcal L(\theta, \sigma_{i})
```

# Sampling by Langevin Dynamics

To generate samples from the model one can uniformly sample some points in the region of the data manifold, and then update the sampled points in the direction of the gradient predicted by the score model and add some small noise perturbation to make the trajectory stochastic: 

```math
x_{t+1} = x_{t} + \frac{1}{2} \alpha_{t} *  s_{\theta}  (x) + \sqrt{\alpha} \, n
```
where $` n \sim \mathcal N(0, 1) `$ and  $` \alpha `$ is a step size hyperparameter. 


### Energy Based De-noising Score Matching
Evolution of the estimated vector field of the density score
![Training Vector Filed Evolution](https://github.com/claCase/ScoreMatching/blob/master/figures/Denoising%20Score%20Matching/Gaussian%20Mixture/2023-11-04T03_59_29/ebm_de-noising_animation_animation.gif)

Evolution of samples points via annealed Langevin Dynamics 
![Langevin Sampling](https://github.com/claCase/ScoreMatching/blob/master/figures/Denoising%20Score%20Matching/Gaussian%20Mixture/2023-11-04T03_59_29/default_trajectory_animation.gif)


# Results 
### Energy Based Sliced Score Matching 
![Training Vector Filed Evolution](https://github.com/claCase/ScoreMatching/blob/master/figures/Sliced%20Score%20Matching/2023-11-02T03_56_44/sliced_score_matching_2gif_animation.gif)


