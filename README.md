# Scliced Score Matching

Non Ufficial implementation of sliced score matching in TensorFlow v2. The basic idea is to estimate the gradient field
of an un-normalized energy function to match the true gradient field of the normalized data generating density function.
The gradient where the sampled points lie should be close to zero to indicate high probability of sampling.
The loss function takes the following form:

```math
\mathcal{L}(\theta) = \mathbb{E}_{\mathbb{P}(x)} [ Tr(\nabla_{x}\bf{s}_{\theta}(x)) + \frac{1}{2} \| \bf{s}_{\theta}(x) \|^{2}_{2} ]
```

Where Tr is the trace operator and $` \bf{s}_{\theta}(x) `$ is the score (gradient) estimator of the energy function $` \bf E_{\theta}: \mathbb{R}^{d} \rightarrow \mathbb{R} `$  with parameters $` \theta `$ :

```math
\bf{s}_{\theta}(x) = \nabla_{x} \bf E_{\theta}(x)
```

The energy $` \bf E_{\theta} `$ is estimates the un-normalized log-density function:

```math
log(\mathbb{P}_{\theta}(x)) = log(\frac{\bf e^{E_{\theta}(x)}}{\bf Z})
```

```math
\Rightarrow \bf E_{\theta}(x) \propto log(\mathbb{P}_{\theta}(x))  
```

where $` \bf Z `$ is the partition function (normalizing constant), which does not depend on parameters.
\
\
The trace of the hessian matrix $` Tr(\nabla_{x}\bf{s}_{\theta}(x)) `$ can be estimated by the Hutchinson trace estimator:

```math
Tr(\nabla_{x}\bf{s}_{\theta}(x)) = \mathbb{E}_{v \sim \mathbb{P}(v) }[v^\top \nabla_{x} \bf{s}_{\theta}(x) v]
```

where $` \mathbb{P}(v) `$ can be a normal distribution $` \mathcal{N}(0,1) `$ or a rademacher distribution. 

# Results 

![Gradient Estimator](https://github.com/claCase/ScoreMatching/blob/master/figures/2023-11-02T03_56_44/GradPlot.png)
![Training Vector Filed Evolution](https://github.com/claCase/ScoreMatching/blob/master/figures/2023-11-02T03_56_44/30fps_dark_20fontsize_animation_compressed.gif)
