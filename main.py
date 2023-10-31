import matplotlib.pyplot as plt
from src import models, utils
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import (MultivariateNormalTriL,
                                                         Mixture,
                                                         Categorical,
                                                         Logistic,
                                                         Distribution,
                                                         MultivariateNormalDiag)

mix = Mixture(cat=Categorical(logits=[1, 1]),
              components=[MultivariateNormalDiag([-1., -1.], [.5, .5]),
                          MultivariateNormalDiag([1., 1.], [.5, .5])])

x = mix.sample(3000)
model = models.SlicedScoreMatching(hidden_layers=(512, 512))
model.compile("adam")
model.fit(x, epochs=1000, batch_size=3000)
fig, ax = plt.subplots(2)
fig, ax0 = utils.make_distribution_grad_plot(mix, fig=fig, ax=ax[0])
ax0.scatter(x[:, 0], x[:, 1], color="black", s=5)
ax1 = utils.make_grad_plot(model, ax=ax[1])