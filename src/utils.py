import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy.random import choice
from sklearn.utils import shuffle as skshuffle
# from tensorflow_probability.python.distributions import Mixture, Categorical, MultivariateNormalTriL
# import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import (MultivariateNormalTriL,
                                                         Mixture,
                                                         Categorical,
                                                         Logistic,
                                                         Distribution,
                                                         MultivariateNormalDiag)

# from tensorflow_probability.python.layers import DistributionLambda
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')


def make_spiral_galaxy(n_spirals=5, length=1, angle=np.pi / 2, n_samples=100, noise=0, shuffle=True):
    thetas = np.linspace(0, np.pi * 2, n_spirals + 1)
    thetas = thetas[:-1]
    radius = np.linspace(np.zeros(len(thetas)) + 0.1, np.ones(len(thetas)) * length + 0.1, n_samples)
    angles = np.linspace(thetas, thetas + angle, n_samples)
    if noise:
        angles += np.random.normal(size=angles.shape) * noise * np.linspace(1.5, .1, n_samples)[:, None]
    x0 = np.cos(angles) * radius
    x1 = np.sin(angles) * radius
    x0 = x0.T.reshape(-1, 1)
    x1 = x1.T.reshape(-1, 1)
    xy = np.concatenate([x0, x1], -1)
    y = np.repeat(np.arange(n_spirals), n_samples)
    if shuffle:
        xy, y = skshuffle(xy, y)
    return xy, y


def make_base_points(x_lim=(-5, 5), y_lim=(-5, 5), num=200):
    x = np.linspace(*x_lim, num=num)
    y = np.linspace(*y_lim, num=num)
    xx, yy = np.meshgrid(x, y)
    xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], -1)
    return xx, yy, xy


def make_cross_shaped_distribution(n_samples):
    mix = Mixture(cat=Categorical(probs=[1 / 4, 1 / 4, 1 / 4, 1 / 4]),
                  components=[
                      MultivariateNormalTriL(loc=[0, 2], scale_tril=tf.linalg.cholesky([[.15 ** 2, 0], [0, 1]])),
                      MultivariateNormalTriL(loc=[-2, 0], scale_tril=tf.linalg.cholesky([[1, 0], [0, .15 ** 2]])),
                      MultivariateNormalTriL(loc=[2, 0], scale_tril=tf.linalg.cholesky([[1, 0], [0, .15 ** 2]])),
                      MultivariateNormalTriL(loc=[0, -2], scale_tril=tf.linalg.cholesky([[.15 ** 2, 0], [0, 1]]))
                  ])
    return mix.sample(n_samples).numpy()


def make_grad_plot(model, x_lim=(-5, 5), y_lim=(-5, 5), num=50, ax=None, fig=None, iter=None):
    xx, yy, xy = make_base_points(x_lim, y_lim, num)
    o = model(xy)
    if len(o) == 1:
        grad = o
        e = None
        title = "Estimated Vector Field of the Probability Distribution $ \\mathbb{\\hat{{P}}} $ : $ \\nabla_{x} \\mathbb{\\hat{{P}}}(x) $"
    elif len(o) == 3:
        e, grad, hess = o
        ee = e.numpy().reshape(num, num)
        title = "Estimated Vector Field of the Estimated Energy Model $ \\mathbb{\\hat{{E}}} $ : $ \\nabla_{x} \\mathbb{\\hat{{E}}}(x) $"
    if iter is not None:
        title += f" at Iteration {iter}"
    assert grad.shape[-1] == 2
    dxx, dyy = np.split(grad.numpy(), 2, -1)
    dxx, dyy = dxx.reshape(num, num), dyy.reshape(num, num)
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_title(title)
    if e is not None:
        ax.set_title(title)
        img = ax.contourf(xx, yy, ee, levels=100)
        ax.contour(xx, yy, ee, levels=20, colors="black")
        if fig is not None:
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(img, cax=cax1, orientation="vertical")
    ax.quiver(xx, yy, dxx, dyy)
    return ax


def make_distribution_grad_plot(distr, x_lim=(-5, 5), y_lim=(-5, 5), num=200, reduce=10, ax=None, fig=None):
    xx, yy, xy = make_base_points(x_lim, y_lim, num)
    with tf.GradientTape() as tape:
        xy = tf.convert_to_tensor(xy, tf.float32)
        tape.watch(xy)
        ll = distr.log_prob(xy)
    grads = tape.gradient(ll, xy)
    dxx, dyy = tf.split(grads, 2, -1)
    dxx, dyy = tf.reshape(dxx, (num, num)).numpy(), tf.reshape(dyy, (num, num)).numpy()
    ll = ll.numpy().reshape(num, num)
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_title("True Vector Field $ \\nabla_{x} \\mathbb{P}(x) $")
    img = ax.contourf(xx, yy, ll, levels=100)
    ax.contour(xx, yy, ll, levels=20, colors="black")
    ax.quiver(xx[::reduce, ::reduce], yy[::reduce, ::reduce], dxx[::reduce, ::reduce], dyy[::reduce, ::reduce])
    if fig is not None:
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax1, orientation="vertical")
        return fig, ax
    return ax
