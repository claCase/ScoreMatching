from math import pi
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
import numpy as np
from numpy.random import choice
from sklearn.utils import shuffle as skshuffle
from tensorflow_probability.python.distributions import (MultivariateNormalTriL,
                                                         Mixture,
                                                         Categorical,
                                                         Logistic,
                                                         Distribution,
                                                         MultivariateNormalDiag)

from datetime import datetime

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


def make_cross_shaped_distribution(n_samples):
    mix = Mixture(cat=Categorical(probs=[1 / 4, 1 / 4, 1 / 4, 1 / 4]),
                  components=[
                      MultivariateNormalTriL(loc=[0, 2], scale_tril=tf.linalg.cholesky([[.15 ** 2, 0], [0, 1]])),
                      MultivariateNormalTriL(loc=[-2, 0], scale_tril=tf.linalg.cholesky([[1, 0], [0, .15 ** 2]])),
                      MultivariateNormalTriL(loc=[2, 0], scale_tril=tf.linalg.cholesky([[1, 0], [0, .15 ** 2]])),
                      MultivariateNormalTriL(loc=[0, -2], scale_tril=tf.linalg.cholesky([[.15 ** 2, 0], [0, 1]]))
                  ])
    return mix.sample(n_samples).numpy()


def save_output_callback(model, inputs, save_path, name="default", every=5, stop=300):
    now = datetime.now().isoformat()[:-7].replace(":", "_")
    save_path = os.path.join(save_path, now)
    os.makedirs(save_path)
    if isinstance(inputs, tf.Tensor):
        inputs = inputs.numpy()
    np.save(os.path.join(save_path, "inputs.npy"), inputs)

    def on_epoch_end(epoch, logs):
        if not epoch % every and epoch <= stop:
            o = model(inputs, training=True)
            if len(o) == 1:
                e, grad, hess = None, o, None
            elif len(o) == 3:
                e, grad, hess = o
            else:
                raise NotImplementedError

            if e is not None:
                e = e.numpy()
                np.save(os.path.join(save_path, str(epoch) + "_" + name + "_energy.npy"), e)
            if hess is not None:
                hess = hess.numpy()
                np.save(os.path.join(save_path, str(epoch) + "_" + name + "_hessian.npy"), hess)
            np.save(os.path.join(save_path, str(epoch) + "_" + name + "_grad.npy"), grad)

    return tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end), save_path


class AnnealNoiseSamples(tf.keras.callbacks.Callback):
    def __init__(self, start=2, anneal=0.1, type="linear"):
        super().__init__()
        self.anneal = tf.abs(anneal)
        self.type = type
        self.start = start
        self.lr = start

    def linear(self, i):
        self.lr = tf.maximum(self.start - self.anneal * i, 0.)
        return self.lr

    def cosine(self, i):
        self.lr = tf.cos(i * pi / 4) ** 2 * self.sigmoid(i)
        return self.lr

    def sigmoid(self, i):
        self.lr = tf.maximum(2. * self.start / (1. + tf.exp(i) ** self.anneal), 0.)

    def get_schedule(self, type):
        if type == "linear":
            return self.linear
        elif type == "cosine":
            return self.cosine
        elif type == "sigmoid":
            return self.sigmoid
        else:
            raise NotImplementedError(
                f"Type of schedule {type} is not supported, choose from 'linear', 'cosine', 'sigmoid'")

    def on_epoch_end(self, epoch, logs=None):
        epoch = tf.cast(epoch, tf.float32)
        if hasattr(self.model, "anneal_samples"):
            self.model.anneal_samples = self.get_schedule(self.type)(epoch)
        else:
            raise AttributeError(f"Model does not have anneal_samples attribute")

    def on_train_end(self, logs=None):
        self.lr = self.start
        self.model.anneal_samples = 1.
