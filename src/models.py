import numpy as np
import tensorflow as tf
from typing import List, Tuple
from src.losses import score_loss

k = tf.keras
Model, Sequential = k.models.Model, k.models.Sequential
Dense = k.layers.Dense


class SlicedScoreMatching(Model):
    """
    https://proceedings.mlr.press/v115/song20a.html
    Sliced Score Matching: A Scalable Approach to Density and Score Estimation
    """

    def __init__(self,
                 hidden_layers: Tuple[int, ...] = (100, 50),
                 output_dim: int = 2,
                 activation: str = "relu",
                 vr=False,
                 noise_type="gaussian",
                 anneal_samples=0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="score")
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_dim = output_dim
        self.vr = vr
        self.noise_type = noise_type
        self.anneal_samples = anneal_samples

    def build(self, input_shape):
        super().build(input_shape)
        self.f = self.make_score_model(hidden_layers=self.hidden_layers,
                                       activation=self.activation,
                                       output_dim=self.output_dim)
        self.f.build(input_shape)

    def call(self, inputs, training=False, mask=None):
        if training:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                grad = self.f(inputs)
            hess = tape.batch_jacobian(grad, inputs)
            return grad, hess
        return self.f(inputs)

    def train_step(self, data):
        data += tf.random.normal(shape=tf.shape(data)) * self.anneal_samples
        with tf.GradientTape() as tape:
            grad, hess = self(data, training=True)
            loss = score_loss(grad, hess, vr=self.vr, noise_type=self.noise_type)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"Score": self.loss_tracker.result()}

    def langevin_dynamics(self,
                          initial_points=None,
                          steps=500,
                          x_lim=(-6, 6),
                          n_samples=100,
                          trajectories=False):
        try:
            in_dim = self.layers[0].input_shape[-1]
        except AttributeError as e:
            raise e
        except RuntimeError as e:
            raise e

        def alpha(i):
            return 100 / (100 + i)

        if initial_points is None:
            x = tf.random.uniform(minval=x_lim[0], maxval=x_lim[1], shape=(n_samples, in_dim))
        else:
            x = initial_points

        if trajectories:
            traj = np.empty(shape=(steps, n_samples, in_dim))
            traj[0, :, :] = x

        for t in range(1, steps):
            a = alpha(t)
            x = x + 0.5 * a * self.f(x) + tf.math.sqrt(a) * tf.random.normal(shape=(x.shape[0], in_dim))
            if trajectories:
                traj[t, :, :] = x.numpy()
        if trajectories:
            return traj
        return x

    def annealed_langevin_dynamics(self,
                                   initial_points=None,
                                   steps=500,
                                   n_samples=100,
                                   x_lim=(-6, 6),
                                   sigma_high=2,
                                   sigma_low=0.1,
                                   levels=10, e=1.,
                                   trajectories=False):
        try:
            in_dim = self.layers[0].input_shape[-1]
        except AttributeError as e:
            raise e
        except RuntimeError as e:
            raise e

        alphas = tf.linspace(sigma_low, sigma_high, levels)[::-1]

        if initial_points is None:
            x = tf.random.uniform(minval=x_lim[0], maxval=x_lim[1], shape=(n_samples, in_dim))
        else:
            assert initial_points.shape[-1] == in_dim
            x = initial_points

        if trajectories:
            traj = np.empty(shape=(steps, n_samples, in_dim))
            traj[0, :, :] = x

        for l in range(len(alphas)):
            a = e * alphas[l + 1] / alphas[l]
            for t in range(1, steps):
                if trajectories:
                    traj[t, :, :] = x.numpy()
                    print(traj[t, 0, :])
                x = x + 0.5 * a * self.f(x) + tf.math.sqrt(a) * tf.random.normal(shape=(x.shape[0], in_dim))
        if trajectories:
            return traj
        return x

    @staticmethod
    def make_score_model(hidden_layers, activation, output_dim):
        i = tf.keras.layers.Input(shape=(output_dim,))
        for k, h in enumerate(hidden_layers):
            l = Dense(h, activation)
            if k == 0:
                x = l(i)
            else:
                x = l(x)
        l = Dense(output_dim, activation="linear")
        o = l(x)
        return Model(i, o)


class EBMSlicedScoreMatching(SlicedScoreMatching):
    """
    https://proceedings.mlr.press/v115/song20a.html
    Sliced Score Matching: A Scalable Approach to Density and Score Estimation
    """

    def __init__(self,
                 hidden_layers: Tuple[int, ...] = (100, 50),
                 activation: str = "relu",
                 vr=False,
                 noise_type="gaussian",
                 anneal_samples=0.,
                 **kwargs):
        super().__init__(anneal_samples=anneal_samples, output_dim=1, noise_type=noise_type, vr=vr, **kwargs)
        self.f = self.make_score_model(hidden_layers=hidden_layers, activation=activation)

    def build(self, input_shape):
        super(Model, self).build(input_shape)
        self.f = self.make_score_model(hidden_layers=self.hidden_layers,
                                       activation=self.activation)
        self.f.build(input_shape)

    def call(self, inputs, training=False, mask=None):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                e = self.f(inputs)
            grad = tape1.gradient(e, inputs)
        hess = tape.batch_jacobian(grad, inputs)
        if training:
            return e, grad, hess
        else:
            return grad

    def train_step(self, data):
        data += tf.random.normal(shape=tf.shape(data)) * self.anneal_samples
        with tf.GradientTape() as tape:
            e, grad, hess = self(data, training=True)
            loss = score_loss(grad, hess, vr=self.vr, noise_type=self.noise_type)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"Score": self.loss_tracker.result()}

    @staticmethod
    def make_score_model(hidden_layers, activation):
        layers = []
        for h in hidden_layers:
            layers.append(Dense(h, activation))
        layers.append(Dense(1, activation="elu"))
        return Sequential(layers)
