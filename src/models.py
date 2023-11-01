import tensorflow as tf
from typing import List, Tuple
from src.losses import score_loss

k = tf.keras
Model = k.models.Model
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
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="score")
        self.f = self.make_score_model(hidden_layers=hidden_layers,
                                       activation=activation,
                                       output_dim=output_dim)
        self.vr = vr

    def build(self, input_shape):
        super().build()
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
        with tf.GradientTape() as tape:
            grad, hess = self(data, training=True)
            loss = score_loss(grad, hess, vr=self.vr)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"Score": self.loss_tracker.result()}

    def langevin_dynamics(self, initial_points=None, steps=500):
        try:
            out_dim = self.layers[-1].output_shape[-1]
        except AttributeError as e:
            raise e
        except RuntimeError as e:
            raise e

        def alpha(i):
            return 100 / (100 + i)

        if initial_points is None:
            x = tf.random.normal(shape=(1000, out_dim))
        else:
            x = initial_points

        for t in range(steps):
            a = alpha(t)
            x = x + 0.5 * a * self.f(x) + tf.math.sqrt(a) * tf.random.normal(shape=(1, out_dim))
        return x

    def annealed_langevin_dynamics(self, initial_points=None, steps=500, sigma_high=2, sigma_low=0.1, levels=10, e=1.):
        try:
            out_dim = self.layers[-1].output_shape[-1]
        except AttributeError as e:
            raise e
        except RuntimeError as e:
            raise e

        alphas = tf.linspace(sigma_low, sigma_high, levels)[::-1]

        if initial_points is None:
            x = tf.random.normal(shape=(1000, out_dim))
        else:
            x = initial_points

        for l in range(len(alphas) - 1):
            a = e * alphas[l + 1] / alphas[l]
            for t in range(steps):
                x = x + 0.5 * a * self.f(x) + tf.math.sqrt(a) * tf.random.normal(shape=(1, out_dim))
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
                 output_dim: int = 2,
                 activation: str = "relu",
                 vr=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="score")
        self.f = self.make_score_model(hidden_layers=hidden_layers,
                                       activation=activation,
                                       output_dim=output_dim)
        self.vr = vr

    def call(self, inputs, training=False, mask=None):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                e = self.f(inputs)
            grad = tape1.gradient(e, inputs)
        hess = tape.batch_jacobian(grad, inputs)
        if training:
            return -e, grad, hess
        else:
            return grad

    def train_step(self, data):
        with tf.GradientTape() as tape:
            e, grad, hess = self(data, training=True)
            loss = self.loss_fn(grad, hess, vr=self.vr)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"Score": self.loss_tracker.result()}

    @staticmethod
    def make_score_model(hidden_layers, activation, output_dim):
        i = tf.keras.layers.Input(shape=(output_dim,))
        for k, h in enumerate(hidden_layers):
            l = Dense(h, activation)
            if k == 0:
                x = l(i)
            else:
                x = l(x)
        l = Dense(1, activation="elu")
        o = l(x)
        return Model(i, o)
