import tensorflow as tf
from typing import List, Tuple

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
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="score")
        self.f = self.make_score_model(hidden_layers=hidden_layers,
                                       activation=activation,
                                       output_dim=output_dim)

    def call(self, inputs, training=False, mask=None):
        if training:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                grad = self.f(inputs)
            hess = tape.batch_jacobian(grad, inputs)
            return grad, hess
        return self.f(inputs)

    def sliced_score_estimator(self, data):
        grad, hess = self(data, training=True)
        rnd = tf.random.normal(shape=tf.shape(hess)[:2])
        sliced_hess = 0.5 * tf.einsum("bi,bio,bo->b", rnd, hess, rnd)  # eq. 8 trace estimator
        sliced_grad = 0.5 * tf.einsum("bi,bi->b", rnd, grad) ** 2  # # eq. 8 trace estimator
        return sliced_hess + sliced_grad

    def loss_fn(self, data):
        loss = self.sliced_score_estimator(data)
        return tf.reduce_mean(loss)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(data)
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



class EBMSlicedScoreMatching(Model):
    """
    https://proceedings.mlr.press/v115/song20a.html
    Sliced Score Matching: A Scalable Approach to Density and Score Estimation
    """

    def __init__(self,
                 hidden_layers: Tuple[int, ...] = (100, 50),
                 output_dim: int = 2,
                 activation: str = "relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="score")
        self.f = self.make_score_model(hidden_layers=hidden_layers,
                                       activation=activation,
                                       output_dim=output_dim)

    def call(self, inputs, training=None, mask=None):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                e = self.f(inputs)
            grad = tape1.gradient(e, inputs)
        hess = tape.batch_jacobian(grad, inputs)
        return e, grad, hess

    def sliced_score_estimator(self, data):
        e, grad, hess = self(data, training=True)
        rnd = tf.random.normal(shape=tf.shape(hess)[:2])
        sliced_hess = 0.5 * tf.einsum("bi,bio,bo->b", rnd, hess, rnd)  # eq. 8 trace estimator
        sliced_grad = 0.5 * tf.einsum("bi,bi->b", rnd, grad) ** 2  # # eq. 8 trace estimator
        return sliced_hess + sliced_grad

    def loss_fn(self, data):
        loss = self.sliced_score_estimator(data)
        return tf.reduce_mean(loss)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(data)
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
        l = Dense(1, activation="elu")
        o = l(x)
        return Model(i, o)
