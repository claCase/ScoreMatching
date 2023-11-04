import tensorflow as tf
from tensorflow_probability.python.distributions import Normal
from src.utils import noise
from typing import Union, Tuple


def sliced_score_estimator(grad, hess, noise_type="gaussian"):
    rnd = noise(shape=tf.shape(hess)[:2], type=noise_type)
    sliced_hess = 0.5 * tf.einsum(
        "bi,bio,bo->b", rnd, hess, rnd
    )  # eq. 8 trace estimator
    sliced_grad = 0.5 * tf.einsum("bi,bi->b", rnd, grad) ** 2  # # eq. 8 trace estimator
    return sliced_hess + sliced_grad


def sliced_score_estimator_vr(grad, hess, noise_type="gaussian"):
    rnd = tf.random.normal(shape=tf.shape(hess)[:2], type=noise_type)
    sliced_hess = 0.5 * tf.einsum(
        "bi,bio,bo->b", rnd, hess, rnd
    )  # eq. 8 trace estimator
    sliced_grad = 0.5 * tf.linalg.norm(grad, axis=-1) ** 2
    return sliced_hess + sliced_grad


def score_loss(grad, hess, vr=False, noise_type="gaussian"):
    if vr:
        return tf.reduce_mean(sliced_score_estimator_vr(grad, hess, noise_type))
    return tf.reduce_mean(sliced_score_estimator(grad, hess, noise_type))


def noise_conditional_score_matching(
    grad: tf.Tensor, samples: tf.Tensor, corrupted_samples: tf.Tensor, sigma: float
):
    """
    Computes de-noising score estimation loss function
    :param grad: gradients of energy function of noised samples B x d
    :param samples: original uncorrupted samples os shape B x d
    :param corrupted_samples: noised samples of shape B x d
    :param sigma: standard deviation
    :return: B
    """
    targets = -(corrupted_samples - samples) / sigma**2
    loss = tf.reduce_sum(0.5 * (grad - targets) ** 2, -1) * sigma**2  # B
    loss = tf.reduce_mean(loss)
    return loss


def annealed_noise_conditional_score_matching(
    grad: tf.Tensor,
    samples: tf.Tensor,
    corrupted_samples: tf.Tensor,
    sigmas: Union[tf.Tensor, Tuple],
):
    """
    Computes de-noising score estimation loss function
    :param grad: gradients of energy function of noised samples B*|σ| x d
    :param samples: original uncorrupted samples os shape B x d
    :param corrupted_samples: noised samples of shape B*|σ| x d
    :param sigmas: standard deviation of shape |σ|
    :return: B
    """
    if not isinstance(sigmas, tf.Tensor):
        sigmas = tf.convert_to_tensor(sigmas, grad.dtype)
    ss = tf.shape(samples)
    B, d = ss[0], ss[1]
    s = tf.shape(sigmas)[0]
    r_corrupted = tf.reshape(corrupted_samples, (B, s, d))  # B x |σ| x d
    targets = -(r_corrupted - samples[:, None, :]) / sigmas[None, :, None] ** 2
    grad = tf.reshape(grad, (B, s, d))  # B x |σ| x d
    loss = (
        tf.reduce_sum(0.5 * (grad - targets) ** 2, -1) * sigmas[None, :] ** 2
    )  # B x |σ|
    loss = tf.reduce_mean(loss)
    return loss


def noise_conditional_score_matching_loss(grad, samples, corrupted_samples, sigmas):
    if isinstance(sigmas, float):
        tf.debugging.assert_shapes(
            [(grad, ("B", "d")), (samples, ("B", "d")), (corrupted_samples, ("B", "d"))]
        )
        return noise_conditional_score_matching(
            grad, samples, corrupted_samples, sigmas
        )
    elif isinstance(sigmas, tuple) or isinstance(sigmas, tf.Tensor):
        tf.debugging.assert_shapes(
            [
                (grad, ("BS", "d")),
                (corrupted_samples, ("BS", "d")),
                (samples, ("B", "d")),
            ]
        )
        return annealed_noise_conditional_score_matching(
            grad, samples, corrupted_samples, sigmas
        )
