import tensorflow as tf


def noise(shape, type="gaussian"):
    rnd = tf.random.normal(shape=shape)
    if type == "gaussian":
        return rnd
    elif type == "radermacher":
        return tf.sign(rnd)
    elif type == "spherical":
        return rnd / tf.linalg.norm(rnd, axis=-1, keepdims=True) * tf.math.sqrt(tf.shape(rnd)[-1])
    else:
        raise NotImplementedError(f"Noise type must be in 'gaussian', 'radermacher', 'spherical'")


def sliced_score_estimator(grad, hess, noise_type="gaussian"):
    rnd = noise(shape=tf.shape(hess)[:2], type=noise_type)
    sliced_hess = 0.5 * tf.einsum("bi,bio,bo->b", rnd, hess, rnd)  # eq. 8 trace estimator
    sliced_grad = 0.5 * tf.einsum("bi,bi->b", rnd, grad) ** 2  # # eq. 8 trace estimator
    return sliced_hess + sliced_grad


def sliced_score_estimator_vr(grad, hess, noise_type="gaussian"):
    rnd = tf.random.normal(shape=tf.shape(hess)[:2], type=noise_type)
    sliced_hess = 0.5 * tf.einsum("bi,bio,bo->b", rnd, hess, rnd)  # eq. 8 trace estimator
    sliced_grad = 0.5 * tf.linalg.norm(grad, axis=-1) ** 2
    return sliced_hess + sliced_grad


def score_loss(grad, hess, vr=False, noise_type="gaussian"):
    if vr:
        return tf.reduce_mean(sliced_score_estimator_vr(grad, hess))
    return tf.reduce_mean(sliced_score_estimator(grad, hess))
