import tensorflow as tf


def sliced_score_estimator(grad, hess):
    rnd = tf.random.normal(shape=tf.shape(hess)[:2])
    sliced_hess = 0.5 * tf.einsum("bi,bio,bo->b", rnd, hess, rnd)  # eq. 8 trace estimator
    sliced_grad = 0.5 * tf.einsum("bi,bi->b", rnd, grad) ** 2  # # eq. 8 trace estimator
    return sliced_hess + sliced_grad


def sliced_score_estimator_vr(grad, hess):
    rnd = tf.random.normal(shape=tf.shape(hess)[:2])
    sliced_hess = 0.5 * tf.einsum("bi,bio,bo->b", rnd, hess, rnd)  # eq. 8 trace estimator
    sliced_grad = 0.5 * tf.linalg.norm(grad, -1) ** 2
    return sliced_hess + sliced_grad


def score_loss(grad, hess, vr=False):
    if vr:
        return tf.reduce_mean(sliced_score_estimator_vr(grad, hess))
    return tf.reduce_mean(sliced_score_estimator(grad, hess))
