import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
import numpy as np


def make_base_points(x_lim=(-5, 5), y_lim=(-5, 5), num=200):
    x = np.linspace(*x_lim, num=num)
    y = np.linspace(*y_lim, num=num)
    xx, yy = np.meshgrid(x, y)
    xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], -1)
    return xx, yy, xy


def make_grad_plot(model=None, x_lim=(-5, 5), y_lim=(-5, 5), num=50, reduce=5, ax=None, fig=None, iter=None,
                   grad=None, e=None, xy=None):
    assert model is not None or (grad is not None and xy is not None)
    if model is not None:
        xx, yy, xy = make_base_points(x_lim, y_lim, num)
        o = model(xy, training=True)
        if len(o) == 3:
            e, grad, _ = o
            e, grad = e.numpy(), grad.numpy()
        elif len(o) == 2:
            grad, _ = o.numpy()
            e = None
        else:
            raise ValueError(
                f"The model in training mode must return either (energy, grad, hessian) or (grad, hessian)")
    else:
        if isinstance(grad, tf.Tensor):
            grad = grad.numpy()
        num = int(np.sqrt(xy.shape[0]))
        assert num ** 2 == xy.shape[0]
        xx, yy = np.split(xy, 2, -1)
        xx, yy = np.reshape(xx, (num, num)), np.reshape(yy, (num, num))
    if e is not None:
        title = "Estimated Vector Field of the Estimated Energy Model $ \\mathbb{\\hat{{E}}} $ : $ \\nabla_{x} \\mathbb{\\hat{{E}}}(x) $"
        ee = e.reshape(num, num)
    else:
        title = "Estimated Vector Field of the Probability Distribution $ \\mathbb{\\hat{{P}}} $ : $ \\nabla_{x} \\mathbb{\\hat{{P}}}(x) $"

    if iter is not None:
        title += f" at Iteration {iter}"
    assert grad.shape[-1] == 2
    dxx, dyy = np.split(grad, 2, -1)
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
    ax.quiver(xx[::reduce, ::reduce], yy[::reduce, ::reduce], dxx[::reduce, ::reduce], dyy[::reduce, ::reduce])
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


def make_training_animation(save_path, dpi=250, fps=60):
    path, dirs, files = next(os.walk(save_path))
    epochs = set()
    types = set()
    names = set()
    for i in files:
        if i != "inputs.npy":
            try:
                splits = i.split("_")
                k = int(splits[0])
                obj = splits[-1].split(".")[0]
                epochs.update({k})
                types.update({obj})
                name = splits[1]
                names.update({name})
            except:
                pass
    epochs = list(epochs)
    epochs.sort()
    inputs = np.load(os.path.join(save_path, "inputs.npy"))
    fig, ax = plt.subplots(1, figsize=(15, 15))

    def plotter_grad(i):
        print(i)
        ax.clear()
        grad = np.load(os.path.join(save_path, str(epochs[i]) + "_" + name + "_grad.npy"))
        if "energy" in types:
            energy = np.load(os.path.join(save_path, str(epochs[i]) + "_" + name + "_energy.npy"))
        else:
            energy = None
        make_grad_plot(grad=grad, e=energy, xy=inputs, ax=ax, iter=i)

    anim = animation.FuncAnimation(fig, plotter_grad, frames=len(epochs) - 1)
    anim.save(os.path.join(save_path, "animation.gif"), fps=fps, dpi=dpi)
