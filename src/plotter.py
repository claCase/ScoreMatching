from tensorflow_probability.python.distributions import Distribution
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
import numpy as np

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
plt.style.use("dark_background")


def make_base_points(x_lim=(-5, 5), y_lim=(-5, 5), num=200):
    x = np.linspace(*x_lim, num=num)
    y = np.linspace(*y_lim, num=num)
    xx, yy = np.meshgrid(x, y)
    xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], -1)
    return xx, yy, xy


def make_grad_plot(
    model=None,
    x_lim=(-5, 5),
    y_lim=(-5, 5),
    num=50,
    reduce=5,
    ax=None,
    fig=None,
    iter=None,
    grad=None,
    e=None,
    xy=None,
    fontsize=20,
):
    assert model is not None or (grad is not None and xy is not None)
    if model is not None:
        xx, yy, xy = make_base_points(x_lim, y_lim, num)
        o = model(xy, training=True)
        if len(o) == 3:
            grad, _, e = o
            e, grad = e.numpy(), grad.numpy()
        elif len(o) == 2:
            grad, z = o
            z = z.numpy()
            if z.shape[-1] == 1:
                e = z
            else:
                e = None
        else:
            raise ValueError(
                f"The model in training mode must return either (energy, grad, hessian) or (grad, hessian)"
            )
    else:
        if isinstance(grad, tf.Tensor):
            grad = grad.numpy()
        num = int(np.sqrt(xy.shape[0]))
        assert num**2 == xy.shape[0]
        xx, yy = np.split(xy, 2, -1)
        xx, yy = np.reshape(xx, (num, num)), np.reshape(yy, (num, num))
    if e is not None:
        title = "Estimated Vector Field parametrized by the Energy Model $ \\hat{\\bf E}_{\\theta} $ : $ \\nabla_{x} \\hat{ \\bf E}_{\\theta}(x) $"
        ee = e.reshape(num, num)
    else:
        title = "Estimated Vector Field of the Probability Distribution $ \\mathbb{\\hat{{P}}} $ : $ \\nabla_{x} \\mathbb{\\hat{{P}}}(x) $"

    if iter is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        text = ax.text(
            xlim[0] - 0.05 * xlim[0],
            ylim[1] - 0.12 * ylim[1],
            f"Iteration {iter}",
            fontsize=15,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "black", "pad": 10},
        )

    assert grad.shape[-1] == 2
    dxx, dyy = np.split(grad, 2, -1)
    dxx, dyy = dxx.reshape(num, num), dyy.reshape(num, num)
    if ax is None:
        fig, ax = plt.subplots(1, dpi=300)
    ax.set_title(title)
    if e is not None:
        ax.set_title(title, fontsize=fontsize)
        img = ax.contourf(xx, yy, ee, levels=100)
        ax.contour(
            xx, yy, ee, levels=20, colors="black", linestyles="solid", linewidths=0.51
        )
        if fig is not None:
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(img, cax=cax1, orientation="vertical")
    ax.quiver(
        xx[::reduce, ::reduce],
        yy[::reduce, ::reduce],
        dxx[::reduce, ::reduce],
        dyy[::reduce, ::reduce],
    )
    if fig is not None:
        return fig, ax
    return ax


def make_distribution_grad_plot(
    distr,
    x_lim=(-5, 5),
    y_lim=(-5, 5),
    num=200,
    reduce=10,
    ax=None,
    fig=None,
    fontsize=20,
):
    assert issubclass(type(distr), Distribution)
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
        fig, ax = plt.subplots(1, dpi=300)
    ax.set_title(
        "True Vector Field of the probability distribution $\\mathbb{P} $: $ \\nabla_{x} \\mathbb{P}(x) $",
        fontsize=fontsize,
    )
    img = ax.contourf(xx, yy, ll, levels=100)
    ax.contour(
        xx, yy, ll, levels=20, colors="black", linestyles="solid", linewidths=0.51
    )
    ax.quiver(
        xx[::reduce, ::reduce],
        yy[::reduce, ::reduce],
        dxx[::reduce, ::reduce],
        dyy[::reduce, ::reduce],
    )
    if fig is not None:
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax1, orientation="vertical")
        return fig, ax
    return ax


def make_training_animation(
    save_path,
    dpi=150,
    fps=60,
    max_frames=None,
    fig=None,
    ax=None,
    name="default",
    **kwargs_grad_plot,
):
    save_name = name
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
    if max_frames is not None:
        max_frames = np.minimum(max_frames, len(epochs))
        epochs = epochs[:max_frames]
    try:
        inputs = np.load(os.path.join(save_path, "inputs.npy"))
    except Exception as e:
        raise e
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, figsize=(15, 15), dpi=dpi)

    maxl = inputs.max(0)
    minl = inputs.min(0)
    x_lim = (minl[0], maxl[0])
    y_lim = (minl[1], maxl[1])
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    def plotter_grad(i, ax=ax):
        print(f"Processing frame {i}")
        ax.clear()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        grad = np.load(
            os.path.join(save_path, str(epochs[i]) + "_" + name + "_grad.npy")
        )
        if "energy" in types:
            energy = np.load(
                os.path.join(save_path, str(epochs[i]) + "_" + name + "_energy.npy")
            )
        else:
            energy = None
        ax = make_grad_plot(
            grad=grad, e=energy, xy=inputs, ax=ax, iter=i, **kwargs_grad_plot
        )

    # fig.tight_layout()

    anim = animation.FuncAnimation(fig, plotter_grad, frames=len(epochs) - 1)
    anim.save(os.path.join(save_path, save_name + "_animation.gif"), fps=fps, dpi=dpi)


def plot_trajectories(
    ebm=None,
    trajectories=None,
    fig=None,
    ax=None,
    x_lim=(-10, 10),
    save_path=None,
    name="default",
    **kwargs_grad_plot,
):
    assert ebm is not None or trajectories is not None
    if trajectories is None:
        trajectories = ebm.langevin_dynamics(trajectories=True, n_samples=500)
    else:
        assert len(trajectories.shape) == 3

    if ax is None or fig is None:
        print(f"axis is {ax} and fig is {fig}")
        fig, ax = plt.subplots(1, figsize=(10, 10))

    def plot_traj(i):
        ax.clear()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*x_lim)
        _ = make_grad_plot(ebm, x_lim, x_lim, ax=ax, iter=i, **kwargs_grad_plot)
        ax.scatter(trajectories[i, :, 0], trajectories[i, :, 1], s=5, color="black")

    anim = mpl.animation.FuncAnimation(fig, plot_traj, trajectories.shape[0])
    if save_path is not None:
        anim.save(os.path.join(save_path, name + "_animation.gif"), fps=60, dpi=150)
    return fig, ax, anim
