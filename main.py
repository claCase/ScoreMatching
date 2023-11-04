import os
import matplotlib.pyplot as plt
from src import models, utils, plotter
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import (MultivariateNormalTriL,
                                                         Mixture,
                                                         Categorical,
                                                         Logistic,
                                                         Distribution,
                                                         MultivariateNormalDiag)
import argparse

'''mix = Mixture(cat=Categorical(logits=[1, 1]),
              components=[MultivariateNormalDiag([-1., -1.], [.5, .5]),
                          MultivariateNormalDiag([1., 1.], [.5, .5])])

x = mix.sample(3000)
'''

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="EBMDSM")
parser.add_argument("--dataset", default="gaussian_mixture")
parser.add_argument("--epochs", default=400)
parser.add_argument("--batch_size", default=500)
parser.add_argument("--n_samples", default=500)
args = parser.parse_args()
dataset = args.dataset
model_type = args.model
epochs = args.epochs
batch_size = args.batch_size
n_samples = args.n_samples

distr = None
if dataset == "gaussian_mixture":
    x, y, distr = utils.make_circle_gaussian(n_gaussians=4, radius=2.5, sigma=.9, n_samples=n_samples)
    save_path = os.path.join(os.getcwd(), "figures", "Gaussian Mixture")
elif dataset == "spiral":
    x, y = utils.make_spiral_galaxy(n_spirals=4, legnth=2, n_samples=n_samples, noise=0.1)
    save_path = os.path.join(os.getcwd(), "figures", "Spiral")
elif dataset == "cross":
    x, y, distr = utils.make_cross_shaped_distribution(n_samples=n_samples)
    save_path = os.path.join(os.getcwd(), "figures", "Cross")
else:
    raise NotImplementedError("Datasets must be in ['gaussian_mixture', 'spiral', 'cross']")

if not os.path.exists(save_path):
    os.makedirs(save_path)

lim = (-5, 5)
xx, yy, xy = plotter.make_base_points(lim, lim, 200)

if distr is not None:
    fig, ax = plt.subplots(2, figsize=(11, 15))
    ax0 = ax[0]
    ax1 = ax[1]
    ax0.set_box_aspect(1)
    ax1.set_box_aspect(1)
else:
    fig, ax1 = plt.subplots(1, figsize=(10, 10))
    ax1.set_box_aspect(1)

if model_type == "SSM":
    save_path = os.path.join(save_path, "Sliced Score Matching")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = models.SlicedScoreMatching(hidden_layers=(512, 512))
    model.compile("adam")
    saver_clbk, save_path1 = utils.save_output_callback(model, xy, save_path, 3, 300, "SSM")
    model.fit(x, epochs=epochs, batch_size=3000, callbacks=[saver_clbk])
    if distr is not None:
        fig, ax0 = plotter.make_distribution_grad_plot(distr, fig=fig, ax=ax0)
        ax0.scatter(x[:, 0], x[:, 1], color="black", s=5)
    plotter.make_training_animation(save_path1, name="sliced_score_matching", dpi=90, fontsize=20, reduce=10,
                                    fig=fig, ax=ax1)

elif model_type == "EBMSSM":
    save_path = os.path.join(save_path, "Energy Based Sliced Score Matching")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = models.EBMSlicedScoreMatching(hidden_layers=(512, 512))
    model.compile("adam")
    saver_clbk, save_path1 = utils.save_output_callback(model, xy, save_path, 3, 300, "EBMSSM")
    model.fit(x, epochs=epochs, batch_size=500, callbacks=[saver_clbk])
    if distr is not None:
        fig, ax0 = plotter.make_distribution_grad_plot(distr, fig=fig, ax=ax0)
        ax0.scatter(x[:, 0], x[:, 1], color="black", s=5)
    plotter.make_training_animation(save_path1, name="ebm_sliced_score_matching", dpi=90, fontsize=20, reduce=10,
                                    fig=fig, ax=ax1)
elif model_type == "EBMDSM":
    save_path = os.path.join(save_path, "Energy Based Denoising Score Matching")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = models.EBMNoiseConditionalScoreModel(hidden_layers=[512, 100])
    o = model(x)
    model.compile("adam")
    saver_clbk, save_path1 = utils.save_output_callback(model, xy, save_path, 3, 300, "EBM-Denoising")
    model.fit(x, epochs=epochs, batch_size=1000, callbacks=[saver_clbk])
    if distr is not None:
        ax0 = plotter.make_distribution_grad_plot(distr, lim, lim, ax=ax0, fontsize=20)
        ax0.scatter(x[:100, 0], x[:100, 1], color="black")
    fig.subplots_adjust(bottom=0.06, top=0.96)
    plotter.make_training_animation(save_path1, name="ebm_de-noising_animation", dpi=90, fontsize=20, reduce=10,
                                    fig=fig, ax=ax1)
elif model_type == "DSM":
    save_path = os.path.join(save_path, "Denoising Score Matching")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = models.NoiseConditionalScoreModel(hidden_layers=[512, 100])
    o = model(x)
    model.compile("adam")
    saver_clbk, save_path1 = utils.save_output_callback(model, xy, save_path, 3, 300, "EBM-Denoising")
    model.fit(x, epochs=epochs, batch_size=1000, callbacks=[saver_clbk])
    if distr is not None:
        ax0 = plotter.make_distribution_grad_plot(distr, lim, lim, ax=ax0, fontsize=20)
        ax0.scatter(x[:100, 0], x[:100, 1], color="black")
    fig.subplots_adjust(bottom=0.06, top=0.96)
    plotter.make_training_animation(save_path1, name="ebm_de-noising_animation", dpi=90, fontsize=20, reduce=10,
                                    fig=fig, ax=ax1)
