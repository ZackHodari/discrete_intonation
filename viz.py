from collections import OrderedDict
import itertools
import logging
import math
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.signal import savgol_filter
import torch

from morgana import utils
from tts_data_tools import file_io
from tts_data_tools.wav_gen import world
from tts_data_tools.utils import make_dirs


logger = logging.getLogger('morgana')

FRAME_SHIFT_MS = 5


def _fig_width(n_frames):
    return int(min(2000, n_frames) * 1.75 / 50.)


def batch_synth(lf0, vuv, mcep, bap, seq_len=None, names=None, out_dir=None, sample_rate=16000):
    if out_dir is not None:
        if names is None:
            raise ValueError('If `out_dir` is given, then `names` of individual sentences must also be given')

        synth_dir = os.path.join(out_dir, 'synth')
        make_dirs(synth_dir, names)

    lf0, vuv, mcep, bap = utils.detach_batched_seqs(lf0, vuv, mcep, bap, seq_len=seq_len)

    wavs = []
    for i, name in enumerate(names):
        f0_i = np.exp(lf0[i])
        f0_i = savgol_filter(f0_i, 7, 1) if len(f0_i) >= 7 else f0_i

        wav = world.synthesis(f0_i, vuv[i], mcep[i], bap[i], sample_rate=sample_rate)
        wavs.append(wav)

        if out_dir is not None:
            wav_path = os.path.join(synth_dir, f'{names[i]}.wav')
            file_io.save_wav(wav, wav_path, sample_rate=sample_rate)

    return wavs


def _plot_vuv(vuv, ax, color='grey'):
    # Group vuv by contiguous segments.
    segment_lengths = [(value, len(list(items_in_group))) for value, items_in_group in itertools.groupby(vuv)]

    # Plot a transparent area over the unvoiced sections.
    frame_index = 0
    for segment_value, segment_length in segment_lengths:
        is_unvoiced = abs(segment_value) < 1e-8
        if is_unvoiced:
            ax.axvspan(frame_index, frame_index + segment_length, alpha=0.1, color=color)

        frame_index += segment_length


def plot_f0(f0, vuv, pred_f0, pred_vuv=None, frame_shift_ms=FRAME_SHIFT_MS, name=None, out_file=None):
    n_frames = max(f0.shape[0], pred_f0.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(_fig_width(n_frames), 4))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$F_0$ (Hz)')

    if name is None and out_file is not None:
        name, _ = os.path.splitext(os.path.basename(out_file))
    if name is not None:
        ax.set_title(name)

    # Plot vuv as a grey transparent area.
    if pred_vuv is not None:
        _plot_vuv(pred_vuv, ax, color='blue')
    _plot_vuv(vuv, ax, color='grey')

    # Plot input F0.
    ax.plot(f0, color='black', label='target')

    # Plot predicted F0.
    ax.plot(pred_f0, color='red', label='prediction')

    # Plot smoothed predicted F0 (using a Savitzky-Golay filter).
    if len(pred_f0) >= 7:
        pred_f0_smoothed = savgol_filter(pred_f0, 7, 1)
        ax.plot(pred_f0_smoothed, color='blue', label='smoothed_prediction')

    ax.set_xticklabels(ax.get_xticks() * frame_shift_ms / 1000.)

    ax.legend()

    # Save the plot.
    if out_file is not None:
        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)


def plot_batch_f0(features, predicted, use_vuv=None, out_dir=None):
    if use_vuv is None:
        use_vuv = 'vuv' in predicted

    if out_dir is not None:
        plots_dir = os.path.join(out_dir, 'plots', 'f0')
        make_dirs(plots_dir, features['name'])

    n_frames = features['n_frames'].cpu().detach().numpy()
    target_f0, target_vuv = utils.detach_batched_seqs(
        torch.exp(features['lf0']), features['vuv'], seq_len=n_frames)

    pred_f0 = utils.detach_batched_seqs(
        torch.exp(predicted['lf0']), seq_len=n_frames)

    if use_vuv:
        pred_vuv = utils.detach_batched_seqs(
            predicted['vuv'] > 0.5, seq_len=n_frames)
    else:
        pred_vuv = [None] * len(pred_f0)

    for i, name in enumerate(features['name']):
        if out_dir is None:
            out_file = None
        else:
            out_file = os.path.join(plots_dir, f'{name}.pdf')

        plot_f0(target_f0[i], target_vuv[i], pred_f0[i], pred_vuv[i], name=name, out_file=out_file)


def plot_repeated_f0(f0, vuv, *repeated_pred_f0, frame_shift_ms=FRAME_SHIFT_MS, name=None, out_file=None):
    n_frames = max(f0.shape[0], *(pred_f0.shape[0] for pred_f0 in repeated_pred_f0))

    fig, ax = plt.subplots(1, 1, figsize=(_fig_width(n_frames), 4))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$F_0$ (Hz)')

    if name is None and out_file is not None:
        name, _ = os.path.splitext(os.path.basename(out_file))
    if name is not None:
        ax.set_title(name)

    # Plot vuv as a grey transparent area.
    _plot_vuv(vuv, ax, color='grey')

    for pred_f0 in repeated_pred_f0:
        if len(pred_f0) >= 7:
            # Plot smoothed predicted F0 (using a Savitzky-Golay filter).
            pred_f0_smoothed = savgol_filter(pred_f0, 7, 1)
            ax.plot(pred_f0_smoothed, color='gray')
        else:
            # Plot predicted F0.
            ax.plot(pred_f0, color='gray')

    # Plot input F0.
    ax.plot(f0, color='black')

    ax.set_xticklabels(ax.get_xticks() * frame_shift_ms / 1000.)

    # Save the plot.
    if out_file is not None:
        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)


def plot_repeated_batch_f0(features, *repeated_predictions, out_dir=None):
    if out_dir is not None:
        plots_dir = os.path.join(out_dir, 'plots', 'repeated_f0')
        make_dirs(plots_dir, features['name'])

    n_frames = features['n_frames'].cpu().detach().numpy()

    target_f0, target_vuv = utils.detach_batched_seqs(
        torch.exp(features['lf0']), features['vuv'], seq_len=n_frames)

    repeated_pred_f0 = utils.detach_batched_seqs(
        *(torch.exp(predicted['lf0']) for predicted in repeated_predictions), seq_len=n_frames)

    for i, name in enumerate(features['name']):
        if out_dir is None:
            out_file = None
        else:
            out_file = os.path.join(plots_dir, f'{name}.pdf')

        plot_repeated_f0(target_f0[i], target_vuv[i],
                         *(pred_f0[i] for pred_f0 in repeated_pred_f0),
                         name=name, out_file=out_file)


def scatter_plot(latents, latent_names, centroids=None, centroid_names=tuple(),
                 classes=None, class_title='', gradients=None, gradient_title='',
                 dims=None, projection='PCA', title='', out_file=None):
    r"""Plots latents and centroids on a scatter plot. Can colour points by various information if given.

    If classes is given then points will be plotted according to colours and shapes given in a legend on the right.

    If gradients is given then points will be plotted according to colours on a colour bar on the left.

    If both are given then gradients is plotted as a coloured ring around the legend-coloured point.

    Parameters
    ----------
    latents : np.ndarray, shape (batch_size, latent_dim)
        Embeddings to plot.
    latent_names : list[str]
        Base names of each embedding.
    centroids : np.ndarray, shape (n_components, latent_dim)
        Embedding of centroids to plot.
    centroid_names : list[str]
        Base names of each centroid.
    classes : list[str], optional
        Class that each embedding belongs to.
    class_title : str, optional
        Name of the class type being plotted.
    gradients : list[float], optional
        Real valued number to be plotted as a gradient on a colour bar.
    gradient_title : str, optional
        Name of the numbers being represented using the colour bar.
    dims : int or list[int] or slice, optional
        Dimensions of the latent space to plot. If `None`, all dimensions are represented (projected to 2 dimensions).
    projection : {'PCA', 'tSNE'}, optional
        Type of projection to perform if the latent space used is more than 2 dimensions, defaults to tSNE.
    title : str, optional
        Title to display at the top of the scatter plot.
    out_file : str, optional
        Full path of the location to save the scatter plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    scatter_paths : list[`~matplotlib.collections.PathCollection`]
        List of path collections, each containing one point in the scatter plot.
    """
    if centroids is None:
        centroids = np.zeros((0, latents.shape[-1]))

    if gradients is not None:
        viridis = plt.cm.get_cmap('viridis')

        gradients = list(gradients)
        vmin, vmax = min(gradients), max(gradients)

    if classes is not None:
        available_classes = list(set(classes))

        colour_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        marker_list = ['o', 'v', '^', '<', '>', 's', 'P', '*', 'X', 'D']

        colours, markers = [], []
        for klass in classes:
            index = available_classes.index(klass)
            colours.append(colour_list[index % len(colour_list)])
            markers.append(marker_list[int(index / len(colour_list)) % len(marker_list)])

    if dims is not None:
        latents = latents[:, dims]
        centroids = centroids[:, dims]

    if latents.shape[1] != 2:
        if projection == 'PCA':
            norm_latents = latents - np.mean(latents, axis=0)
            U, _, _ = np.linalg.svd(norm_latents.T, full_matrices=False)
            U2 = U[:, :2]

            latents = latents @ U2
            centroids = centroids @ U2

        elif projection == 'tSNE':
            from sklearn.manifold import TSNE
            embeddings = np.concatenate((latents, centroids))
            embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

            latents = embeddings_2d[:latents.shape[0]]
            centroids = embeddings_2d[-centroids.shape[0]:]

    fig, ax = plt.subplots(figsize=(8, 6))
    bbox_extra_artists = []

    scatter_paths = []
    for i, (latent, name) in enumerate(zip(latents, latent_names)):
        # If no additional labelling info is provided, plot a plain point.
        if classes is None and gradients is None:
            point = ax.scatter(
                latent[0], latent[1], s=12, picker=5)

        # Plot point with colour of the given class.
        elif classes is not None and gradients is None:
            point = ax.scatter(
                latent[0], latent[1], s=12, c=colours[i], marker=markers[i], label=classes[i], picker=5)

        # Plot point with colour according to the colour map.
        elif classes is None and gradients is not None:
            point = ax.scatter(
                latent[[0]], latent[[1]], s=12, c=[gradients[i]], cmap=viridis, vmin=vmin, vmax=vmax, picker=5)

        # Plot both class and gradient visual information
        elif classes is not None and gradients is not None:
            # Plot cmap-gradient as a ring around the class point.
            ax.scatter(
                latent[[0]], latent[[1]], s=140, c=[gradients[i]], marker='o', cmap=viridis, vmin=vmin, vmax=vmax)
            ax.scatter(
                latent[[0]], latent[[1]], s=80, c=['white'], marker='o', cmap=viridis, vmin=vmin, vmax=vmax)

            # Plot the class as a point with a particular colour and marker shape.
            point = ax.scatter(
                latent[0], latent[1], s=12, c=colours[i], marker=markers[i], label=classes[i], picker=5)

        point.full_name = name
        scatter_paths.append(point)

    for centroid, name in zip(centroids, centroid_names):
        point = ax.scatter(
            centroid[0], centroid[1], s=80, c='k', marker='D', label='pseudo_input', picker=5)
        point.full_name = name
        scatter_paths.append(point)

    # Create legend, but ensure that there are no repeated labels.
    if classes is not None:
        handles, labels = ax.get_legend_handles_labels()
        label_handle_set = OrderedDict(zip(labels, handles))
        ncol = max(1, math.ceil(len(label_handle_set) / 25))
        lgd = ax.legend(label_handle_set.values(), label_handle_set.keys(), title=class_title,
                        loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=ncol)
        bbox_extra_artists.append(lgd)

    # Create colour bar.
    if gradients is not None:
        sm = plt.cm.ScalarMappable(cmap=viridis)
        sm.set_clim(vmin, vmax)
        sm.set_array([])
        cax = inset_axes(ax, width="5%", height="100%", loc='center left',
                         bbox_to_anchor=(-0.09, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        fig.colorbar(sm, cax=cax, label=gradient_title)
        cax.yaxis.set_ticks_position('left')
        bbox_extra_artists.append(cax)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if out_file is not None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        fig.savefig(out_file, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
        plt.close(fig)

    return fig, ax, scatter_paths

