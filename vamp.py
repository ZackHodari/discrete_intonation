from collections import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from morgana.base_models import BaseVAE
from morgana import data
from morgana import utils

import vae


def log_normal(z, mean, log_variance):
    return torch.sum(-0.5 * (log_variance + (z - mean) ** 2 / torch.exp(log_variance)), dim=-1)


class _BaseVAMPPriorVAE(BaseVAE):
    def __init__(self, z_dim, kld_weight, input_dim):
        r"""Initialises GMM-VAE parameters and settings."""
        super(_BaseVAMPPriorVAE, self).__init__(z_dim=z_dim, kld_weight=kld_weight)
        self.input_dim = input_dim

        self._pseudo_inputs = None
        self.n_components = None
        self.pseudo_inputs_seq_lens = None
        self.max_pseudo_inputs_seq_len = None
        self.pseudo_input_names = None

    def get_pseudo_inputs(self):
        device = self._pseudo_inputs[0].device
        _pseudo_inputs = torch.zeros((self.n_components, self.max_pseudo_inputs_seq_len, self.input_dim), device=device)
        for i, seq_len in enumerate(self.pseudo_inputs_seq_lens):
            _pseudo_inputs[i, :seq_len] = self._pseudo_inputs[i]
        return _pseudo_inputs

    def set_pseudo_inputs(self, pseudo_inputs, requires_grad=True):
        class ParameterList(nn.ParameterList):
            def __repr__(self):
                return self._get_name() + f'(len={len(self)}, requires_grad={requires_grad})'

        self._pseudo_inputs = ParameterList(
            [nn.Parameter(pseudo_input, requires_grad=requires_grad) for pseudo_input in pseudo_inputs])

        self.n_components = len(pseudo_inputs)
        self.pseudo_inputs_seq_lens = torch.tensor([pseudo_input.shape[0] for pseudo_input in pseudo_inputs])
        self.max_pseudo_inputs_seq_len = max(self.pseudo_inputs_seq_lens)

        if self.pseudo_input_names is None:
            self.pseudo_input_names = [f'pseudo_input_{i}' for i in range(self.n_components)]

    # Must be a lambda to allow for getter and setter to be re-implemented in subclasses.
    pseudo_inputs = property(fget=lambda self: self.get_pseudo_inputs(),
                             fset=lambda self, *args, **kwargs: self.set_pseudo_inputs(*args, **kwargs))

    def KL_divergence(self, latent, mean, log_variance):
        # Use batch dimension to process all components at once.
        prior_mean, prior_log_variance = self.encoder_layer(self.pseudo_inputs, seq_len=self.pseudo_inputs_seq_lens)

        # For all dimensions (e.g. batch and phrase) not including the latent we must unsqueeze for broadcasting.
        for _ in range(latent[:-1].ndim):
            prior_mean = prior_mean.unsqueeze(dim=1)
            prior_log_variance = prior_log_variance.unsqueeze(dim=1)

        # Calculate the prior log-likelihood, normalise by the number of components, and log-sum-exp over components.
        prior_log_likelihood = log_normal(latent.unsqueeze(dim=0), prior_mean, prior_log_variance)
        prior_log_likelihood -= np.log(self.n_components)
        prior_log_likelihood = prior_log_likelihood.logsumexp(dim=0)

        # Calculate the approximate posterior log-likelihood.
        approximate_posterior_log_likelihood = log_normal(latent, mean, log_variance)

        KL = -(prior_log_likelihood - approximate_posterior_log_likelihood)
        return torch.mean(KL)


class VAMPPriorVAE(_BaseVAMPPriorVAE):
    def __init__(self, z_dim, kld_weight, input_dim,
                 n_components, pseudo_inputs_seq_lens, pseudo_inputs_mean, pseudo_inputs_std):
        r"""Initialises GMM-VAE parameters and settings."""
        super(VAMPPriorVAE, self).__init__(z_dim, kld_weight, input_dim)
        self.init_pseudo_inputs(n_components, pseudo_inputs_seq_lens, pseudo_inputs_mean, pseudo_inputs_std)

    def get_pseudo_inputs(self):
        _pseudo_inputs = super(VAMPPriorVAE, self).get_pseudo_inputs()
        return F.hardtanh(_pseudo_inputs, min_val=0.0, max_val=1.0)

    def init_pseudo_inputs(self, n_components, pseudo_inputs_seq_lens, pseudo_inputs_mean, pseudo_inputs_std):
        if not isinstance(pseudo_inputs_seq_lens, Iterable):
            pseudo_inputs_seq_lens = [pseudo_inputs_seq_lens] * n_components
        if n_components != len(pseudo_inputs_seq_lens):
            raise ValueError(f'Number of components ({n_components}) must match the '
                             f'length of pseudo_inputs_seq_lens ({len(pseudo_inputs_seq_lens)})')

        if not isinstance(pseudo_inputs_mean, Iterable):
            pseudo_inputs_mean = [
                torch.full((pseudo_inputs_seq_lens[i], self.input_dim), pseudo_inputs_mean)
                for i in range(n_components)]
        if not isinstance(pseudo_inputs_std, Iterable):
            pseudo_inputs_std = [
                torch.full((pseudo_inputs_seq_lens[i], self.input_dim), pseudo_inputs_std)
                for i in range(n_components)]

        init = [torch.normal(pseudo_inputs_mean[i], pseudo_inputs_std[i]) for i in range(n_components)]
        self.pseudo_inputs = init


class VAMPPriorDataVAE(_BaseVAMPPriorVAE):
    def __init__(self, z_dim, kld_weight, input_dim, input_names):
        super(VAMPPriorDataVAE, self).__init__(z_dim, kld_weight, input_dim)
        self.input_names = utils.listify(input_names)

    def set_pseudo_inputs(self, pseudo_inputs):
        super(VAMPPriorDataVAE, self).set_pseudo_inputs(pseudo_inputs, requires_grad=False)

    def init_pseudo_inputs(self, pseudo_inputs_loader):
        pseudo_input_names = []
        pseudo_inputs = []
        for pseudo_feature in pseudo_inputs_loader:
            pseudo_input_names.extend(pseudo_feature['name'])
            pseudo_inputs.append(
                torch.cat([pseudo_feature[input_name].squeeze(0) for input_name in self.input_names], dim=1))

        self.pseudo_input_names = pseudo_input_names
        self.pseudo_inputs = pseudo_inputs

    def load_parameters(self, checkpoint_path, device=None):
        state_dict = super(VAMPPriorDataVAE, self).load_parameters(checkpoint_path, strict=False, device=device)

        _pseudo_inputs = [v for k, v in state_dict.items() if k.startswith('_pseudo_inputs')]
        self.pseudo_inputs = _pseudo_inputs


class VAMPPriorExperimentBuilder(vae.VAEExperimentBuilder):

    @classmethod
    def add_args(cls, parser):
        super(VAMPPriorExperimentBuilder, cls).add_args(parser)

        parser.add_argument("--use_data_as_pseudo_inputs",
                            dest="use_data_as_pseudo_inputs", action="store_true", default=False,
                            help="If True, will use real data samples as the pseudo input modes.")
        parser.add_argument("--pseudo_inputs_id_list",
                            dest="pseudo_inputs_id_list", action="store", type=str, default='pseudo_inputs_id_list.scp',
                            help="File name in --train_dir containing basenames of pseudo-inputs (using real data).")
        parser.add_argument("--pseudo_inputs_dir",
                            dest="pseudo_inputs_dir", action="store", type=str, default='train',
                            help="Name of the sub-directory in --data_root containing pseudo inputs data.")

    def __init__(self, model_class, experiment_name, **kwargs):
        self.use_data_as_pseudo_inputs = kwargs['use_data_as_pseudo_inputs']
        self.pseudo_inputs_id_list = kwargs['pseudo_inputs_id_list']
        self.pseudo_inputs_dir = kwargs['pseudo_inputs_dir']

        super(VAMPPriorExperimentBuilder, self).__init__(model_class, experiment_name, **kwargs)

        if self.use_data_as_pseudo_inputs:
            train_data_sources = self.model.train_data_sources()
            pseudo_inputs_dataset = data.FilesDataset(
                train_data_sources, self.pseudo_inputs_dir, self.pseudo_inputs_id_list,
                self.model.normalisers, self.data_root)
            pseudo_inputs_loader = data.batch(
                pseudo_inputs_dataset, batch_size=1, shuffle=False, device=self.device)

            self.model.init_pseudo_inputs(pseudo_inputs_loader)

