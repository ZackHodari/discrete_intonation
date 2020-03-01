import os

import torch.nn as nn
from tqdm import tqdm

from morgana.experiment_builder import ExperimentBuilder
from morgana import lr_schedules
from morgana import utils
from morgana import _logging

from tts_data_tools import file_io


class VAELayer(nn.Module):
    def __init__(self, shared_layer, shared_output_dim, z_dim):
        super(VAELayer, self).__init__()

        self.shared_layer = shared_layer
        self.shared_output_dim = shared_output_dim
        self.z_dim = z_dim

        self.mu_layer = nn.Linear(self.shared_output_dim, self.z_dim)
        self.logvar_layer = nn.Linear(self.shared_output_dim, self.z_dim)

    def forward(self, inputs, seq_len=None, segment_durs=None):
        encodings, _ = self.shared_layer(inputs, seq_len=seq_len)

        # Select the latents corresponding to the final frame of each segment.
        if segment_durs is None:
            encoded = utils.get_segment_ends(encodings, seq_len[:, None])
            encoded = encoded.squeeze(dim=1)
        else:
            encoded = utils.get_segment_ends(encodings, segment_durs)

        mean = self.mu_layer(encoded)
        log_variance = self.logvar_layer(encoded)

        return mean, log_variance


class VAEExperimentBuilder(ExperimentBuilder):

    @classmethod
    def add_args(cls, parser):
        super(VAEExperimentBuilder, cls).add_args(parser)

        parser.add_argument("--kld_wait_epochs",
                            dest="kld_wait_epochs", action="store", type=int, default=5,
                            help="Number of epochs to wait with the KLD cost at 0.0")
        parser.add_argument("--kld_warmup_epochs",
                            dest="kld_warmup_epochs", action="store", type=int, default=20,
                            help="Number of epochs to increase the KLD cost from 0.0, to avoid posterior collapse.")

    def __init__(self, model_class, experiment_name, **kwargs):
        self.kld_wait_epochs = kwargs['kld_wait_epochs']
        self.kld_warmup_epochs = kwargs['kld_warmup_epochs']

        super(VAEExperimentBuilder, self).__init__(model_class, experiment_name, **kwargs)

    def train_epoch(self, data_generator, optimizer, lr_schedule=None, gen_output=False, out_dir=None):
        self.model.mode = 'train'
        self.model.metrics.reset_state('train')

        loss = 0.0
        pbar = _logging.ProgressBar(len(data_generator))
        for i, features in zip(pbar, data_generator):
            self.model.step = (self.epoch - 1) * len(data_generator) + i + 1

            # Anneal the KL divergence, linearly increasing from 0.0 to the initial KLD weight set in the model.
            if self.kld_wait_epochs != 0 and self.epoch == self.kld_wait_epochs + 1 and self.kld_warmup_epochs == 0:
                self.model.kld_weight = self.model.max_kld_weight
            if self.kld_warmup_epochs != 0 and self.epoch > self.kld_wait_epochs:
                if self.model.kld_weight < self.model.max_kld_weight:
                    self.model.kld_weight += self.model.max_kld_weight / (
                                self.kld_warmup_epochs * len(data_generator))
                    self.model.kld_weight = min(self.model.max_kld_weight, self.model.kld_weight)

            self.model.tensorboard.add_scalar('kl_weight', self.model.kld_weight,
                                              global_step=self.model.step)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            batch_loss, output_features = self.model(features)

            batch_loss.backward()
            optimizer.step()

            # Update the learning rate.
            if lr_schedule is not None and self.lr_schedule_name in lr_schedules.BATCH_LR_SCHEDULES:
                lr_schedule.step()

            loss += batch_loss.item()

            # Update the exponential moving average model if it exists.
            if self.ema_decay:
                self.ema.update_params(self.model)

            # Log metrics.
            pbar.print('train', self.epoch,
                       kld_weight=tqdm.format_num(self.model.kld_weight),
                       batch_loss=tqdm.format_num(batch_loss),
                       **self.model.metrics.results_as_str_dict('train'))

            if gen_output:
                self.model.analysis_for_train_batch(features, output_features,
                                                    out_dir=out_dir, **self.analysis_kwargs)

        if gen_output:
            self.model.analysis_for_train_epoch(out_dir=out_dir, **self.analysis_kwargs)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            file_io.save_json(self.model.metrics.results_as_json_dict('train'),
                              os.path.join(out_dir, 'metrics.json'))

        self.model.mode = ''

        return loss / (i + 1)

    def run_train(self):
        if self.kld_wait_epochs > 0 or self.kld_warmup_epochs != 0:
            self.model.max_kld_weight = self.model.kld_weight
            self.model.kld_weight = 0.0
        super(VAEExperimentBuilder, self).run_train()

