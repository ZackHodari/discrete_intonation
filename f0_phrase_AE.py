import os

import torch
import torch.nn as nn

from morgana.base_models import BaseSPSS
from morgana.experiment_builder import ExperimentBuilder
from morgana.metrics import LF0Distortion, TensorHistory, History
from morgana.viz.synthesis import MLPG
from morgana.viz import io
from morgana import data
from morgana import losses
from morgana import utils

from tts_data_tools import data_sources
from tts_data_tools import file_io

import k_means
import viz


class F0_AE(BaseSPSS):
    def __init__(self, z_dim=16, conditioning_dim=70+9, output_dim=1*3, dropout_prob=0.,
                 phone_set_file='data/unilex_phoneset.txt'):
        """Initialises VAE parameters and settings."""
        super(F0_AE, self).__init__()
        self.z_dim = z_dim
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.phone_set_file = phone_set_file

        self.encoder_layer = utils.SequentialWithRecurrent(
            nn.Linear(self.output_dim, 256),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(256, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, self.z_dim)
        )

        self.decoder_layer = utils.SequentialWithRecurrent(
            nn.Linear(self.conditioning_dim + self.z_dim, 256),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(256, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, self.output_dim),
        )

        self.metrics.add_metrics('all',
                                 LF0_RMSE_Hz=LF0Distortion(),
                                 embeddings=TensorHistory(self.z_dim, hidden=True),
                                 name=History(hidden=True),
                                 n_segments=TensorHistory(1, dtype=torch.long, hidden=True),
                                 segment_mean_F0=TensorHistory(1, hidden=True))

    def normaliser_sources(self):
        return {
            'dur': data.MeanVarianceNormaliser('dur'),
            'counters': data.MinMaxNormaliser('counters'),
            'lf0': data.MeanVarianceNormaliser('lf0', use_deltas=True),
        }

    def train_data_sources(self):
        return {
            'n_frames': data_sources.TextSource('n_frames', sentence_level=True),
            'dur': data_sources.TextSource('dur'),
            'phones': data_sources.VocabSource('phones', vocab_file=self.phone_set_file),
            'counters': data_sources.NumpyBinarySource('counters'),
            'lf0': data_sources.NumpyBinarySource('lf0', use_deltas=True),
            'vuv': data_sources.NumpyBinarySource('vuv'),
            'n_segments': data_sources.TextSource('n_segments', sentence_level=True),
            'segment_n_frames': data_sources.TextSource('segment_n_frames'),
        }

    def valid_data_sources(self):
        sources = self.train_data_sources()
        sources['mcep'] = data_sources.NumpyBinarySource('mcep')
        sources['bap'] = data_sources.NumpyBinarySource('bap')

        return sources

    def encode(self, features):
        # Run the encoder.
        embedding, _ = self.encoder_layer(features['normalised_lf0_deltas'], seq_len=features['n_frames'])

        segment_durs = features['segment_n_frames']
        embedding = utils.get_segment_ends(embedding, segment_durs)

        return embedding

    def decode(self, latent, features):
        # Prepare the inputs.
        latents_at_frame_rate = utils.upsample_to_repetitions(latent, features['segment_n_frames'])
        phones_at_frame_rate = utils.upsample_to_repetitions(features['phones'], features['dur']).type(torch.float)
        norm_counters = features['normalised_counters']

        decoder_inputs = torch.cat((latents_at_frame_rate, phones_at_frame_rate, norm_counters), dim=-1)

        # Run the decoder.
        pred_norm_lf0_deltas, _ = self.decoder_layer(decoder_inputs, seq_len=features['n_frames'])

        # Prepare the outputs.
        pred_lf0_deltas = self.normalisers['lf0'].denormalise(pred_norm_lf0_deltas, deltas=True)

        # MLPG to select the most probable trajectory given the delta and delta-delta features.
        pred_lf0 = MLPG(means=pred_lf0_deltas,
                        variances=self.normalisers['lf0'].delta_params['std_dev'] ** 2)

        outputs = {
            'normalised_lf0_deltas': pred_norm_lf0_deltas,
            'lf0_deltas': pred_lf0_deltas,
            'lf0': pred_lf0
        }

        sentence_f0 = torch.exp(features['lf0'])
        segment_f0 = utils.split_to_segments(sentence_f0, features['segment_n_frames'])
        segment_mean_f0 = torch.sum(segment_f0, dim=2) / features['segment_n_frames'].type(segment_f0.dtype)

        self.metrics.accumulate(self.mode,
                                embeddings=(latent, features['n_segments']),
                                name=[features['name']],
                                n_segments=features['n_segments'],
                                segment_mean_F0=(segment_mean_f0, features['n_segments']))

        return outputs

    def predict(self, features):
        embedding = self.encode(features)
        output_features = self.decode(embedding, features)

        output_features['latent'] = embedding

        return output_features

    def loss(self, features, output_features):
        seq_len = features['n_frames']

        mse = losses.mse(output_features['normalised_lf0_deltas'], features['normalised_lf0_deltas'], seq_len)

        self.metrics.accumulate(
            self.mode,
            LF0_RMSE_Hz=(features['lf0'], output_features['lf0'], features['vuv'], seq_len))

        return mse

    def analysis_for_train_epoch(self, out_dir, **kwargs):
        pred_dir = os.path.join(out_dir, 'feats')
        os.makedirs(pred_dir, exist_ok=True)

        embeddings = self.metrics.metrics['embeddings'].result().detach().cpu().numpy()
        names = self.metrics.metrics['name'].result()

        # Names and classes are at a sentence level, change these to segment level for use in the scatter plot.
        n_segments = self.metrics.metrics['n_segments'].result().detach().cpu().numpy().squeeze(1)
        segment_names = [f'{names[i]}_{j}' for i, n_segment in enumerate(n_segments) for j in range(n_segment)]

        segment_mean_F0 = self.metrics.metrics['segment_mean_F0'].result().detach().cpu().numpy().squeeze(1)

        title = out_dir.split('experiments/')[-1]
        for proj in ['PCA', 'tSNE']:
            viz.scatter_plot(
                embeddings, segment_names,
                gradients=segment_mean_F0, gradient_title='Mean phrase F0 (Hz)', projection=proj,
                title=title, out_file=os.path.join(out_dir, f'scatter_{proj}_mean_F0.pdf'))

        n_clusters = kwargs.get('n_clusters', 20)
        k_means.cluster(embeddings, n_clusters, names=segment_names, out_dir=out_dir)

    def analysis_for_valid_batch(self, features, output_features, out_dir, **kwargs):
        super(F0_AE, self).analysis_for_valid_batch(features, output_features, out_dir, **kwargs)
        io.save_batched_seqs(output_features, features['name'], out_dir,
                             seq_len=features['n_frames'],
                             feat_names=['lf0'])

        viz.plot_batch_f0(features, output_features, out_dir=out_dir)
        viz.batch_synth(output_features['lf0'], features['vuv'], features['mcep'], features['bap'],
                        seq_len=features['n_frames'], names=features['name'],
                        out_dir=out_dir, sample_rate=kwargs.get('sample_rate', 16000))

    def analysis_for_test_batch(self, features, output_features, out_dir, **kwargs):
        batch_size = len(features['name'])
        max_n_segments = torch.max(features['n_segments']).item()

        # Oracle encoding as the latent.
        oracle_out_dir = os.path.join(out_dir, 'oracle')
        embedding = self.encode(features)
        oracle_output_features = self.decode(embedding, features)
        super(F0_AE, self).analysis_for_test_batch(features, oracle_output_features, oracle_out_dir, **kwargs)

        # Use k-means cluster centroids.
        if 'n_clusters' in kwargs:

            # If the directory containing cluster centroids is not given, then infer it from the current test directory.
            if 'cluster_dir' in kwargs:
                cluster_dir = kwargs['cluster_dir']
            else:
                cluster_dir = os.path.join(out_dir.replace('/test/', '/train/'), 'k_means', 'clusters')

            all_clusters_features = []
            for i in range(kwargs['n_clusters']):
                cluster_out_dir = os.path.join(out_dir, f'cluster_{i}')

                cluster = file_io.load_bin(f'{cluster_dir}/cluster_{i}.npy').reshape((1, 1, -1))
                cluster = torch.tensor(cluster, device=features['lf0'].device)
                cluster = cluster.repeat(batch_size, max_n_segments, 1)

                mode_features = self.decode(cluster, features)
                all_clusters_features.append(mode_features)

                super(F0_AE, self).analysis_for_test_batch(features, mode_features, cluster_out_dir, **kwargs)

            viz.plot_repeated_batch_f0(features, *all_clusters_features, out_dir=out_dir)


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(F0_AE, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

