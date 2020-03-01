import argparse
import os

import numpy as np
from sklearn.cluster import KMeans

from tts_data_tools import file_io
from tts_data_tools.utils import get_file_ids, make_dirs


def add_arguments(parser):
    parser.add_argument("--embeddings_dir", action="store", dest="embeddings_dir", type=str, required=True,
                        help="Directory of the emebddings.")
    parser.add_argument("--n_clusters", action="store", dest="n_clusters", type=int, required=True,
                        help="Number of clusters for k-means.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids.")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")


def cluster(embeddings, n_clusters, names=None, out_dir=None):
    """Processes wav files in id_list, saves the log-F0 and MVN parameters to files.

    Args:
        embeddings_dir (str): Directory containing the embedding files.
        n_clusters (int): Number of clusters for k-means.
        names (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
    """
    if out_dir is not None:
        if names is None:
            raise ValueError('If `out_dir` is given, then `names` of individual sentences must also be given')

        centres_path = os.path.join(out_dir, 'k_means', 'clusters')
        make_dirs(centres_path, names)

        assignments_path = os.path.join(out_dir, 'k_means', 'cluster_assignments')
        make_dirs(assignments_path, names)

    # Cluster with k-means.
    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)
    cluster_centres = kmeans.cluster_centers_
    cluster_assignments = kmeans.labels_

    # Save the cluster assignments and clusters to files.
    if out_dir is not None:
        cluster_names = [f'cluster_{i}' for i in range(n_clusters)]
        file_io.save_dir(file_io.save_bin, centres_path, cluster_centres, cluster_names, feat_ext='npy')
        file_io.save_dir(file_io.save_txt, assignments_path, cluster_assignments, names, feat_ext='txt')

        counts = np.array([(i, cluster_assignments.reshape(-1).tolist().count(i)) for i in range(n_clusters)])
        file_io.save_txt(counts, f'{assignments_path}_counts.txt')

    return cluster_centres, cluster_assignments


def process(embeddings_dir, n_clusters, id_list, out_dir):
    """Processes wav files in id_list, saves the log-F0 and MVN parameters to files.

    Args:
        embeddings_dir (str): Directory containing the embedding files.
        n_clusters (int): Number of clusters for k-means.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
    """
    file_ids = get_file_ids(id_list=id_list)

    # Load the embeddings.
    embeddings = file_io.load_dir(file_io.load_bin, embeddings_dir, file_ids, feat_ext='npy')
    embeddings = np.array(list(embeddings))

    cluster(embeddings, n_clusters, names=file_ids, out_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Extracts log-F0, V/UV, smoothed spectrogram, and aperiodicity using WORLD and Reaper.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.embeddings_dir, args.n_clusters, args.id_list, args.out_dir)


if __name__ == "__main__":
    np.random.seed(1234567890)
    main()

