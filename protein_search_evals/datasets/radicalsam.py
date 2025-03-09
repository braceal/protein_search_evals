"""Radical SAM dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path

from protein_search_evals.utils import read_fasta
from protein_search_evals.utils import Sequence
from protein_search_evals.utils import write_fasta

# The available cluster partitions for the Radical SAM dataset
PARTITIONS = {
    'all': 'All_Clusters_Mapping.txt',
    'mega2': 'Mega-2_Mapping.txt',
    'mega3': 'Mega-3_Mapping.txt',
    'mega4': 'Mega-4_Mapping.txt',
    'mega5': 'Mega-5_Mapping.txt',
}


class RadicalSamDataset:
    """Radical SAM dataset."""

    def __init__(
        self,
        data_dir: str | Path,
        partition: str,
        length_cutoff: int = 1022,
    ) -> None:
        """Initialize the Radical SAM dataset.

        Parameters
        ----------
        data_dir : str or Path
            The directory where the dataset files will be stored.
        partition : str
            The dataset partition to load. Available partitions are:
            'all', 'mega2', 'mega3', 'mega4', 'mega5'.
        length_cutoff : int, optional
            The maximum length of the sequences to include in the dataset,
            by default 1022.
        """
        # Check if the partition is valid
        if partition not in PARTITIONS:
            raise ValueError(
                f'Invalid partition {partition}. '
                f'Available partitions are: {list(PARTITIONS)}',
            )

        self._data_dir = Path(data_dir)
        self.partition = partition
        self.length_cutoff = length_cutoff

        # File containing all sequences from which the partitions are derived
        self._all_seqs_file = self._data_dir / 'RSS.2024_05.UniRef50.fasta'

    @property
    def data_dir(self) -> Path:
        """The directory where the dataset will be stored."""
        return self._data_dir / self.partition

    @property
    def clusters_path(self) -> Path:
        """The path to the clusters file."""
        return self.data_dir / 'clusters.json'

    @property
    def sequences_path(self) -> Path:
        """The path to the sequences file."""
        return self.data_dir / 'sequences.fasta'

    @cached_property
    def uniprot_to_cluster(self) -> dict[str, str]:
        """Map Uniprot IDs to clusters.

        Returns
        -------
        dict[str, str]
            The Uniprot ID to cluster mapping.
        """
        # Load the clusters metadata
        clusters = self.load_clusters()

        # Make a uniprot_id to cluster mapping
        uid_to_cluster = {}
        for cluster, uids in clusters.items():
            for uid in uids:
                assert uid not in uid_to_cluster
                uid_to_cluster[uid] = cluster

        return uid_to_cluster

    def load_clusters(self) -> dict[str, list[str]]:
        """Load the Radical SAM clusters.

        Will cache the clusters in the data directory to avoid
        parsing the input file multiple times. The parsed clusters
        will be stored in a clusters.json file.

        Returns
        -------
        dict[str | list[str]]
            A dictionary where each key is a cluster ID
            (e.g., '1') and the value is a list of UniProt IDs
            (e.g., ['X8GR02', ...]) that belong to the cluster.
        """
        # Load the cluster data if it's already cached
        if self.clusters_path.exists():
            print(f'Loading Radical SAM clusters from {self.clusters_path}')
            with open(self.clusters_path) as f:
                return json.load(f)

        # Select the cluster partition file to load
        cluster_file = self._data_dir / PARTITIONS[self.partition]

        # Load the cluster labels
        print(f'Bulding Radical SAM clusters from {cluster_file}')
        file_content = Path(cluster_file).read_text()

        # Skip the header and split the lines
        lines = file_content.split('\n')[1:]

        # For each sequence, a list containing [UniProt ID, cluster ID]
        data = [line.split('\t')[:2] for line in lines if line.strip()]

        # Group the data by cluster ID
        clusters = defaultdict(list)
        for uniprot_id, cluster_id in data:
            clusters[cluster_id].append(uniprot_id)

        # Filter out uniprot IDs of sequences longer than the cutoff
        print('Filtering sequences by length...')
        sequences = read_fasta(self._all_seqs_file)
        valid_uids = {
            x.tag.split()[0]
            for x in sequences
            if len(x.sequence) <= self.length_cutoff
        }
        clusters = {  # type: ignore[assignment]
            cluster: [x for x in uids if x in valid_uids]
            for cluster, uids in clusters.items()
        }

        # Save the clusters to disk
        print(f'Saving Radical SAM clusters to {self.clusters_path}')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.clusters_path, 'w') as f:
            json.dump(clusters, f)

        return dict(clusters)

    def load_sequences(self) -> list[Sequence]:
        """Load the Radical SAM sequences.

        Returns
        -------
        list[Sequence]
            The benchmark sequences.
        """
        # Load the sequences if already cached
        if self.sequences_path.exists():
            print(f'Loading Radical SAM sequences from {self.sequences_path}')
            return read_fasta(self.sequences_path)

        # Load the sequences
        print(f'Creating Radical SAM sequence partition {self.partition}...')
        sequences = read_fasta(self._all_seqs_file)

        # Keep only the UniProt ID in the tag. The initial format is, e.g.:
        # >A0A011RFL0 1|Accumulibacter regalis.|PF02310,PF04055
        print('Cleaning sequence tags...')
        for seq in sequences:
            seq.tag = seq.tag.split()[0]

        # Keep only the sequences part of the selected benchmark
        print('Filtering sequences by cluster...')
        sequences = [
            seq for seq in sequences if seq.tag in self.uniprot_to_cluster
        ]

        # Save the sequences to disk
        num_clusters = len(set(self.uniprot_to_cluster.values()))
        print(
            f'Saving RadicalSam with {len(sequences)} sequences '
            f'and {num_clusters} clusters to {self.sequences_path}',
        )
        write_fasta(sequences, self.sequences_path)

        return sequences


if __name__ == '__main__':
    # Create the Radical SAM datasets
    data_dir = Path('data/radicalsam')

    # Create the Radical SAM datasets
    RadicalSamDataset(data_dir, partition='all').load_sequences()
    RadicalSamDataset(data_dir, partition='mega2').load_sequences()
    RadicalSamDataset(data_dir, partition='mega3').load_sequences()
    RadicalSamDataset(data_dir, partition='mega4').load_sequences()
    RadicalSamDataset(data_dir, partition='mega5').load_sequences()
