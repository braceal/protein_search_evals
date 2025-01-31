"""Pfam dataset."""

from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from collections import defaultdict
from itertools import chain
from pathlib import Path

from tqdm import tqdm

from protein_search_evals.utils import read_fasta
from protein_search_evals.utils import Sequence
from protein_search_evals.utils import write_fasta


def download_and_unzip(url: str, output_dir: str | Path) -> Path:
    """Download and unzip a gzipped file from the specified URL.

    Parameters
    ----------
    url : str
        The URL of the gzipped file to download.
    output_dir : str or Path
        The directory where the downloaded and unzipped file will be stored.

    Returns
    -------
    Path
        The path to the unzipped file.
    """
    # Create the output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Define the paths for the gzipped and unzipped files
    gz_file_path = output_dir_path / Path(url).name
    output_file_path = gz_file_path.with_suffix('')

    # If the file already exists, return the path
    if output_file_path.exists():
        return output_file_path

    # Log the download
    print(f'Downloading and unzipping: {url}')

    # Download the file using curl with resume support
    command = f'curl -C - -O {url}'
    subprocess.run(command.split(), cwd=output_dir_path, check=True)

    # Unzip the file using gunzip
    command = f'gunzip -v {gz_file_path}'
    subprocess.run(command.split(), check=True)

    # Log the unzipped file path
    print(f'Downloaded and unzipped: {output_file_path}')

    return output_file_path


class PfamDataset:
    """Pfam dataset."""

    def __init__(self, data_dir: str | Path):
        """Initialize the Pfam dataset.

        Parameters
        ----------
        data_dir : str or Path
            The directory where the Pfam dataset files will be stored.
        """
        self._data_dir = Path(data_dir)

    def download(self) -> None:
        """Download the Pfam37.0 dataset.

        If the files already exist in the data directory, they will not
        be downloaded again.

        The Pfam37.0 dataset is used because it is the latest version
        containing the pfamseq file. Later versions do not contain this file.

        The following files will be downloaded and unzipped:
        - Pfam.version.gz
            Contains the Pfam version. The file contents are:
                Pfam release       : 37.0
                Pfam-A families    : 21979
                Date               : 2024-03
                Based on UniProtKB : 2023_05

        - Pfam-A.fasta.gz
            Contains the Pfam domains with format:
                >{uniprot_id}/{start}-{end} {uniprot_id} {pfam_id};{pfam_name};
                {sequence}

                Where start and end are the sequence residue positions of the
                domain in the UniProt sequence with id `uniprot_id`.

            Example:
            >A0A7L1FGH7_SYLBO/154-189 A0A7L1FGH7.1 PF10417.14;1-cysPrx_C;
            AFQYTDKHGEVCPAGWKPGSETIIPDPAGKLKYFDK

        - pfamseq.gz
            Contains the sequences of the Pfam families. The format is:
            >{uniprot_id} {uniprot_name} {description}
            {sequence}

            Example:
            >A0A8J7XB61.1 A0A8J7XB61_9EURY lipoate--protein ligase ...
            MEWRLLTLDQKDGYYIQSVYEAVAKA...

        Parameters
        ----------
        output_dir : str or Path
            The directory where the downloaded and unzipped files will
            be stored.
        """
        # Define the URLs for the Pfam version and sequences
        urls = [
            'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/Pfam.version.gz',
            'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/Pfam-A.fasta.gz',
            'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/pfamseq.gz',
        ]

        # Download and unzip the files
        for url in urls:
            download_and_unzip(url, self._data_dir)

    def load_families(self) -> dict[str, list[str]]:
        """Load the Pfam families metadata.

        Will cache the families metadata in the data directory to avoid
        parsing the Pfam-A.fasta file multiple times. The parsed families
        metadata will be stored in a families.json file.

        Returns
        -------
        dict[str | list[str]]
            A dictionary where each key is a Pfam family ID
            (e.g., 'PF10417.14') and the value is a list of
            UniProt IDs (e.g., ['A0A7L1FGH7.1', ...]) that
            belong to the family.
        """
        # Download the Pfam dataset if it hasn't been downloaded
        self.download()

        # Load the Pfam families metadata from the families.json file
        # if it exists
        families_path = self._data_dir / 'families.json'
        if families_path.exists():
            print(f'Loading Pfam families metadata from {families_path}')
            with open(families_path) as f:
                return json.load(f)

        # Load the Pfam families from the Pfam-A.fasta file
        print('Parsing Pfam families from Pfam-A.fasta...')
        domains = read_fasta(self._data_dir / 'Pfam-A.fasta')

        # Parse the Pfam families from the domain descriptions
        families = defaultdict(list)
        for domain in tqdm(domains, desc='Parsing Pfam families'):
            # The tag looks like:
            # '>A0A7L1FGH7_SYLBO/154-189 A0A7L1FGH7.1 PF10417.14;1-cysPrx_C;'
            _, uniprot_id, pfam_id = domain.tag.split()

            # Pfam ID looks like 'PF10417.14;1-cysPrx_C;' before split
            # We only want the ID part, e.g., 'PF10417.14'
            pfam_id = pfam_id.split(';')[0]

            # Add the UniProt ID to the family
            families[pfam_id].append(uniprot_id)

        # Save the families metadata to disk
        print(f'Saving Pfam families metadata to {families_path}')
        with open(families_path, 'w') as f:
            json.dump(families, f)

        return dict(families)

    def load_uniprot_ids_by_length(self, length_cutoff: int) -> set[str]:
        """Load the UniProt IDs for sequences that meet the length cutoff.

        Parameters
        ----------
        length_cutoff : int
            The maximum length of the sequences to include in the dataset.

        Returns
        -------
        set[str]
            A set of UniProt IDs for the sequences that meet the length cutoff.
        """
        # The file is a bit large to load all the sequences at once
        # so we iterate over the file and collect the UniProt IDs
        # that meet the length cutoff
        uniprot_ids = []
        with open(self._data_dir / 'pfamseq') as file:
            # Initialize the sequence buffer and the current UniProt ID
            uid: str | None = None
            seq: list[str] = []

            for line in tqdm(file, desc='Parsing Pfam sequence lengths'):
                # Remove leading/trailing whitespace
                contents = line.strip()

                # If we are starting a new entry, collect the previous entry
                # and reset the sequence buffer
                if contents.startswith('>'):
                    # Collect the previous entry
                    if uid is not None and len(''.join(seq)) <= length_cutoff:
                        uniprot_ids.append(uid)

                    # Extract UniProt ID from the header
                    # (assuming format ">uniprot_id uniprot_name description")
                    parts = contents.split(' ')

                    # Remove '>' from the first part
                    uid = parts[0][1:] if parts else None

                    # Reset sequence buffer
                    seq = []
                else:
                    # Append the sequence contents
                    seq.append(contents)

            # Capture last entry
            if uid is not None and len(''.join(seq)) <= length_cutoff:
                uniprot_ids.append(uid)

        return set(uniprot_ids)

    def load_sequences_by_ids(self, query_ids: set[str]) -> list[Sequence]:
        """Retrieve sequences for a set of UniProt IDs from the Pfam dataset.

        Parameters
        ----------
        query_ids : set[str]
            Set of valid UniProt IDs to match.

        Returns
        -------
        list[Sequence]
            List of Sequence objects containing the Pfam sequences that
            match the query IDs.
        """
        seqs = []
        with open(self._data_dir / 'pfamseq') as file:
            # Initialize the sequence buffer and the current UniProt ID
            uid: str | None = None
            seq: list[str] = []

            for line in tqdm(file, desc='Parsing Pfam sequences'):
                # Remove leading/trailing whitespace
                contents = line.strip()

                # If we are starting a new entry, check if the previous entry
                # was a match, if so collect it, and reset the sequence buffer
                if contents.startswith('>'):
                    if uid in query_ids:
                        seqs.append(Sequence(tag=uid, sequence=''.join(seq)))

                    # Extract UniProt ID from the header
                    # (assuming format ">uniprot_id uniprot_name description")
                    parts = contents.split(' ')

                    # Remove '>' from the first part
                    uid = parts[0][1:] if parts else None

                    # Reset sequence buffer
                    seq = []
                else:
                    # Append the sequence contents
                    seq.append(contents)

            # Capture last entry if it was a match
            if uid in query_ids:
                seqs.append(Sequence(tag=uid, sequence=''.join(seq)))

        return seqs


class PfamSubsetDataset(PfamDataset):
    """PfamSubset dataset.

    This dataset generalizes the recipe outlined in the paper:
    "Nearest neighbor search on embeddings rapidly identifies
    distant protein relations", by Schutze et al. (2022).
    Full text:
    https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2022.1033775/full

    To avoid over-representing large families, the authors picked 20
    sequences from each Pfam family with at least 20 members. This
    dataset is referred to as Pfam20.

    We generalize this recipe by allowing the user to specify the
    number of sequences to select from each family `subset_size`.
    This allows for the creation of datasets with different sizes
    and properties, such as Pfam20, Pfam50, etc.

    We ensure an additional constraint that no selected sequence appears
    in more than one family, to avoid multiple "correct" answers during
    evaluation. We also filter out sequences longer than `length_cutoff`
    residues to avoid very long sequences in the benchmark.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seed: int = 42,
        subset_size: int = 20,
        length_cutoff: int = 1022,
    ):
        """Initialize the Pfam20 dataset.

        Parameters
        ----------
        data_dir : str or Path
            The directory where the Pfam{subset_size} dataset directory
            will be stored.
        seed : int, optional
            The random seed used to randomly pick the `subset_size` domains
            from each family, by default 42.
        subset_size : int, optional
            The number of sequences to select from each family, by default 20.
        length_cutoff : int, optional
            The maximum length of the sequences to include in the dataset,
            by default 1022.
        """
        super().__init__(data_dir)
        self.seed = seed
        self.subset_size = subset_size
        self.length_cutoff = length_cutoff

    @property
    def data_dir(self) -> Path:
        """The directory where the Pfam{subset_size} dataset will be stored."""
        return self._data_dir / f'pfam{self.subset_size}_seed-{self.seed}'

    @property
    def families_path(self) -> Path:
        """The path to the families metadata file."""
        return self.data_dir / 'families.json'

    @property
    def sequences_path(self) -> Path:
        """The path to the sequences file."""
        return self.data_dir / 'sequences.fasta'

    def _filter_by_seq_length(
        self,
        families: dict[str, list[str]],
        length_cutoff: int,
    ) -> dict[str, list[str]]:
        print(f'Removing sequences longer than {length_cutoff} residues...')

        # Compute the number of sequences before size filtering
        num_sequences = sum(len(v) for v in families.values())
        print(f'\tNum. sequences before size filtering: {num_sequences}')

        # Load the sequence lengths for the Pfam sequences (this takes ~3 mins)
        valid_uids = self.load_uniprot_ids_by_length(length_cutoff)

        # Filter the families by the sequence lengths
        families = {
            pfam_id: [x for x in uniprot_ids if x in valid_uids]
            for pfam_id, uniprot_ids in families.items()
        }

        # Compute the number of sequences after size filtering
        num_sequences = sum(len(v) for v in families.values())
        print(f'\tNum. families after size filtering: {num_sequences}')

        return families

    def _filter_by_uniprot_ids(
        self,
        families: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        print('Removing uniprot IDs that appear in more than one family...')

        # Count the number of occurrences of each uniprot ID across families
        # A dictionary with the form {uniprot_id: count across families}
        uniprot_counts = Counter(chain.from_iterable(families.values()))
        print(f'\tNum. IDs before filtering: {len(uniprot_counts)}')

        # A set of unique uniprot IDs that appear in only one family
        unique_uniprot_ids = {k for k, v in uniprot_counts.items() if v == 1}

        # Remove uniprot IDs that appear in more than one family
        families = {
            pfam_id: [x for x in uniprot_ids if x in unique_uniprot_ids]
            for pfam_id, uniprot_ids in families.items()
        }

        print(f'\tNum. unique IDs after filtering: {len(unique_uniprot_ids)}')

        return families

    def _filter_by_family_size(
        self,
        families: dict[str, list[str]],
        subset_size: int,
    ) -> dict[str, list[str]]:
        print(f'Removing families with less than {subset_size} members...')
        print(f'\tNum. families before size filtering: {len(families)}')
        families = {k: v for k, v in families.items() if len(v) >= subset_size}
        print(f'\tNum. families after size filtering: {len(families)}')
        return families

    def _filter_by_random_subset(
        self,
        families: dict[str, list[str]],
        subset_size: int,
    ) -> dict[str, list[str]]:
        print(
            f'Randomly selecting {subset_size} domains (Uniprot IDs) '
            f'from each family to balance the dataset using random seed: '
            f'{self.seed}...',
        )

        # Count the number of uniprot IDs in the Pfam families
        num_uniprot_ids = sum(len(v) for v in families.values())
        print(f'\tNum. Uniprot IDs before filtering: {num_uniprot_ids}')

        # Set the random seed for reproducibility
        rng = random.Random(self.seed)

        # Randomly select `subset_size` domains from each family
        families_subset = {}
        for pfam_id, uniprot_ids in families.items():
            # Shuffle the uniprot IDs in the family
            rng.shuffle(uniprot_ids)
            # Add the selection to the Pfam{subset_size} families
            families_subset[pfam_id] = uniprot_ids[:subset_size]

        # Count the number of uniprot IDs in the Pfam{subset_size} families
        num_uniprot_ids = sum(len(v) for v in families_subset.values())
        print(f'\tNum. Uniprot IDs after filtering: {num_uniprot_ids}')

        return families_subset

    def load_families(self) -> dict[str, list[str]]:
        """Load the Pfam`subset_size` families metadata.

        Create the Pfam{subset_size} dataset by randomly selecting
        `subset_size` domains from each family with at least `subset_size`
        members. Selected domains (UniProt IDs) are guaranteed to not appear
        in more than one family, preventing multiple "correct" answers during
        evaluation.

        Will cache the families metadata in the
        {data_dir}/pfam{subset_size}_seed-{seed} directory to avoid parsing
        the underlying Pfam-A.fasta file multiple times.

        Returns
        -------
        dict[str | list[str]]
            A dictionary where each key is a Pfam family ID
            (e.g., 'PF10417.14') and the value is a list of
            UniProt IDs (e.g., ['A0A7L1FGH7.1', ...]) that
            belong to the family.
        """
        # Load the Pfam{subset_size} families metadata if it's already cached
        if self.families_path.exists():
            print(
                f'Loading Pfam{self.subset_size} families metadata '
                f'from {self.families_path}',
            )
            with open(self.families_path) as f:
                return json.load(f)

        # Load the underlying Pfam families metadata from families.json
        families = super().load_families()

        # Filter the families by uniprot IDs to avoid multiple correct answers
        families = self._filter_by_uniprot_ids(families)

        # Filter the families by the sequence lengths (we don't want very long
        # sequences in the benchmark)
        families = self._filter_by_seq_length(families, self.length_cutoff)

        # Remove families with less than subset_size members
        families = self._filter_by_family_size(families, self.subset_size)

        # Randomly select subset_size domains (uniprot IDs) from each family
        families = self._filter_by_random_subset(families, self.subset_size)

        # Save the Pfam{subset_size} families metadata to disk
        print(
            f'Saving Pfam{self.subset_size} families metadata '
            f'to {self.families_path}',
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.families_path, 'w') as f:
            json.dump(families, f)

        return families

    def load_sequences(self) -> list[Sequence]:
        """Load the Pfam{subset_size} sequences.

        Returns
        -------
        list[Sequence]
            A list of Sequence objects containing the Pfam{subset_size}
            sequences.
        """
        # Load the Pfam{subset_size} sequences if already cached
        if self.sequences_path.exists():
            print(
                f'Loading Pfam{self.subset_size} families metadata '
                f'from {self.sequences_path}',
            )
            return read_fasta(self.sequences_path)

        # Load the Pfam families metadata
        families = self.load_families()

        # Build a set of the uniprot IDs in the Pfam{subset_size} families
        print(
            f'Building a set of the uniprot IDs for the '
            f'selected Pfam{self.subset_size} families...',
        )
        uniprot_ids = {uid for uids in families.values() for uid in uids}

        # Load the Pfam sequences from the pfamseq associated with the
        # uniprot IDs.
        print(f'Collecting Pfam{self.subset_size} sequences...')
        sequences = self.load_sequences_by_ids(query_ids=uniprot_ids)

        # Save the Pfam{subset_size} sequences to disk
        print(
            f'Saving Pfam{self.subset_size} with {len(sequences)} '
            f'sequences to {self.sequences_path}',
        )
        write_fasta(sequences, self.sequences_path)

        return sequences


class Pfam20Dataset(PfamSubsetDataset):
    """Pfam20 dataset.

    This dataset roughly follows the recipe outlined in the paper:
    "Nearest neighbor search on embeddings rapidly identifies
    distant protein relations", by Schutze et al. (2022).
    Full text:
    https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2022.1033775/full

    To avoid over-representing large families, the authors picked 20
    sequences from each Pfam family with at least 20 members. This
    dataset is referred to as Pfam20.

    We ensure an additional constraint that no selected sequence appears
    in more than one family, to avoid multiple "correct" answers during
    evaluation. We also filter out sequences longer than 1022 residues
    to avoid very long sequences in the benchmark.

    Examples
    --------
    >>> # Set the Pfam dataset path to the data directory
    >>> dataset = Pfam20Dataset('data/pfam')

    >>> # Load the Pfam families metadata
    >>> families = dataset.load_families()

    >>> # Load the Pfam sequences
    >>> sequences = dataset.load_sequences()
    """

    def __init__(self, data_dir: str | Path, seed: int = 42) -> None:
        """Initialize the Pfam20 dataset.

        Parameters
        ----------
        data_dir : str or Path
            The directory where the Pfam20 dataset directory will be stored.
        seed : int, optional
            The random seed used to randomly pick the 20 domains from
            each family, by default 42.
        """
        super().__init__(data_dir, seed, subset_size=20)
