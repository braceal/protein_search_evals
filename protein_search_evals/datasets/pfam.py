"""Pfam dataset."""

from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from collections import defaultdict
from itertools import chain
from pathlib import Path

from distllm.embed.datasets.fasta import read_fasta
from tqdm import tqdm


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
        self.data_dir = Path(data_dir)

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
            download_and_unzip(url, self.data_dir)

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
        families_path = self.data_dir / 'families.json'
        if families_path.exists():
            print(f'Loading Pfam families metadata from {families_path}')
            with open(families_path) as f:
                return json.load(f)

        # Load the Pfam families from the Pfam-A.fasta file
        print('Parsing Pfam families from Pfam-A.fasta...')
        domains = read_fasta(self.data_dir / 'Pfam-A.fasta')

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
    evaluation.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seed: int = 42,
        subset_size: int = 20,
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
        """
        super().__init__(data_dir)
        self.seed = seed
        self.subset_size = subset_size

    def _filter_by_uniprot_ids(
        self,
        families: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        print('Removing uniprot IDs that appear in more than one family...')
        # Count the number of occurrences of each uniprot ID across families
        # A dictionary with the form {uniprot_id: count across families}
        uniprot_counts = Counter(chain.from_iterable(families.values()))
        # A set of unique uniprot IDs that appear in only one family
        unique_uniprot_ids = {k for k, v in uniprot_counts.items() if v == 1}
        print(f'\tNum. IDs before filtering: {len(uniprot_counts)}')

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
        data_dir = self.data_dir / f'pfam{self.subset_size}_seed-{self.seed}'
        families_path = data_dir / 'families.json'
        if families_path.exists():
            print(
                f'Loading Pfam{self.subset_size} families metadata '
                f'from {families_path}',
            )
            with open(families_path) as f:
                return json.load(f)

        # Load the underlying Pfam families metadata from families.json
        families = super().load_families()

        # Filter the families by uniprot IDs to avoid multiple correct answers
        families = self._filter_by_uniprot_ids(families)

        # Remove families with less than subset_size members
        families = self._filter_by_family_size(families, self.subset_size)

        # Randomly select subset_size domains (uniprot IDs) from each family
        families = self._filter_by_random_subset(families, self.subset_size)

        # Save the Pfam{subset_size} families metadata to disk
        print(
            f'Saving Pfam{self.subset_size} families metadata '
            f'to {families_path}',
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(families_path, 'w') as f:
            json.dump(families, f)

        return families


#    def _create_pfam20_sequences(self, families: dict[str, list[str]]) -> None: # noqa
#         """Create the Pfam20 sequence file.

#         Parameters
#         ----------
#         families : dict[str | list[str]]
#             A dictionary where each key is a Pfam family ID
#             (e.g., 'PF10417.14') and the value is a list of
#             UniProt IDs (e.g., ['A0A7L1FGH7.1', ...]) that
#             belong to the family.
#         """
#         # Load the Pfam sequences from the pfamseq file
#         pfamseq = read_fasta(self.data_dir / 'pfamseq')

#         # Build a set of the uniprot IDs in the Pfam20 families
#         uniprot_ids = {x for y in families.values() for x in y}

#         # Parse the uniprot IDs from the Pfam sequences
#         # the format is >{uniprot_id} {uniprot_name} {description}
#         uniprot_ids = {seq.tag.split()[0] for seq in pfamseq}

#         # Save the Pfam20 sequences to disk
#         pfam20_path = self.data_dir / 'pfam20.fasta'
#         print(f'Saving Pfam20 sequences to {pfam20_path}')
#         with open(pfam20_path, 'w') as f:
#             for seq in pfam20:
#                 f.write(f'>{seq.tag}\n{seq.sequence}\n')


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
    evaluation.
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


# TODO: Make function or class to download the version and relevant files
# (give option for current version). The function can return the
# Sequence objects. Maybe it's a good idea to also organize the data by
# families, to provide those splits with logic encapsulated in this module.

if __name__ == '__main__':
    # Set the Pfam dataset path to the data directory
    dataset = PfamDataset('data/pfam')

    # Download the Pfam dataset
    dataset.download()

    # Load the Pfam families metadata
    families = dataset.load_families()
