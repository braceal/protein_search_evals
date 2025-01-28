"""Pfam dataset."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
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
            with open(families_path) as f:
                return json.load(f)

        # Load the Pfam families from the Pfam-A.fasta file
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
        with open(families_path, 'w') as f:
            json.dump(families, f)

        return dict(families)


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
