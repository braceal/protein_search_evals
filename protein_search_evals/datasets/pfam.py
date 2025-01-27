"""Pfam dataset."""

from __future__ import annotations

import subprocess
from pathlib import Path


def download_and_unzip(url: str, output_dir: str | Path) -> Path:
    """
    Download and unzip a gzipped file from the specified URL.

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

    # Download the file using curl with resume support
    command = f'curl -C - -O {url}'
    subprocess.run(command.split(), cwd=output_dir_path, check=True)

    # Unzip the file using gunzip
    command = f'gunzip -v {gz_file_path}'
    subprocess.run(command.split(), check=True)

    return output_file_path


def download_pfam(output_dir: str | Path) -> None:
    """
    Download the Pfam37.0 dataset.

    The Pfam37.0 dataset is used because it is the latest version
    containing the pfaamseq file. Later versions do not contain this file.

    The following files will be downloaded and unzipped:
    - Pfam.version.gz
        - Contains the Pfam version. The file contents are:
            TODO

    - Pfam-A.fasta.gz
        - Contains the Pfam domains with format:
            >{uniprot_id}/{start}-{end} {uniprot_id} {pfam_id};{pfam_name};
            {sequence}

            Where start and end are the sequence residue positions of the
            domain in the UniProt sequence with id `uniprot_id`.

            Example:
            >A0A7L1FGH7_SYLBO/154-189 A0A7L1FGH7.1 PF10417.14;1-cysPrx_C;
            AFQYTDKHGEVCPAGWKPGSETIIPDPAGKLKYFDK

    - pfamseq.gz
        TODO

    Parameters
    ----------
    output_dir : str or Path
        The directory where the downloaded and unzipped files will be stored.
    """
    # Define the URLs for the Pfam version and sequences
    urls = [
        'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/Pfam.version.gz',
        'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/Pfam-A.fasta.gz',
        'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/pfamseq.gz',
    ]

    # Download and unzip the files
    for url in urls:
        print(f'Downloading and unzipping: {url}')
        output_file = download_and_unzip(url, output_dir)
        print(f'Downloaded and unzipped: {output_file}')


# TODO: Make function or class to download the version and relevant files
# (give option for current version). The function can return the
# Sequence objects. Maybe it's a good idea to also organize the data by
# families, to provide those splits with logic encapsulated in this module.

if __name__ == '__main__':
    download_pfam('data')
