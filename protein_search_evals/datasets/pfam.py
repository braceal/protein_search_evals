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
    command = f'gunzip {gz_file_path}'
    subprocess.run(command.split(), cwd=output_dir_path, check=True)

    # Clean up gz file
    # gz_file_path.unlink()

    return output_file_path


def download_pfam(output_dir: str | Path) -> None:
    """
    Download the Pfam dataset.

    Parameters
    ----------
    output_dir : str or Path
        The directory where the downloaded and unzipped files will be stored.
    """
    # Define the URLs for the Pfam version and sequences
    version_url = 'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.2/Pfam.version.gz'
    sequences_url = 'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.2/Pfam-A.fasta.gz'

    # Download and unzip the version file
    version_file = download_and_unzip(version_url, output_dir)
    print(f'Version file downloaded and unzipped to: {version_file}')

    # Download and unzip the sequences file
    sequences_file = download_and_unzip(sequences_url, output_dir)
    print(f'Sequences file downloaded and unzipped to: {sequences_file}')


# TODO: Make function or class to download the version and relevant files
# (give option for current version). The function can return the
# Sequence objects. Maybe it's a good idea to also organize the data by
# families, to provide those splits with logic encapsulated in this module.

if __name__ == '__main__':
    download_pfam('data')
