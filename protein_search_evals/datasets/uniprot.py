"""Download the latest UniProt release."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

RELEASE_URL = 'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/RELEASE.metalink'

DOWNLOAD_URLS = [
    'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/reldate.txt',
    'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/README',
    'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/LICENSE',
    'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz',
    'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz',
]


def _download_version(download_dir: Path) -> str:
    """Download the latest UniProt release version."""
    # Set the command to fetch the latest UniProt release
    command = f'wget -c "{RELEASE_URL}"'

    # Execute the command
    subprocess.run(command, shell=True, check=True, cwd=download_dir)

    # Set the command to extract the version from the RELEASE.metalink file
    command = r"sed -n 's:.*<version>\([0-9]\{4\}_[0-9]\{1,2\}\)</version>.*:\1:p' RELEASE.metalink"  # noqa: E501

    # Execute the command
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
        cwd=download_dir,
    )

    # Get the version from the output
    version = result.stdout.strip()

    return version


def download_latest_uniprot(download_dir: Path, download_trembl: bool) -> None:
    """Download the latest UniProt release.

    Parameters
    ----------
    download_dir : Path
        Directory to download the UniProt release.
    download_trembl : bool
        If True, download the UniProt TrEMBL release.
    """
    # Download the latest UniProt release version
    version = _download_version(download_dir)

    # Print the version
    print(f'Latest UniProt release version: {version}')

    # Remove the trembl file from the download URLs if not downloading TrEMBL
    download_urls = DOWNLOAD_URLS if download_trembl else DOWNLOAD_URLS[:-1]

    # Download the files
    for url in download_urls:
        # Set the command to download the file
        command = f'wget -c "{url}"'

        # Execute the command
        subprocess.run(command, shell=True, check=True, cwd=download_dir)

    # Unzip the downloaded files
    download_paths = [
        download_dir / 'uniprot_sprot.fasta.gz',
        download_dir / 'uniprot_trembl.fasta.gz',
    ]

    for path in download_paths:
        # Check if the file exists
        if path.exists():
            # Unzip the file
            subprocess.run(['gunzip', str(path)], check=True)
        else:
            print(f'File {path} does not exist. Skipping unzipping.')


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Download the latest UniProt release.',
    )
    parser.add_argument(
        '--download_dir',
        type=Path,
        default='data/uniprot',
        help='Directory to download the UniProt release.',
    )
    parser.add_argument(
        '--download_trembl',
        action='store_true',
        help='Download the UniProt TrEMBL release.',
    )
    args = parser.parse_args()

    # Create the directory if it doesn't exist
    args.download_dir.mkdir(parents=True, exist_ok=True)

    # Call the function to download the latest UniProt release
    download_latest_uniprot(args.download_dir, args.download_trembl)
