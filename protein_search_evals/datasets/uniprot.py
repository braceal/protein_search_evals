"""Download the latest UniProt release."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from Bio.SeqRecord import SeqRecord

from protein_search_evals.utils import chunk_fasta_file

# Define the URLs for the latest UniProt release
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
            # Print the file being unzipped
            print(f'Unzipping {path}...')

            # Unzip the file
            subprocess.run(['gunzip', str(path)], check=True)
        else:
            print(f'File {path} does not exist. Skipping unzipping.')


def _uniprot_header_format(record: SeqRecord) -> SeqRecord:
    """Format the header of a UniProt sequence.

    When building the Faiss index, it is convenient to keep the
    sequence header simply as the UniProt accession number to
    make cross-referencing easier. The raw sequence header format
    is as follows for UniProtKB/Swiss-Prot and UniProtKB/TrEMBL:
        >sp|UNIPROT_ID|REST_OF_HEADER
        >tr|UNIPROT_ID|REST_OF_HEADER
    """
    # Split the header into parts
    parts = record.description.split('|')

    # Set the new header to the UniProt ID
    record.id = parts[1]
    record.description = ''

    return record


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

    # Chunk the Swiss-Prot FASTA file
    chunk_fasta_file(
        input_file=args.download_dir / 'uniprot_sprot.fasta',
        output_dir=args.download_dir / 'sprot',
        num_seqs_per_file=50_000,
        header_formatter=_uniprot_header_format,
    )

    # If downloading TrEMBL, chunk the TrEMBL FASTA file as well
    if args.download_trembl:
        # This can take a while (1hr+) due to the size of the file
        chunk_fasta_file(
            input_file=args.download_dir / 'uniprot_trembl.fasta',
            output_dir=args.download_dir / 'trembl',
            num_seqs_per_file=500_000,
            header_formatter=_uniprot_header_format,
        )
