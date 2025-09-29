"""Generate FASTA files for each family."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from protein_search_evals.datasets.pfam import Pfam20Dataset


def main() -> None:
    """Write FASTA files for each family."""
    # Parse arguments from the command line
    parser = ArgumentParser(
        description='Generate FASTA files for each family.',
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        required=True,
        help='The directory containing the dataset.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='The directory to write the FASTA files to.',
    )
    args = parser.parse_args()

    # Load the Pfam20 dataset
    dataset = Pfam20Dataset(args.data_dir)
    sequences = dataset.load_sequences()
    families = dataset.load_clusters()

    # Get a map of the uniprot ids to the sequences
    uniprot_to_sequence = {seq.tag: seq for seq in sequences}

    # Write a fasta file for each family
    for family, uniprot_ids in families.items():
        # Get the sequences for the family
        sequences = [uniprot_to_sequence[x] for x in uniprot_ids]

        # Get the file contents
        file_contents = '\n'.join(
            [f'>{seq.tag}\n{seq.sequence}' for seq in sequences],
        )

        # Write the sequences to a fasta file
        with open(args.output_dir / f'{family}.fasta', 'w') as f:
            f.write(file_contents)


if __name__ == '__main__':
    main()
