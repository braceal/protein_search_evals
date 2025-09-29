"""Compute statistics for the similarity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_statistics(csv_file: Path) -> dict[str, float]:
    """Compute statistics for the similarity analysis."""
    # Read the dataframe
    df = pd.read_csv(csv_file)

    # Get the percent identity values
    data = np.array(df['pident'])

    # Remove the 100% identity entries (self hits)
    data = data[data != 100]

    # Compute the mean, median, std, max, and min of the pident values
    # (rounded to 2 decimal places)
    results = {
        'mean': round(np.mean(data), 2),
        'median': round(np.median(data), 2),
        'std': round(np.std(data), 2),
        'max': round(np.max(data), 2),
        'min': round(np.min(data), 2),
    }

    return results


def main() -> None:
    """Run the statistics computation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=Path, required=True)
    parser.add_argument('--output_file', type=Path, required=True)
    args = parser.parse_args()

    # Get all the csv files in the directory
    csv_files = list(args.csv_dir.glob('*.csv'))

    # Compute the statistics for each csv file
    # The file name is of the format {pfam_id}_blastp.csv
    # So we need to extract the pfam_id from the file name
    results = {f.stem.split('_')[0]: compute_statistics(f) for f in csv_files}

    # Save the results to a json file
    with open(args.output_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
