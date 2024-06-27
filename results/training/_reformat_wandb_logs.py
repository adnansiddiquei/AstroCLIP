"""
THIS SCRIPT CAN BE IGNORED.

This script was used to reformat the original WandB csv export to an easier and clearer format for the analysis
in the Jupyter notebook. This script has already been run on the csv files in this directory.
"""

import pandas as pd
import argparse
import re


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Which file to reformat.',
    )

    args = parser.parse_args()
    file = args.file

    data = pd.read_csv(file)

    cols_to_drop = [col for col in data.columns if 'MIN' in col or 'MAX' in col]
    data = data.drop(columns=cols_to_drop)

    pattern = re.compile(r'(8|16|32|64|128|256|512)-dim')

    rename_dict = {
        col: pattern.search(col).group(0) for col in data.columns if pattern.search(col)
    }

    # Rename the columns
    data = data.rename(columns=rename_dict)

    # sort the columns
    cols = list(data.columns)
    sorted_cols = sorted(cols[1:], key=lambda x: int(re.search(r'\d+', x).group()))
    data = data[['Step'] + sorted_cols]

    data.to_csv(file, index=False)


if __name__ == '__main__':
    main()
