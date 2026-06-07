# data_preparation.py
# Organises the raw downloaded .cdf files into the directory structure
# expected by train.py and evaluate.py.
#
# The database from Dreier et al. has two splits: 'Train' and 'Test'.
# For FSL in this paper, the convention (following the original scripts) is:
#   database 'Test' folder  -> FSL training set  (fewer images, 360 used per class)
#   database 'Train' folder -> FSL test set       (more images, up to 1500 per class)
#
# The downloaded files (from data_download.py) have '/' replaced with '_' in
# their names. This script uses HSI_GT_Source.csv to parse the original paths
# and recover class labels and database split information.
#
# Output structure:
#   output_dir/
#     fsl_train/
#       Rye_Midsummer/  file1.cdf  file2.cdf ...
#       Wheat_H5/       ...
#     fsl_test/
#       Rye_Midsummer/  ...
#       Wheat_H5/       ...
#
# Example:
#   python data_preparation.py \
#       --download_dir  /path/to/downloaded_files \
#       --csv_path      /path/to/HSI_GT_Source.csv \
#       --output_dir    /path/to/organised_data \
#       --max_train     360 \
#       --max_test      1500

import argparse
import os
import shutil
import random
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True,
                        help='Directory containing the flat downloaded .cdf files')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to HSI_GT_Source.csv (from data_access/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to create the organised fsl_train/ and fsl_test/ folders')
    parser.add_argument('--max_train', type=int, default=360,
                        help='Max .cdf files per class for FSL training set')
    parser.add_argument('--max_test', type=int, default=1500,
                        help='Max .cdf files per class for FSL test set')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible train/test sampling')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.csv_path)
    # only .cdf files
    df = df[df['source'].str.endswith('.cdf')].copy()

    # parse class name and database split from original path
    # path format: .../GrainClassData_Aug2020/<Split>/<ClassName>/filename.cdf
    def parse_row(source):
        parts = source.strip().split('/')
        db_split = parts[3]       # 'Test' or 'Train'
        class_name = parts[4]     # e.g. 'Rye_Midsummer'
        filename = parts[5]
        downloaded_name = source.replace('/', '_')
        return db_split, class_name, filename, downloaded_name

    rows = df['source'].apply(parse_row)
    df[['db_split', 'class_name', 'filename', 'downloaded_name']] = pd.DataFrame(
        rows.tolist(), index=df.index
    )

    # FSL train: from database 'Test' split
    # FSL test:  from database 'Train' split
    split_map = {'Test': 'fsl_train', 'Train': 'fsl_test'}
    max_map = {'fsl_train': args.max_train, 'fsl_test': args.max_test}

    classes = df['class_name'].unique()
    print(f'Found {len(classes)} classes: {sorted(classes)}')

    copied = {'fsl_train': 0, 'fsl_test': 0}
    missing = 0

    for db_split, fsl_split in split_map.items():
        subset = df[df['db_split'] == db_split]
        for cls in classes:
            cls_rows = subset[subset['class_name'] == cls]
            files = cls_rows['downloaded_name'].tolist()

            available = [f for f in files if os.path.isfile(os.path.join(args.download_dir, f))]
            if not available:
                print(f'  WARNING: no files found for {cls} ({db_split})')
                continue

            random.shuffle(available)
            selected = available[:max_map[fsl_split]]

            dest_dir = os.path.join(args.output_dir, fsl_split, cls)
            os.makedirs(dest_dir, exist_ok=True)

            for fname in selected:
                src = os.path.join(args.download_dir, fname)
                # use original filename (last segment of downloaded name) as dest name
                # this keeps filenames short and avoids path length issues
                orig_filename = fname.split('_')[-1]  # just the .cdf filename
                dst = os.path.join(dest_dir, fname)   # keep full name to avoid collisions
                shutil.copy2(src, dst)
                copied[fsl_split] += 1

            if len(available) < len(files):
                missing += len(files) - len(available)

    print(f'\nFiles copied:')
    print(f'  fsl_train: {copied["fsl_train"]}')
    print(f'  fsl_test:  {copied["fsl_test"]}')
    if missing:
        print(f'  Missing from download dir: {missing}  (run data_download.py to fetch them)')
    print(f'\nOutput: {args.output_dir}')


if __name__ == '__main__':
    main()
