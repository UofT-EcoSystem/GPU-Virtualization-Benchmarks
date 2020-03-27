import data.scripts.common.help_iso as hi
import data.scripts.common.constants as const

import argparse
import os
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Generate dataframe pickle for '
                                     'isolation-inter run from csv.')
    parser.add_argument('--csv',
                        default=os.path.join(const.DATA_HOME, 'csv/inter.csv'),
                        help='CSV file to parse')

    parser.add_argument('--out_inter',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/inter.pkl'),
                        help='Output path for the intra dataframe pickle')

    parser.add_argument('--seq',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/seq.pkl'),
                        help='Pickle file for seq run')

    results = parser.parse_args()
    return results


def process_df_inter(df_inter, df_seq):
    df_inter = df_inter.copy()

    df_seq.set_index('pair_str', inplace=True)

    # Process df_inter
    def norm_over_seq(index, metric, value, inverse=True):
        return hi.normalize(df_seq, index, metric, value, inverse)

    # normalized IPC
    df_inter['norm_ipc'] = df_inter.apply(lambda row:
                                          norm_over_seq(row['pair_str'],
                                                        'ipc',
                                                        row['ipc'],
                                                        False),
                                          axis=1)

    # gpusim config
    hi.process_config_column('inter', df=df_inter)

    # avg dram bandwidth
    df_inter['avg_dram_bw'] = df_inter['dram_bw'].transform(hi.avg_array)

    # avg row buffer locality
    df_inter['avg_rbh'] = df_inter['row_buffer_locality']\
        .transform(hi.avg_array)

    # avg dram efficiency
    df_inter['avg_dram_eff'] = df_inter['dram_eff'].transform(hi.avg_array)

    return df_inter


def main():
    # Parse arguments
    args = parse_args()

    # Read CSV file
    df_inter = pd.read_csv(args.csv)

    df_seq = pd.read_pickle(args.seq)

    df_inter = process_df_inter(df_inter, df_seq)

    # Output pickle
    df_inter.to_pickle(args.out_inter)


if __name__ == "__main__":
    main()
