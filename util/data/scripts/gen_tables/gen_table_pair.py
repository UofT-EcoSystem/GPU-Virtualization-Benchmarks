import common.help_iso as hi
import common.constants as const

import argparse
import os
import pandas as pd
import re
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Generate application pair pickle for concurrent run from csv.')
    parser.add_argument('--csv',
                        default=os.path.join(const.DATA_HOME, 'csv/pair.csv'),
                        help='CSV file to parse')

    parser.add_argument('--output',
                        default=os.path.join(const.DATA_HOME, 'pickles/pair.pkl'),
                        help='Output path for the pair dataframe pickle')

    parser.add_argument('--baseline_pkl',
                        help='Pickle file for baseline run')

    parser.add_argument('--baseline', choices=['seq', 'intra'], help='Type of baseline.')

    parser.add_argument('--smk', default=False, action='store_true',
                        help='Whether we should parse gpusim config.')

    results = parser.parse_args()
    return results


def preprocess_df_pair(df_pair, parse_config):
    df_pair = df_pair.copy()

    # avg mem bandwidth
    df_pair['avg_dram_bw'] = df_pair['dram_bw'].transform(hi.avg_array)

    # split pair into two benchmarks
    pair = [re.split(r'-(?=\D)', p) for p in df_pair['pair_str']]
    df_bench = pd.DataFrame(pair, columns=['1_bench', '2_bench'])
    df_pair = pd.concat([df_bench, df_pair], axis=1)

    # extract resource allocation size
    if parse_config:
        hi.process_config_column('1_intra', '2_intra', '1_l2', '2_l2', df=df_pair)

    return df_pair


def evaluate_df_pair(df_pair, df_baseline, baseline):
    df_pair = df_pair.copy()

    def get_index_seq(row, bench_id):
        return row[bench_id + '_bench']

    def get_index_intra(row, bench_id):
        return row[bench_id + '_bench'], row[bench_id + '_intra'], row[bench_id + '_l2']

    get_index = get_index_seq if baseline == 'seq' else get_index_intra

    def calculate_metric(col_suffix, metric, invert):
        for app_id in ['1', '2']:
            df_pair[app_id + col_suffix] = \
                df_pair.apply(lambda row:
                              hi.normalize(df_baseline, get_index(row, app_id),
                                           metric, row[app_id + '_' + metric], invert),
                              axis=1)

    calculate_metric('_sld', 'runtime', invert=True)

    df_pair['ws'] = df_pair['1_sld'] + df_pair['2_sld']

    df_pair['fairness'] = np.minimum(df_pair['1_sld'] / df_pair['2_sld'],
                                     df_pair['2_sld'] / df_pair['1_sld'])

    calculate_metric('_norm_mflat', 'avg_mem_lat', invert=False)

    return df_pair


def main():
    # Parse arguments
    args = parse_args()

    # Read CSV file
    df_pair = pd.read_csv(args.csv)
    df_pair = preprocess_df_pair(df_pair, not args.smk)

    if args.baseline == 'seq':
        index = 'pair_str'
    else:
        index = ['pair_str', 'intra', 'l2']

    df_baseline = pd.read_pickle(args.baseline_pkl)
    df_baseline.set_index(index, inplace=True)
    df_pair = evaluate_df_pair(df_pair, df_baseline, args.baseline)

    # Output pickle
    df_pair.to_pickle(args.output)


if __name__ == "__main__":
    main()
