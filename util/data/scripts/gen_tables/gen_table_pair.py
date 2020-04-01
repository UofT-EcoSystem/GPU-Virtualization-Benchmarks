import data.scripts.common.help_iso as hi
import data.scripts.common.constants as const
import data.scripts.gen_tables.gen_pair_configs as intra_pair_config
import data.scripts.gen_tables.gen_inter_configs as inter_pair_config

import argparse
import os
import pandas as pd
import re
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Generate application pair pickle for '
                                     'concurrent run from csv.')
    parser.add_argument('--csv',
                        default=os.path.join(const.DATA_HOME, 'csv/pair.csv'),
                        help='CSV file to parse')

    parser.add_argument('--output',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/pair.pkl'),
                        help='Output path for the pair dataframe pickle')

    parser.add_argument('--seq_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/seq.pkl'),
                        help='Pickle file for seq run')

    parser.add_argument('--isolated_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/intra.pkl'),
                        help='Pickle file for isolated intra/inter run')

    parser.add_argument('--how', choices=['smk', 'dynamic', 'inter'],
                        default='dynamic',
                        help='How to partition resources between benchmarks.')

    parser.add_argument('--random', action='store_true',
                        help='Is baseline using random address mapping.')

    parser.add_argument('--cap',
                        type=float,
                        default=2.5,
                        help='Fail fast simulation: cap runtime at n times '
                             'the longer kernel. Default is 2.5x.')

    parser.add_argument('--qos',
                        default=0.75,
                        type=float,
                        help='Quality of Service for each benchmark '
                             'in terms of normalized IPC.')

    results = parser.parse_args()
    return results


def preprocess_df_pair(df_pair, how):
    df_pair = df_pair.copy()

    # filter out any entries that have zero instructions (failed one way the
    # other)
    df_pair = df_pair[df_pair['1_instructions'] > 0]
    df_pair.reset_index(inplace=True, drop=True)

    # avg mem bandwidth
    df_pair['avg_dram_bw'] = df_pair['dram_bw'].transform(hi.avg_array)

    # split pair into two benchmarks
    pair = [re.split(r'-(?=\D)', p) for p in df_pair['pair_str']]
    df_bench = pd.DataFrame(pair, columns=['1_bench', '2_bench'])
    df_pair = pd.concat([df_bench, df_pair], axis=1)

    # extract resource allocation size
    if how == 'dynamic':
        hi.process_config_column('1_intra', '2_intra', df=df_pair)
    elif how == 'inter':
        hi.process_config_column('1_inter', '2_inter', df=df_pair)

    return df_pair


def evaluate_df_pair(df_pair, df_baseline):
    df_baseline.set_index('pair_str', inplace=True)
    df_pair = df_pair.copy()

    # Special handling: if simulation breaks due to cycle count limit,
    # Runtime per stream might be empty if the kernel did not run to completion.
    # In this case, copy global runtime to the empty runtime per stream and
    # infer IPC from incomplete runs. Label this case in a new column 'infer'
    # = [True/False]
    def handle_incomplete(row):
        result = {}
        for app_id in ['1', '2']:
            result[app_id + '_infer'] = (row[app_id + '_runtime'] == 0)

            runtime = row['runtime'] if result[app_id + '_infer'] \
                else row[app_id + '_runtime']
            result[app_id + '_ipc'] = row[app_id + '_instructions'] / runtime

        return pd.Series(result)

    df_pair = pd.concat([df_pair, df_pair.apply(handle_incomplete, axis=1)],
                        axis=1)

    def get_index(row, bench_id):
        return row[bench_id + '_bench']

    def calculate_metric(col_suffix, metric, invert):
        for app_id in ['1', '2']:
            df_pair[app_id + col_suffix] = \
                df_pair.apply(lambda row:
                              hi.normalize(df_baseline, get_index(row, app_id),
                                           metric,
                                           row[app_id + '_' + metric], invert),
                              axis=1)

    calculate_metric('_sld', 'ipc', invert=False)

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
    df_pair.dropna(how="any", inplace=True)

    df_pair = preprocess_df_pair(df_pair, args.how)

    # Calculate weighted speedup and fairness w.r.t seq
    df_seq = pd.read_pickle(args.seq_pkl)
    df_pair = evaluate_df_pair(df_pair, df_seq)

    if args.how == 'how':
        df_pair.to_pickle(args.output)
    else:
        # Get profiled info from intra pkl
        app_pairs = df_pair[['1_bench', '2_bench']].drop_duplicates().values

        if args.how == 'dynamic':
            df_profiled = [
                intra_pair_config.build_df_prod(
                    args.isolated_pkl, args.qos, app, args.cap)
                for app in app_pairs]

        else:
            df_profiled = [
                inter_pair_config.build_df_prod(
                    args.isolated_pkl, app, args.cap)
                for app in app_pairs]

        df_profiled = pd.concat(df_profiled, axis=0, ignore_index=True)

        # Join table with profiled info and predicted performance
        df_join = pd.merge(df_pair, df_profiled,
                           left_on=['1_bench', '2_bench', 'config'],
                           right_on=['pair_str_x', 'pair_str_y', 'config'])

        # Output pickle
        df_join.to_pickle(args.output)


if __name__ == "__main__":
    main()
