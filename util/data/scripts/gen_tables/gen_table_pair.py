import data.scripts.common.help_iso as hi
import data.scripts.common.constants as const
import data.scripts.gen_tables.gen_pair_configs as intra_pair_config
import data.scripts.gen_tables.gen_inter_configs as inter_pair_config

import argparse
import os
import pandas as pd
import re
import numpy as np
import sys

args = None


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

    parser.add_argument('--how', choices=['smk', 'dynamic', 'inter', 'ctx'],
                        default='dynamic',
                        help='How to partition resources between benchmarks.')

    parser.add_argument('--multi',
                        action='store_true',
                        help='Whether we are dealing with the new '
                             'multi-kernel simulation file format.')

    parser.add_argument('--cap',
                        type=float,
                        default=2.5,
                        help='Fail fast simulation: cap runtime at n times '
                             'the longer kernel. Default is 2.5x.')

    parser.add_argument('--qos',
                        default=0.5,
                        type=float,
                        help='Quality of Service for each benchmark '
                             'in terms of normalized IPC.')

    global args
    args = parser.parse_args()


def preprocess_df_pair(df_pair):
    df_pair = df_pair.copy()
    print(df_pair.columns)

    df_pair.reset_index(inplace=True, drop=True)

    # Extract multi-kernel information, if any
    hi.process_config_column('1_kidx', '2_kidx', df=df_pair)

    # avg mem bandwidth
    df_pair['avg_dram_bw'] = df_pair['dram_bw'].transform(hi.avg_array)

    # split pair into two benchmarks
    pair = [re.split(r'-(?=\D)', p) for p in df_pair['pair_str']]
    df_bench = pd.DataFrame(pair, columns=['1_bench', '2_bench'])
    df_pair = pd.concat([df_bench, df_pair], axis=1)

    # extract resource allocation size
    if args.how == 'dynamic':
        hi.process_config_column('1_intra', '2_intra', df=df_pair)

        def intra_conc_cta(idx):
            def grid_size(row):
                return const.get_grid_size(row[idx + '_bench'],
                                           row[idx + '_kidx'])

            df_pair[idx + '_conc_cta'] = np.minimum(
                const.num_sm_volta * df_pair[idx + '_intra'],
                df_pair.apply(grid_size, axis=1)
            )

        intra_conc_cta('1')
        intra_conc_cta('2')
    elif args.how == 'inter':
        hi.process_config_column('1_inter', '2_inter', df=df_pair)

        def inter_conc_cta(idx):
            df_pair[idx + '_conc_cta'] = np.minimum(
                df_pair[idx + '_inter'] *
                df_pair[idx + '_bench'].apply(const.get_max_cta_per_sm),
                df_pair[idx + '_bench'].apply(const.get_grid_size)
            )

        inter_conc_cta('1')
        inter_conc_cta('2')
    elif args.how == 'ctx':
        hi.process_config_column('1_ctx', '2_ctx', df=df_pair)

        def cta_quota(row):
            cta_quota = [[], [], []]
            cta_quota[1] = const.calc_cta_quota(row['1_bench'], row['1_ctx'])
            cta_quota[2] = const.calc_cta_quota(row['2_bench'], row['2_ctx'])
            return cta_quota

        df_pair['cta_quota'] = df_pair.apply(cta_quota, axis=1)

    # Parse per stream per kernel information
    if args.multi:
        df_pair['runtime'] = df_pair['runtime'].apply(hi.parse_multi_array)
        df_pair['instructions'] = df_pair['instructions'].apply(
            hi.parse_multi_array)
        df_pair['l2_bw'] = df_pair['l2_bw'].apply(hi.parse_multi_array)
        df_pair['avg_mem_lat'] = df_pair['avg_mem_lat'].apply(
            hi.parse_multi_array)

    return df_pair


def evaluate_single_kernel(df_pair, df_baseline):
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


def evaluate_multi_kernel(df_pair, df_baseline):
    df_baseline.set_index(['pair_str', '1_kidx'], inplace=True)

    # def handle_incomplete(row):
    #     runtime_adj = row['runtime']
    #     tot_runtime = row['tot_runtime']
    #     # list_time is in the format of [[], [k1, k2, ...], [k1, k2, ...]]
    #     # First, we search if the last value in stream is zero
    #     # Then we replace it with (tot_runtime - sum of previous)
    #     for idx, stream in enumerate(row['runtime']):
    #         if len(stream) == 0:
    #             continue
    #
    #         if stream[-1] == 0:
    #             infer_runtime = tot_runtime - sum(stream)
    #             done_inst = row['instructions'][idx][-1]
    #             tot_inst = df_baseline.loc[(row['{}_bench'.format(idx)],
    #                                         len(stream)-1), 'instructions']
    #             runtime_adj[idx][-1] = tot_runtime - sum(stream)
    #
    #     assert(all(row['runtime'] > 0))
    #     return row['runtime']

    # df_pair['runtime_adj'] = df_pair.apply(handle_incomplete, axis=1)

    # calculate normalized runtime
    if args.how == 'ctx':
        def calc_norm_runtime(row):
            runtime = row['runtime']
            norm_runtime = []
            for stream_id, stream in enumerate(runtime):
                norm_stream = []
                if len(stream) > 0:
                    bench = row['{}_bench'.format(stream_id)]
                    num_kernels = const.get_num_kernels(bench)
                    for kidx, time in enumerate(stream):
                        kidx = kidx % num_kernels
                        base_time = df_baseline.loc[
                            (bench, kidx + 1), 'runtime']
                        norm_stream.append(base_time / time)

                norm_runtime.append(norm_stream)

            return norm_runtime

        def collect_baseline(row):
            runtime = row['runtime']
            baseline = []
            for stream_id, stream in enumerate(runtime):
                base_stream = []
                if len(stream) > 0:
                    bench = row['{}_bench'.format(stream_id)]
                    for kidx in const.kernel_yaml[bench]:
                        base_stream.append(
                            df_baseline.loc[(bench, kidx), 'runtime']
                        )
                baseline.append(base_stream)

            return baseline

        def calculate_total_sld(row):
            runtime = np.array(row['runtime'])
            sld = []

            # 1. Find out which stream has a shorter runtime (single iteration)
            sum_runtime = np.array([sum(arr_run) for arr_run in runtime])
            min_runtime = np.min(sum_runtime[sum_runtime > 0])

            # 2. For every stream, find the max number of full iterations
            # executed before the shortest stream ended
            for stream_id, stream in enumerate(runtime):
                sld_stream = 0
                stream = np.array(stream)

                if len(stream) > 0:
                    bench = row['{}_bench'.format(stream_id)]
                    num_kernels = len(const.kernel_yaml[bench])
                    num_iters = int(len(stream) / num_kernels)
                    tot_time = sum(stream)

                    while tot_time > min_runtime:
                        num_iters -= 1
                        tot_time = sum(stream[0:num_iters*num_kernels])

                    iter_time = sum(row['baseline'][stream_id])
                    sld_stream = num_iters * iter_time / tot_time

                sld.append(sld_stream)

            return sld

        df_pair['norm_ipc'] = df_pair.apply(calc_norm_runtime, axis=1)
        df_pair['baseline'] = df_pair.apply(collect_baseline, axis=1)
        df_pair['sld'] = df_pair.apply(calculate_total_sld, axis=1)
        df_pair['ws'] = df_pair['sld'].apply(np.sum)

    elif args.how == 'dynamic':
        # calculate 1_sld and 2_sld
        def calc_sld(row):
            runtime_1 = row['runtime'][1][0]
            runtime_2 = row['runtime'][2][0]
            assert (runtime_1 > 0)
            assert (runtime_2 > 0)
            base_1 = df_baseline.loc[(row['1_bench'], row['1_kidx']), 'runtime']
            base_2 = df_baseline.loc[(row['2_bench'], row['2_kidx']), 'runtime']

            norm_1 = base_1 / runtime_1
            norm_2 = base_2 / runtime_2

            return [0, norm_1, norm_2]

        df_pair['sld'] = df_pair.apply(calc_sld, axis=1)
        df_pair['ws'] = df_pair['sld'].transform(lambda sld: sld[1] + sld[2])

    return df_pair


def main():
    # Parse arguments
    parse_args()

    # Read CSV file
    df_pair = pd.read_csv(args.csv)
    df_pair.dropna(how="any", inplace=True)

    df_pair = preprocess_df_pair(df_pair)

    # Calculate weighted speedup and fairness w.r.t seq
    df_seq = pd.read_pickle(args.seq_pkl)

    if args.multi:
        df_pair = evaluate_multi_kernel(df_pair, df_seq)
    else:
        df_pair = evaluate_single_kernel(df_pair, df_seq)

    if args.how == 'smk' or args.how == 'ctx':
        df_pair.to_pickle(args.output)
    else:
        # Get profiled info from intra pkl
        if args.multi:
            app_pairs = df_pair[
                ['1_bench', '1_kidx', '2_bench', '2_kidx']
            ].drop_duplicates().values
            app_pairs = [['{0}:{1}'.format(pair[0], pair[1]),
                          '{0}:{1}'.format(pair[2], pair[3])
                          ]
                         for pair in app_pairs]
        else:
            app_pairs = df_pair[['1_bench', '2_bench']].drop_duplicates().values

        if args.how == 'dynamic':
            df_profiled = [
                intra_pair_config.build_df_prod(
                    args.isolated_pkl, args.qos, app, args.cap)
                for app in app_pairs]

        elif args.how == 'inter':
            df_profiled = [
                inter_pair_config.build_df_prod(
                    args.isolated_pkl, app, args.cap)
                for app in app_pairs]
        else:
            sys.exit(1)

        df_profiled = pd.concat(df_profiled, axis=0, ignore_index=True)

        # Join table with profiled info and predicted performance
        df_join = pd.merge(df_pair, df_profiled,
                           how='left',
                           left_on=['1_bench', '2_bench', 'config'],
                           right_on=['pair_str_x', 'pair_str_y', 'config'])

        # Output pickle
        df_join.to_pickle(args.output)


if __name__ == "__main__":
    main()
