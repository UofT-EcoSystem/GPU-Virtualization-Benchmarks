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

    parser.add_argument('--how', choices=['smk', 'dynamic', 'inter', 'ctx',
                                          'lut'],
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
    df_pair.reset_index(inplace=True, drop=True)

    # Extract multi-kernel information, if any
    hi.process_config_column('1_kidx', '2_kidx', df=df_pair, default=1)

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
    # if args.multi:
    df_pair['runtime'] = df_pair['runtime'].apply(hi.parse_multi_array)
    df_pair['instructions'] = df_pair['instructions'].apply(
        hi.parse_multi_array)
    df_pair['l2_bw'] = df_pair['l2_bw'].apply(hi.parse_multi_array)
    df_pair['avg_mem_lat'] = df_pair['avg_mem_lat'].apply(
        hi.parse_multi_array)

    return df_pair


def evaluate_app_wise(df_pair, df_baseline):
    df_baseline = df_baseline.set_index(['pair_str', '1_kidx'])

    def calc_norm_runtime(row):
        runtime = row['runtime']
        norm_runtime = []
        for stream_id, stream in enumerate(runtime):
            norm_stream = []
            if len(stream) > 0:
                bench = row['{}_bench'.format(stream_id)]
                for kidx, time in enumerate(stream):
                    # Handle repeated kernel
                    kidx = const.translate_gpusim_kidx(bench, kidx)

                    base_time = df_baseline.loc[
                        (bench, kidx), 'runtime']
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
                for id in range(const.get_num_kernels(bench)):
                    kidx = const.translate_gpusim_kidx(bench, id)
                    base_stream.append(
                        df_baseline.loc[(bench, kidx), 'runtime']
                    )
            baseline.append(base_stream)

        return baseline

    def calculate_total_sld(row):
        start, end = const.get_from_to(row['runtime'])
        return hi.calculate_sld_short(end, row['baseline'])

    df_pair['norm_ipc'] = df_pair.apply(calc_norm_runtime, axis=1)
    df_pair['baseline'] = df_pair.apply(collect_baseline, axis=1)
    df_pair['sld'] = df_pair.apply(calculate_total_sld, axis=1)
    df_pair['ws'] = df_pair['sld'].apply(np.sum)

    return df_pair


def evaluate_kernel_wise(df_pair, df_baseline):
    if not args.multi:
        df_baseline['1_kidx'] = 1

    df_baseline.set_index(['pair_str', '1_kidx'], inplace=True)

    def get_base_runtime(bench, kidx):
        return df_baseline.loc[(bench, kidx), 'runtime']

    def get_base_inst(bench, kidx):
        return df_baseline.loc[(bench, kidx), 'instructions']

    def handle_incomplete(row):
        runtime = row['runtime']

        for idx in range(len(runtime)):
            if len(runtime[idx]) > 0 and runtime[idx][-1] == 0:
                if len(runtime[idx]) > 1:
                    runtime[idx][-1] = row['tot_runtime'] - runtime[idx][-2]
                else:
                    runtime[idx][-1] = row['tot_runtime']

                if row['instructions'][idx][-1] == 0:
                    # Get rid of the last element since it's zero
                    runtime[idx].pop()
                else:
                    # scale by completed instructions
                    runtime[idx][-1] *= \
                        get_base_inst(row['{}_bench'.format(idx)],
                                      row['{}_kidx'.format(idx)]) \
                        / row['instructions'][idx][-1]

        return runtime

    def calc_sld(row):
        # Take average all of kernel instances, but drop the last one if
        # more than one instance
        def average_runtime(list_runtime):
            if len(list_runtime) > 1:
                return np.average(list_runtime[0:-1])
            else:
                return list_runtime[0]

        runtime_1 = average_runtime(row['runtime'][1])
        runtime_2 = average_runtime(row['runtime'][2])
        assert (runtime_1 > 0)
        assert (runtime_2 > 0)

        base_1 = get_base_runtime(row['1_bench'], row['1_kidx'])
        base_2 = get_base_runtime(row['2_bench'], row['2_kidx'])

        norm_1 = base_1 / runtime_1
        norm_2 = base_2 / runtime_2

        return [0, norm_1, norm_2]

    df_pair['runtime'] = df_pair.apply(handle_incomplete, axis=1)
    df_pair['sld'] = df_pair.apply(calc_sld, axis=1)

    # Backward compatible with existing scripts...
    df_pair['1_sld'] = [sld_list[1] for sld_list in df_pair['sld']]
    df_pair['2_sld'] = [sld_list[2] for sld_list in df_pair['sld']]

    df_pair['fairness'] = df_pair['sld'].transform(
        lambda sld: min(sld[1]/sld[2], sld[2]/sld[1])
    )
    df_pair['ws'] = df_pair['sld'].transform(lambda sld: sld[1] + sld[2])

    if args.multi:
        def calc_row_importance(row):
            def get_importance(bench, kidx):
                # total_time = df_baseline[
                #     df_baseline.index.get_level_values('pair_str') == bench
                #     ]['runtime'].sum()
                total_time = 0
                for idx in const.multi_kernel_app[bench]:
                    total_time += df_baseline.loc[(bench, idx), 'runtime']

                importance = get_base_runtime(bench, kidx) / total_time
                return importance

            importance_1 = get_importance(row['1_bench'], row['1_kidx'])
            importance_2 = get_importance(row['2_bench'], row['2_kidx'])

            return [0, importance_1, importance_2]

        def weighted_by(row):
            result = [importance / sld if sld > 0 else 0
                      for importance, sld in zip(row['importance'],
                                                 row['sld'])]
            return result

        # calculated runtime increase weighted by kernel runtime importance
        df_pair['importance'] = df_pair.apply(calc_row_importance,
                                              axis=1)
        df_pair['weighted_increase'] = df_pair.apply(weighted_by, axis=1)
        df_pair['sum_increase'] = df_pair['weighted_increase'].transform(
            lambda inc: inc[1] + inc[2])

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

    if args.how == 'ctx' or args.how == 'lut':
        df_pair = evaluate_kernel_wise(df_pair, df_seq)
        df_pair.to_pickle(args.output)
    else:
        # how == 'dynamic' or 'smk' or 'inter'
        df_pair = evaluate_kernel_wise(df_pair, df_seq)

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

            # Join table with profiled info and predicted performance
            if args.multi:
                pair_cols = ['1_bench', '2_bench', '1_kidx', '2_kidx', '1_intra',
                             '2_intra']
                profiled_cols = ['pair_str_x', 'pair_str_y', '1_kidx_x', '1_kidx_y',
                              '1_intra_x', '1_intra_y']
            else:
                pair_cols = ['1_bench', '2_bench', '1_intra', '2_intra']
                profiled_cols = ['pair_str_x', 'pair_str_y', 'intra_x', 'intra_y']

        elif args.how == 'inter':
            df_profiled = [
                inter_pair_config.build_df_prod(
                    args.isolated_pkl, app, args.cap, True, args.qos)
                for app in app_pairs]

            # Join table with profiled info and predicted performance
            pair_cols = ['1_bench', '2_bench', '1_inter', '2_inter']
            profiled_cols = ['pair_str_x', 'pair_str_y', 'inter_x', 'inter_y']

        else:
            print('Unimplemented for SMK')
            sys.exit(1)

        df_profiled = pd.concat(df_profiled, axis=0, ignore_index=True)

        df_join = pd.merge(df_pair, df_profiled,
                           how='left',
                           left_on=pair_cols,
                           right_on=profiled_cols)

        # Output pickle
        df_join.to_pickle(args.output)


if __name__ == "__main__":
    main()
