import data.scripts.common.constants as const
import data.scripts.gen_tables.gen_inter_configs as gen_inter_configs

import argparse
import os
import pandas as pd
import numpy as np


def parse_args(_args):
    parser = argparse.ArgumentParser('Search for best inter-SM sharing config '
                                     'based on profiled and simulated best '
                                     'information.')

    parser.add_argument('--seq_csv',
                        default=os.path.join(const.DATA_HOME,
                                             'csv/seq.csv'),
                        help='Seq csv file for runtime cap.')

    parser.add_argument('--inter_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/inter.pkl'),
                        help='Pickle file for isolation inter run')

    parser.add_argument('--pair_inter_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/pair_inter.pkl'),
                        help='Pickle file for pair inter run')

    parser.add_argument('--apps',
                        nargs='+',
                        default=['cut_sgemm-0', 'parb_spmv-0'],
                        help='Pair of benchmarks to be considered.')

    parser.add_argument('--cap',
                        type=float,
                        default=2.5,
                        help='Fail fast simulation: cap runtime at n times '
                             'the longer kernel. Default is 2.5x.')

    parser.add_argument('--how', default='local', choices=['local', 'large'],
                        help='How to search?')

    parser.add_argument('--print',
                        action='store_true',
                        help='Whether to print selected configs to console.')

    results = parser.parse_args(_args)
    return results


def build_config(max_cycle, inter_cta):
    config_base = 'TITANV-PAE-CONCURRENT-SEP_RW-LSRR'

    config_base += '-CAP_{0}_CYCLE'.format(max_cycle)

    config_inter = 'INTER_0:' + str(inter_cta) + ':' \
                   + str(const.num_sm_volta - inter_cta) + '_SM'

    config = '-'.join([config_base, config_inter])

    return config


def build_configs_local(inter_pkl, pair_inter_pkl, apps, max_cycle):
    df_prod = gen_inter_configs.build_df_prod(inter_pkl, apps, 2.5)

    if len(df_prod.index) == 0:
        return []

    df_prod.reset_index(drop=True, inplace=True)
    estimated_best_inter = df_prod.at[0, 'inter_x']

    df_pair_inter = pd.read_pickle(pair_inter_pkl)
    bench_1_match = (df_pair_inter['1_bench'] == apps[0])
    bench_2_match = (df_pair_inter['2_bench'] == apps[1])
    df_pair_inter = df_pair_inter[bench_1_match & bench_2_match]

    if len(df_pair_inter) == 0:
        return []

    df_pair_inter.set_index('1_inter', inplace=True)

    if estimated_best_inter not in df_pair_inter.index:
        print('App {0} and {1} cannot find expected best inter simulation.'.format(
            apps[0], apps[1]))
        return []

    # real ws of estimated best
    best_ws = df_pair_inter.loc[estimated_best_inter, 'ws']

    step = 4
    current_best_ws = best_ws

    # Try positive direction
    inter_x = estimated_best_inter + step

    reverse_search = None
    while inter_x <= const.num_sm_volta - step:
        if inter_x in df_pair_inter.index:
            # check if it's increasing
            if df_pair_inter.loc[inter_x, 'ws'] > current_best_ws:
                current_best_ws = df_pair_inter.loc[inter_x, 'ws']
                reverse_search = False
                inter_x += step
            else:
                # decreasing before increasing, we should change search
                # direction
                if reverse_search is None:
                    reverse_search = True

                # either immediate decreasing or increase then decrease,
                # we should stop going in this current direction
                break
        else:
            c = build_config(max_cycle, inter_x)
            return [c]

    if reverse_search:
        inter_x = estimated_best_inter - 4
        while inter_x >= step:
            if inter_x in df_pair_inter.index:
                if df_pair_inter.loc[inter_x, 'ws'] > current_best_ws:
                    current_best_ws = df_pair_inter.loc[inter_x, 'ws']
                    inter_x -= step
                else:
                    # donezo
                    return []
            else:
                c = build_config(max_cycle, inter_x)
                return [c]
    else:
        return []


def build_configs_large(inter_pkl, pair_inter_pkl, apps, max_cycle):
    df_prod = gen_inter_configs.build_df_prod(inter_pkl, apps, 2.5)

    if len(df_prod.index) == 0:
        return []

    df_prod.reset_index(drop=True, inplace=True)
    estimated_best_config = df_prod.at[0, 'config']

    df_pair_inter = pd.read_pickle(pair_inter_pkl)

    bench_1_match = (df_pair_inter['1_bench'] == apps[0])
    bench_2_match = (df_pair_inter['2_bench'] == apps[1])
    config_match = (df_pair_inter['config'] == estimated_best_config)

    if len(df_pair_inter[bench_1_match & bench_2_match]) == 0:
        return []

    best_row = df_pair_inter[bench_1_match & bench_2_match &
                             config_match].iloc[0]

    best_ws = best_row['ws']
    best_1_inter = best_row['inter_x']
    # print(best_ws)
    # print(best_1_inter)

    # Start to search through df_prod to find cta bounds for app 0
    # Find left bound: maximum inter_x that is smaller than best inter_x
    # and has a sum_ipc below best_ws
    df_left = df_prod[df_prod['inter_x'] < best_1_inter]

    if len(df_left) == 0:
        left_bound = best_1_inter - 4
    else:
        df_left_gt_best = df_left[df_left['norm_ipc_sum'] > best_ws]

        if len(df_left_gt_best) == 0:
            # Take the largest inter_x in df_left
            left_bound = df_left['inter_x'].max()
        else:
            # Take the lowest inter_x in df_left_gt_best, minus 4 if possible
            left_bound = max(df_left_gt_best['inter_x'].min(), 8)

    left_bound = int(left_bound)

    # Find right bound
    df_right = df_prod[df_prod['inter_x'] > best_1_inter]

    if len(df_right) == 0:
        right_bound = best_1_inter + 4
    else:
        df_right_gt_best = df_right[df_right['norm_ipc_sum'] > best_ws]

        if len(df_right_gt_best) == 0:
            # Take the lowest inter_x in df_right
            right_bound = df_right['inter_x'].min()
        else:
            # Take the largest inter_x in df_right_gt_best, plus 4 if possible
            right_bound = min(df_right_gt_best['inter_x'].max(),
                              const.num_sm_volta - 8)

    right_bound = int(right_bound)

    if left_bound == right_bound:
        print('Search range is zero for apps: ', apps)
        return []

    list_configs = []
    for inter_cta in range(left_bound, right_bound + 4, 4):
        if (inter_cta == best_1_inter) or (inter_cta == best_1_inter + 4):
            continue

        config = build_config(max_cycle, inter_cta)
        list_configs.append(config)

    return list_configs


def main(_args):
    args = parse_args(_args)

    if len(args.apps) != 2:
        print("Number of apps is not equal to 2. Abort.")
        exit(1)

    df_seq = pd.read_csv(args.seq_csv)
    df_seq.set_index('pair_str', inplace=True)

    max_cycle = int(args.cap * max(df_seq.loc[args.apps[0], 'runtime'],
                                    df_seq.loc[args.apps[1], 'runtime']))

    if args.how == 'large':
        list_configs = build_configs_large(args.inter_pkl, args.pair_inter_pkl,
                                           args.apps, max_cycle)
    else:
        list_configs = build_configs_local(args.inter_pkl, args.pair_inter_pkl,
                                           args.apps, max_cycle)

    if args.print:
        print('-' * 30)
        print('app order:', args.apps[0], args.apps[1])

        for config in list_configs:
            print('gpusim config:', config)

        print('-' * 30)

    return list_configs


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
