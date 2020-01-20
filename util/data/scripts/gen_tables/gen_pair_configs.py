import data.scripts.common.constants as const

import argparse
import os
import pandas as pd
import numpy as np


def parse_args(_args):
    parser = argparse.ArgumentParser('Generate application pair configs based on intra profiling results.')

    parser.add_argument('--intra_pkl',
                        default=os.path.join(const.DATA_HOME, 'pickles/intra.pkl'),
                        help='Pickle file for isolation intra run')

    parser.add_argument('--apps',
                        nargs='+',
                        default=['cut_sgemm-0', 'parb_spmv-0'],
                        help='Pair of benchmarks to be considered.')

    parser.add_argument('--qos',
                        default=0.75,
                        type=float,
                        help='Quality of Service for each benchmark in terms of normalized IPC.')

    parser.add_argument('--output',
                        default=os.path.join(const.DATA_HOME, 'pickles/pair_candidates.pkl'),
                        help='Output path for the pair candidate dataframe pickle')

    parser.add_argument('--random',
                        default=False,
                        action='store_true',
                        help='Use random address mapping for global access.')

    parser.add_argument('--print',
                        action='store_true',
                        help='Whether to print selected configs to console.')

    results = parser.parse_args(_args)
    return results


def main(_args):
    args = parse_args(_args)

    def print_config(*str):
        if args.print:
            print(str)

    df_intra = pd.read_pickle(args.intra_pkl)

    # Step 1: get rid of all intra configs that do not meet QoS
    df_intra = df_intra[df_intra['norm_ipc'] >= args.qos]

    # intra configs for each app:
    if len(args.apps) != 2:
        print('Number of apps is not equal to 2. Abort.')
        exit(1)

    df_app = []

    for app in args.apps:
        _df = df_intra[df_intra['pair_str'] == app].copy()
        _df['key'] = 0

        df_app.append(_df)

    # create cartesian product of the two dfs
    df_prod = df_app[0].merge(df_app[1], how='outer', on='key')
    df_prod.drop(columns=['key'])

    # Step 2: drop any pair that exceeds resource usage:
    # 'cta_ratio', 'thread_ratio', 'smem_ratio', 'reg_ratio', 'l2', 'dram_busy', 'comp_busy'
    resources = ['cta_ratio', 'thread_ratio', 'smem_ratio', 'reg_ratio', 'l2']

    for rsrc in resources:
        df_prod = df_prod[(df_prod[rsrc+'_x'] + df_prod[rsrc+'_y']) <= 1.0]

    # for l2 only, keep the ones that add up to 1.0 exactly
    df_prod = df_prod[(df_prod['l2_x'] + df_prod['l2_y'] == 1.0)]

    if len(df_prod) == 0:
        print('Error. No feasible pair configs for {0}+{1} at QoS {2}.'.format(args.apps[0], args.apps[1], args.qos))
        return ''

    # calculate difference in memory latency
    df_prod['diff_mflat'] = np.abs(df_prod['avg_mem_lat_y'] - df_prod['avg_mem_lat_x'])
    df_prod['sum_ipc'] = df_prod['norm_ipc_x'] + df_prod['norm_ipc_y']
    df_prod['sum_comp'] = df_prod['comp_busy_x'] + df_prod['comp_busy_y']
    df_prod['sum_dram'] = df_prod['dram_busy_x'] + df_prod['dram_busy_y']

    df_prod['penalized'] = df_prod.apply(lambda row:
                                         row['pair_str_x'] if row['avg_mem_lat_x'] < row['avg_mem_lat_y']
                                         else row['pair_str_y'],
                                         axis=1)

    df_prod.sort_values(['diff_mflat', 'sum_ipc'], inplace=True, ascending=[True, False])

    print_config('')
    print_config('-'*10, 'Best Candidate', '-'*10)

    best = df_prod.iloc[0]
    print_config('app order:', best['pair_str_x'], best['pair_str_y'])

    config_base = 'TITANV-CONCURRENT-SEP_RW'
    if args.random:
        config_base += '-RANDOM'

    config_intra = 'INTRA_0:' + str(best['intra_x']) + ':' + str(best['intra_y']) + '_CTA'
    config_l2 = 'PARTITION_L2_0:' + str(best['l2_x']) + ':' + str(best['l2_y'])
    if best['pair_str_x'] == best['penalized']:
        config_icnt = 'ICNT_0:2:1_PRIORITY'
    else:
        config_icnt = 'ICNT_0:1:2_PRIORITY'

    config = '-'.join([config_base, config_intra, config_l2, config_icnt])
    print_config('gpusim config:', config)

    df_prod.to_pickle(args.output)

    print_config('-'*30)

    return config


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
