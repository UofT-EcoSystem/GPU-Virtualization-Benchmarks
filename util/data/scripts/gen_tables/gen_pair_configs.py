import common.help_iso as hi
import common.constants as const

import argparse
import os
import pandas as pd
import re
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Generate application pair configs based on intra profiling results.')

    parser.add_argument('--intra_pkl',
                        default=os.path.join(const.DATA_HOME, 'pickles/intra.pkl'),
                        help='Pickle file for isolation intra run')

    parser.add_argument('--apps',
                        nargs='+',
                        default=['cut_sgemm-0', 'parb_spmv-0'],
                        help='Pair of benchmarks to be considered.')

    parser.add_argument('--qos',
                        default=0.8,
                        type=float,
                        help='Quality of Service for each benchmark in terms of normalized IPC.')

    parser.add_argument('--output',
                        default=os.path.join(const.DATA_HOME, 'pickles/pair_candidates.pkl'),
                        help='Output path for the pair candidate dataframe pickle')

    results = parser.parse_args()
    return results


def main():
    args = parse_args()
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

    print('Original', df_prod.shape)

    # Step 2: drop any pair that exceeds resource usage:
    # 'cta_ratio', 'thread_ratio', 'smem_ratio', 'reg_ratio', 'l2', 'dram_busy', 'comp_busy'
    resources = ['cta_ratio', 'thread_ratio', 'smem_ratio', 'reg_ratio', 'l2']

    for rsrc in resources:
        df_prod = df_prod[(df_prod[rsrc+'_x'] + df_prod[rsrc+'_y']) <= 1.0]
        print(rsrc, df_prod.shape)

    # for l2 only, keep the ones that add up to 1.0 exactly
    df_prod = df_prod[(df_prod['l2_x'] + df_prod['l2_y'] == 1.0)]

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

    # print('-'*10, 'Candidates', '-'*10)
    # print(df_prod[['norm_ipc_x', 'norm_ipc_y', 'diff_mflat', 'sum_ipc',
    #                'intra_x', 'intra_y', 'l2_x', 'l2_y',
    #                'sum_comp', 'sum_dram', 'penalized']])

    print('')
    print('-'*10, 'Best Candidate', '-'*10)

    best = df_prod.iloc[0]
    print('app order:', best['pair_str_x'], best['pair_str_y'])

    config_base = 'TITANV-CONCURRENT-SEP_RW'
    config_intra = 'INTRA_0:' + str(best['intra_x']) + ':' + str(best['intra_y']) + '_CTA'
    config_l2 = 'PARTITION_L2_0:' + str(best['l2_x']) + ':' + str(best['l2_y'])
    if best['pair_str_x'] == best['penalized']:
        config_icnt = 'ICNT_0:2:1_PRIORITY'
    else:
        config_icnt = 'ICNT_0:1:2_PRIORITY'

    config = '-'.join([config_base, config_intra, config_l2, config_icnt])
    print('gpusim config:', config)

    df_prod.to_pickle(args.output)

    return config


if __name__ == "__main__":
    main()
