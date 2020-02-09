import data.scripts.common.constants as const

import argparse
import os
import pandas as pd
import numpy as np


def parse_args(_args):
    parser = argparse.ArgumentParser('Generate application pair configs '
                                     'based on intra profiling results.')

    parser.add_argument('--intra_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/intra.pkl'),
                        help='Pickle file for isolation intra run')

    parser.add_argument('--apps',
                        nargs='+',
                        default=['cut_sgemm-0', 'parb_spmv-0'],
                        help='Pair of benchmarks to be considered.')

    parser.add_argument('--qos',
                        default=0.75,
                        type=float,
                        help='Quality of Service for each benchmark '
                             'in terms of normalized IPC.')

    parser.add_argument('--output',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/pair_candidates.pkl'),
                        help='Output path for the pair '
                             'candidate dataframe pickle')

    parser.add_argument('--random',
                        default=False,
                        action='store_true',
                        help='Use random address mapping for global access.')

    parser.add_argument('--print',
                        action='store_true',
                        help='Whether to print selected configs to console.')

    parser.add_argument('--cap',
                        type=float,
                        default=2.5,
                        help='Fail fast simulation: cap runtime at n times '
                             'the longer kernel. Default is 2.5x.')

    parser.add_argument('--top',
            action='store_true',
            help='Cherry pick the top configs.')

    results = parser.parse_args(_args)
    return results


def build_df_prod(intra_pkl, qos, apps, random, cap, top_only=False):
    df_intra = pd.read_pickle(intra_pkl)

    # Step 1: get rid of all intra configs that do not meet QoS
    df_intra = df_intra[df_intra['norm_ipc'] >= qos]

    # intra configs for each app:
    if len(apps) != 2:
        print('Number of apps is not equal to 2. Abort.')
        exit(1)

    df_app = []

    for app in apps:
        _df = df_intra[df_intra['pair_str'] == app].copy()
        _df['key'] = 0

        df_app.append(_df)

    # create cartesian product of the two dfs
    df_prod = df_app[0].merge(df_app[1], how='outer', on='key')
    df_prod.drop(columns=['key'])

    # Step 2: drop any pair that exceeds resource usage:
    # 'cta_ratio', 'thread_ratio', 'smem_ratio', 'reg_ratio',
    # 'l2', 'dram_busy', 'comp_busy'
    resources = ['cta_ratio', 'thread_ratio', 'smem_ratio', 'reg_ratio', 'l2']

    for rsrc in resources:
        df_prod = df_prod[(df_prod[rsrc + '_x'] + df_prod[rsrc + '_y']) <= 1.0]

    # for l2 only, keep the ones that add up to 1.0 exactly
    df_prod = df_prod[(df_prod['l2_x'] + df_prod['l2_y'] == 1.0)]

    if len(df_prod) == 0:
        print('Error. No feasible pair configs for {0}+{1} at QoS {2}.'
              .format(apps[0], apps[1], qos))
        return pd.DataFrame()

    # calculate difference in memory latency
    df_prod['diff_mflat'] = np.abs(df_prod['avg_mem_lat_y'] -
                                   df_prod['avg_mem_lat_x'])
    df_prod['sum_ipc'] = df_prod['norm_ipc_x'] + df_prod['norm_ipc_y']
    df_prod['sum_comp'] = df_prod['comp_busy_x'] + df_prod['comp_busy_y']
    df_prod['sum_dram'] = df_prod['dram_busy_x'] + df_prod['dram_busy_y']

    df_prod['penalized'] = df_prod.apply(lambda row:
                                         row['pair_str_x']
                                         if row['avg_mem_lat_x']
                                            < row['avg_mem_lat_y']
                                         else row['pair_str_y'],
                                         axis=1)

    df_prod.sort_values('diff_mflat', inplace=True, ascending=True)

    if top_only:
        # Drop any configs that have a higher mflat diff compared to the lowest
        # and also lower sum_ipc
        baseline_sum_ipc = df_prod.iloc[0]['sum_ipc']
        df_prod = df_prod[df_prod['sum_ipc'] >= baseline_sum_ipc]
        df_prod.reset_index(inplace=True)

        # Top candidates: top 3 choices with lowest mflat diff and top 3 with
        # highest sum_ipc -> take union of the two sets
        top_diff_mflat_idx = df_prod.head(3).index
        top_sum_ipc_idx = df_prod.sort_values('sum_ipc',
                                              ascending=False).head(3).index
        union_idx = list(set(top_diff_mflat_idx).union(set(top_sum_ipc_idx)))

        # Only keep top candidates
        df_prod = df_prod.iloc[union_idx]

    df_prod.reset_index(inplace=True)

    # Build GPGPU-Sim config string
    def build_config(row):
        config_base = 'TITANV-PAE-CONCURRENT-SEP_RW-LSRR'
        if random:
            config_base += '-RANDOM'

        # fail fast config
        max_cycle = int(cap * max(row['runtime_x'], row['runtime_y']))
        config_base += '-CAP_{0}_CYCLE'.format(max_cycle)

        config_intra = 'INTRA_0:' + str(row['intra_x']) + ':' \
                       + str(row['intra_y']) + '_CTA'
        config_l2 = 'PARTITION_L2_0:' + str(row['l2_x']) + ':' + str(row['l2_y'])

#        if row['pair_str_x'] == row['penalized']:
#            config_icnt = 'ICNT_0:2:1_PRIORITY'
#        else:
#            config_icnt = 'ICNT_0:1:2_PRIORITY'

        config = '-'.join([config_base, config_intra, config_l2])

        return config

    df_prod['config'] = df_prod.apply(build_config, axis=1)

    return df_prod


def main(_args):
    args = parse_args(_args)

    df_prod = build_df_prod(args.intra_pkl, args.qos, args.apps,
                            random=args.random, cap=args.cap, top_only=args.top)

    if len(df_prod.index) > 0:
        df_prod.to_pickle(args.output)

        def print_config(*str):
            if args.print:
                print(*str)

        print_config('')
        print_config('-' * 10, 'Top Candidate(s)', '-' * 10)
        print_config('app order:', args.apps[0], args.apps[1])

        for config in df_prod['config']:
            print_config('gpusim config:', config)

        print_config('-' * 30)

        return df_prod['config']
    else:
        return []


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
