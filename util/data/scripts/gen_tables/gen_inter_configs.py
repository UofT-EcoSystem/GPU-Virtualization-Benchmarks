import data.scripts.common.constants as const

import argparse
import os
import pandas as pd
import numpy as np

TOP_CHOICES = 5


def parse_args(_args):
    parser = argparse.ArgumentParser('Generate application pair configs '
                                     'based on inter profiling results.')

    parser.add_argument('--inter_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/inter.pkl'),
                        help='Pickle file for isolation inter run')

    parser.add_argument('--apps',
                        nargs='+',
                        default=['cut_sgemm-0', 'parb_spmv-0'],
                        help='Pair of benchmarks to be considered.')

    parser.add_argument('--output',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/inter_candidates.pkl'),
                        help='Output path for the pair '
                             'candidate dataframe pickle')

    parser.add_argument('--print',
                        action='store_true',
                        help='Whether to print selected configs to console.')

    parser.add_argument('--cap',
                        type=float,
                        default=2.5,
                        help='Fail fast simulation: cap runtime at n times '
                             'the longer kernel. Default is 2.5x.')

    parser.add_argument('--qos',
                        default=0.5,
                        type=float,
                        help='QoS target of each benchmark.')

    parser.add_argument('--top',
                        action='store_true',
                        help='Cherry pick the top configs.')

    results = parser.parse_args(_args)
    return results


def build_df_prod(inter_pkl, apps, cap, top, qos):
    df_inter = pd.read_pickle(inter_pkl)

    df_a1 = df_inter[df_inter['pair_str'] == apps[0]].copy()
    df_a1['key'] = 0
    df_a2 = df_inter[df_inter['pair_str'] == apps[1]].copy()
    df_a2['key'] = 0

    df_prod = df_a1.merge(df_a2, how='outer', on='key')
    df_prod.drop(columns=['key'])

    df_prod['inter_sum'] = df_prod['inter_x'] + df_prod['inter_y']
    df_prod['norm_ipc_sum'] = df_prod['norm_ipc_x'] + df_prod['norm_ipc_y']

    # Filter out configs that exceed resource limit
    df_prod = df_prod[df_prod['inter_sum'] <= const.num_sm_volta]

    # Filter out configs that do not meet qos target
    df_prod = df_prod[(df_prod['norm_ipc_x'] >= qos) &
                      (df_prod['norm_ipc_y'] >= qos)]

    df_prod.sort_values('norm_ipc_sum', inplace=True, ascending=False)

    if len(df_prod.index) == 0:
        print("No feasible inter configs for {} at QoS {}".format(apps, qos))
        return pd.DataFrame()

    # Only run the top rated candidates
    if top:
        df_prod = df_prod.head(TOP_CHOICES)

    def build_config(row):
        config_base = 'TITANV-PAE-CONCURRENT-SEP_RW-LSRR'

        # fail fast config
        max_cycle = int(cap * max(row['runtime_x'] * row['norm_ipc_x'],
                                  row['runtime_y'] * row['norm_ipc_y']))
        config_base += '-CAP_{0}_CYCLE'.format(max_cycle)

        config_inter = 'INTER_0:' + str(row['inter_x']) + ':' \
                       + str(row['inter_y']) + '_SM'

        config_l2 = 'ENABLE_L2D_1:{}:{}'.format(
            int(not row['bypass_l2_x']),
            int(not row['bypass_l2_y'])
        )

        config = '-'.join([config_base, config_inter, config_l2])

        return config

    df_prod['config'] = df_prod.apply(build_config, axis=1)

    return df_prod


def main(_args):
    args = parse_args(_args)

    if len(args.apps) != 2:
        print("Number of apps is not equal to 2. Abort.")
        exit(1)

    df_prod = build_df_prod(args.inter_pkl, args.apps, args.cap, args.top,
                            args.qos)

    if len(df_prod.index) > 0:
        if args.print:
            print('-' * 30)
            print('app order:', args.apps[0], args.apps[1])

            for config in df_prod['config']:
                print('gpusim config:', config)

            print('-' * 30)

        df_prod.to_pickle(args.output)

        return df_prod['config']
    else:
        return []


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
