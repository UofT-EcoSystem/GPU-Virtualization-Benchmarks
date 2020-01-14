import common.help_iso as hi
import common.constants as const

import argparse
import os
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Generate dataframe pickle for isolation-intra run from csv.')
    parser.add_argument('--csv',
                        default=os.path.join(const.DATA_HOME, 'csv/intra.csv'),
                        help='CSV file to parse')

    parser.add_argument('--out_intra',
                        default=os.path.join(const.DATA_HOME, 'pickles/intra.pkl'),
                        help='Output path for the intra dataframe pickle')
    parser.add_argument('--out_best',
                        default=os.path.join(const.DATA_HOME, 'pickles/intra_best.pkl'),
                        help='Output path for the best intra dataframe pickle')

    parser.add_argument('--seq',
                        default=os.path.join(const.DATA_HOME, 'pickles/seq.pkl'),
                        help='Pickle file for seq run')

    results = parser.parse_args()
    return results


def process_df_intra(df_intra, df_seq):
    df_intra = df_intra.copy()

    df_seq.set_index('pair_str', inplace=True)

    # Process df_intra
    def norm_over_seq(index, metric, value, inverse=True):
        return hi.normalize(df_seq, index, metric, value, inverse)

    # normalized IPC
    df_intra['norm_ipc'] = df_intra.apply(lambda row: norm_over_seq(row['pair_str'], 'ipc', row['ipc'], False), axis=1)

    # gpusim config
    hi.process_config_column('intra', 'l2', df=df_intra)

    # avg dram bandwidth
    df_intra['avg_dram_bw'] = df_intra['dram_bw'].transform(hi.avg_array)

    # avg dram efficiency
    df_intra['avg_dram_eff'] = df_intra['dram_eff'].transform(hi.avg_array)

    # dram busy
    df_intra['dram_busy'] = 1 - np.divide(df_intra['mem_idle'].transform(hi.avg_array),
                                          df_intra['total_cmd'].transform(hi.avg_array))

    # compute busy
    idle_sum = df_intra[['empty_warp', 'idle_warp', 'scoreboard_warp']].sum(axis=1)
    df_intra['comp_busy'] = df_intra['tot_warp_insn'] / (df_intra['tot_warp_insn'] + idle_sum)

    # resource usage ratio
    df_intra['cta_ratio'] = df_intra['intra'] / const.max_cta_volta
    threads = df_intra['intra'] * df_intra['block_x'] * df_intra['block_y'] * df_intra['block_z']
    df_intra['thread_ratio'] = threads / const.max_thread_volta
    df_intra['smem_ratio'] = df_intra['intra'] * df_intra['smem'] / const.max_smem
    df_intra['reg_ratio'] = threads * df_intra['regs'] / const.max_register

    def pow_2(*resc_list):
        done_first = False
        for r in resc_list:
            if done_first:
                usage = usage + df_intra[r] ** 2
            else:
                done_first = True
                usage = df_intra[r] ** 2
        return usage

    df_intra['usage'] = pow_2('cta_ratio', 'thread_ratio', 'smem_ratio', 'reg_ratio', 'l2', 'dram_busy', 'comp_busy')

    return df_intra


def get_best_intra(df_intra):
    df_intra = df_intra.copy()

    df_intra['perfdollar'] = df_intra['norm_ipc'] / df_intra['usage']

    # sort = df_intra[cols].sort_values(['pair_str', 'perfdollar'], ascending=[True, True])
    # sort = sort[sort['norm_ipc'] > 0.8]

    best_df = []
    bench_list = df_intra['pair_str'].unique()
    for bench in bench_list:
        idx = df_intra[(df_intra['norm_ipc'] > 0.8) & (df_intra['pair_str'] == bench) & (df_intra['l2'] > 0.25)] \
            ['perfdollar'].idxmax()
        best_df.append(df_intra.iloc[idx])

    best_df = pd.concat(best_df, axis=1).T

    return best_df


def main():
    # Parse arguments
    args = parse_args()

    # Read CSV file
    df_intra = pd.read_csv(args.csv)

    df_seq = pd.read_pickle(args.seq)

    df_intra = process_df_intra(df_intra, df_seq)
    df_best = get_best_intra(df_intra)

    # Output pickle
    df_intra.to_pickle(args.out_intra)
    df_best.to_pickle(args.out_best)


if __name__ == "__main__":
    main()
