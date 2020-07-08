import data.scripts.common.help_iso as hi
import data.scripts.common.constants as const

import argparse
import os
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Generate dataframe pickle for '
                                     'isolation-intra run from csv.')
    parser.add_argument('--csv',
                        default=os.path.join(const.DATA_HOME, 'csv/intra.csv'),
                        help='CSV file to parse')

    parser.add_argument('--out_intra',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/intra.pkl'),
                        help='Output path for the intra dataframe pickle')
    parser.add_argument('--out_best',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/intra_best.pkl'),
                        help='Output path for the best intra dataframe pickle')

    parser.add_argument('--seq',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/seq.pkl'),
                        help='Pickle file for seq run')

    parser.add_argument('--multi',
                        action='store_true',
                        help='Whether simulation file is in multi-kernel '
                             'format.')

    results = parser.parse_args()
    return results


def process_metrics(df_intra, multi):
    # concurrent thread count per SM
    df_intra['thread_count'] = df_intra.apply(
        lambda row: row['intra'] * const.get_block_size(row['pair_str'],
                                                        row['1_kidx']),
        axis=1
    )

    # avg dram bandwidth
    df_intra['avg_dram_bw'] = df_intra['dram_bw'].transform(hi.avg_array)

    # avg row buffer locality
    df_intra['avg_rbh'] = df_intra['row_buffer_locality'] \
        .transform(hi.avg_array)

    # avg dram efficiency
    df_intra['avg_dram_eff'] = df_intra['dram_eff'].transform(hi.avg_array)

    # MPKI
    df_intra['MPKI'] = df_intra['l2_total_accesses'] * \
                       df_intra['l2_miss_rate'] \
                       / (df_intra['instructions'] / 1000)

    # memory requests per cycle
    df_intra['mpc'] = df_intra['mem_count'] / df_intra['runtime']

    # dram busy
    df_intra['dram_busy'] = 1 - np.divide(df_intra['mem_idle']
                                          .transform(hi.avg_array),
                                          df_intra['total_cmd']
                                          .transform(hi.avg_array))

    # compute busy
    idle_sum = df_intra[['empty_warp', 'idle_warp', 'scoreboard_warp']] \
        .sum(axis=1)
    df_intra['comp_busy'] = df_intra['tot_warp_insn'] / \
                            (df_intra['tot_warp_insn'] + idle_sum)

    # resource usage ratio
    df_intra['cta_ratio'] = df_intra['intra'] / const.max_cta_volta
    threads = df_intra['thread_count']
    df_intra['thread_ratio'] = threads / const.max_thread_volta

    if multi:
        # Multi-kernel cannot rely on gpusim to report regs/smem since there
        # are multiple kernels
        df_intra['smem'] = df_intra.apply(
            lambda row: const.get_smem(row['pair_str'], row['1_kidx']),
            axis=1
        )
        df_intra['regs'] = df_intra.apply(
            lambda row: const.get_regs(row['pair_str'], row['1_kidx']),
            axis=1
        )

    df_intra['smem_ratio'] = df_intra['intra'] * df_intra['smem'] \
                             / const.max_smem

    df_intra['reg_ratio'] = threads * df_intra['regs'] / const.max_register

    # Dominant usage
    df_intra['usage'] = df_intra[['cta_ratio', 'thread_ratio', 'smem_ratio',
                                  'reg_ratio']].max(axis=1)

    # PATCH for GPGPU-Sim not selected cycles
    df_intra['not_selected_cycles'] = df_intra['cycles_per_issue'] - \
                                      df_intra['stall_mem_cycles'] - \
                                      df_intra['stall_sfu_cycles'] - \
                                      df_intra['stall_tensor_cycles'] - \
                                      df_intra['stall_int_cycles'] - \
                                      df_intra['stall_dp_cycles'] - \
                                      df_intra['stall_sp_cycles'] - \
                                      df_intra['scoreboard_cycles'] - \
                                      df_intra['branch_cycles'] - \
                                      df_intra['inst_empty_cycles'] - \
                                      df_intra['barrier_cycles']


def normalize_over_seq(df_intra, df_seq, multi):
    df_intra = df_intra.copy()

    hi.multi_array_col_seq(df_intra)

    # calculate ipc
    df_intra['ipc'] = df_intra['instructions'] / df_intra['runtime']

    # gpusim config
    hi.process_config_column('intra', df=df_intra)
    hi.process_config_column('1_kidx', df=df_intra, default=1)
    hi.process_config_column('bypass_l2', df=df_intra, default=False)

    df_seq.set_index(['pair_str', '1_kidx'], inplace=True)

    # Process df_intra
    def norm_over_seq(index, value, inverse=True):
        return hi.normalize(df_seq, index, 'ipc', value, inverse)

    # normalized IPC
    df_intra['norm_ipc'] = df_intra.apply(lambda row:
                                          norm_over_seq((row['pair_str'],
                                                         row['1_kidx']),
                                                        row['ipc'],
                                                        False),
                                          axis=1)
    return df_intra


def get_best_intra(df_intra):
    df_intra = df_intra.reset_index(drop=True)

    df_intra['perfdollar'] = df_intra['norm_ipc'] / df_intra['usage']

    # sort = df_intra[cols].sort_values(['pair_str', 'perfdollar'],
    # ascending=[True, True])
    # sort = sort[sort['norm_ipc'] > 0.8]

    best_df = []
    bench_list = df_intra['pair_str'].unique()
    for bench in bench_list:
        idx = df_intra[(df_intra['norm_ipc'] > 0.8) &
                       (df_intra['pair_str'] == bench)]['perfdollar'].idxmax()
        # (df_intra['l2'] > 0.25)]['perfdollar'].idxmax()
        best_df.append(df_intra.iloc[idx])

    best_df = pd.concat(best_df, axis=1).T

    return best_df


def main():
    # Parse arguments
    args = parse_args()

    # Read CSV file
    df_intra = pd.read_csv(args.csv)

    df_seq = pd.read_pickle(args.seq)
    df_intra = normalize_over_seq(df_intra, df_seq, args.multi)

    process_metrics(df_intra, args.multi)

    # Output pickle
    df_intra.to_pickle(args.out_intra)

    # best intra
    if not args.multi:
        df_best = get_best_intra(df_intra)
        df_best.to_pickle(args.out_best)


if __name__ == "__main__":
    main()
