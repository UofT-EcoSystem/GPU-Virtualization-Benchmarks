import argparse
import os
import pandas as pd
import numpy as np

import data.scripts.common.constants as const


def parse_args():
    parser = argparse.ArgumentParser('Generate inputs for scheduler '
                                     'simulator: kernel idx, kernel runtime '
                                     'in isolation, slowdown matrix for '
                                     'pair-wise kernels')

    parser.add_argument('--seq_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/seq.pkl'),
                        help='Pickle file for seq run')

    parser.add_argument('--pair_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/pair_dynamic.pkl'),
                        help='csv file to parse')

    parser.add_argument('--outdir',
                        default=os.path.join(const.DATA_HOME,
                                             'csv'),
                        help='Output directory to save scheduler inputs.')

    results = parser.parse_args()
    return results


def main():
    # Parse arguments
    args = parse_args()

    # input 1: kernel idx, name, runtime in isolation
    df_seq = pd.read_pickle(args.seq_pkl)
    df_seq.set_index('pair_str', inplace=True)

    # kernel name -> idx
    dict_seq = {}

    with open(os.path.join(args.outdir, 'kernel_idx.csv'), 'w+') as f:
        f.write("idx, name, runtime(cycles)\n")

        for idx, name in enumerate(const.kernel_dict):
            dict_seq[name] = idx
            runtime = df_seq.loc[name, 'runtime']
            f.write("{0}, {1}, {2}\n".format(idx, name, runtime))

    # input 2: slowdown matrix
    df_pair = pd.read_pickle(args.pair_pkl)

    # TODO: decide what policies to select slowdown configuration
    # For now, go with max weighted speedup
    df_pair_ws = df_pair.sort_values('ws', ascending=False).drop_duplicates([
        '1_bench', '2_bench'])
    df_pair_ws.reset_index(inplace=True, drop=True)

    shape = (len(df_seq.index), len(df_seq.index))
    slowdown_matrix = np.zeros(shape)

    for row_idx, row in df_pair_ws.iterrows():
        # fill 1_bench slowdown
        pos_1_bench = dict_seq[row['1_bench']]
        pos_2_bench = dict_seq[row['2_bench']]

        slowdown_matrix[pos_1_bench, pos_2_bench] = row['1_sld']
        slowdown_matrix[pos_2_bench, pos_1_bench] = row['2_sld']

    np.savetxt(os.path.join(args.outdir, 'slowdown.csv'), slowdown_matrix,
               delimiter=',')


if __name__ == "__main__":
    main()
