import pandas as pd
import os
import sys
from ipywidgets import widgets
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import data.scripts.common.format as fmt
import data.scripts.common.constants as const
import data.scripts.gen_tables.gen_pair_configs as gen_pair
import data.scripts.gen_tables.search_best_inter as search_inter

plt.style.use('seaborn-talk')


def parse_args():
    parser = argparse.ArgumentParser('Generate a normalized ipc graph as a \
            function how many sm/mem channels were allocated to the app')

    parser.add_argument('--pickle',
                        required=True,
                        help='pickle file containing the dataframe with results.')

    parser.add_argument('--outfile',
                        help='name of file where the image will be saved.')

    results = parser.parse_args()
    return results


def main():

    args = parse_args()

    df = pd.read_pickle(args.pickle)

    df.drop(df.columns.difference(['pair_str', 'mem_channels', 'norm_ipc']), 1,
            inplace=True)
    df['mem_channels'] = pd.to_numeric(df['mem_channels'])
    df['norm_ipc'] = pd.to_numeric(df['norm_ipc'])
    df.columns=["bm_suite", "mem_channels", "norm_ipc"]

    df.sort_values(['bm_suite', 'mem_channels'], ascending=[True, True],
            inplace=True)
    df['suite'], df['bm'] = df['bm_suite'].str.split('_', 1).str
    df.drop(columns=['bm_suite'], inplace=True)


    # list of dataframes by suite
    df_by_suite = []
    suites = 0

    for suite, df_suite in df.groupby('suite'):
        # rename column 'bm' so it appears as a title
        df_suite.columns=['mem_channels', 'norm_ipc', 'suite', df_suite.iat[0,2]]
        df_by_suite.append(df_suite)
        suites += 1


    f, axes = plt.subplots(1, suites, figsize=(40, 7), sharex=True)

    # create a plot for each of the suites 
    for i in range(suites):
        graph = sns.lineplot(x='mem_channels', y='norm_ipc', marker='o',
                data=df_by_suite[i], hue=df_by_suite[i].iat[0,2],
                ax=axes[i])
        graph.legend(loc='lower right')
        axes[i].set_xticks(range(0,25,4))

    f.savefig(args.outfile)


if __name__ == '__main__':
    main()
