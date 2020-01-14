import matplotlib.backends.backend_pdf
import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import common.help_iso as hi
import common.constants as const

mpl.style.use('seaborn-paper')


def print_intra(df, benchmark):
    filename = '{0}-{1}.pdf'.format(benchmark, 'intra')
    filename = os.path.join(const.DATA_HOME, 'graphs', filename)
    with PdfPages(filename) as pdf:
        hi.plot_page_intra(df, 'norm_ipc', benchmark, pdf)
        hi.plot_page_intra(df, 'avg_dram_bw', benchmark, pdf)
        hi.plot_page_intra(df, 'dram_busy', benchmark, pdf)
        hi.plot_page_intra(df, 'l2_miss_rate', benchmark, pdf)
        hi.plot_page_intra(df, 'l2_BW', benchmark, pdf)
        hi.plot_page_intra(df, 'l2_total_accesses', benchmark, pdf)
        hi.plot_page_intra(df, 'l1D_miss_rate', benchmark, pdf)
        hi.plot_page_intra(df, 'avg_mem_lat', benchmark, pdf)


def print_intra_inter(df_intra, df_inter, benchmark):
    filename = '{0}-{1}.pdf'.format(benchmark, 'both')
    filename = os.path.join('plots', filename)
    with PdfPages(filename) as pdf:
        hi.plot_page_intra_inter(df_intra, df_inter, 'norm_ipc', benchmark, pdf)


def parse_args():
    parser = argparse.ArgumentParser('Generate heatmaps for intra runs.')
    parser.add_argument('--pickle',
                        default=os.path.join(const.DATA_HOME, 'pickles/intra.pkl'),
                        help='Pickle that stores all the intra info.')
    parser.add_argument('--content', choices=['metrics', 'ipc'],
                        default='ipc',
                        help='metrics: print all relevant metrics per benchmark. ipc: print IPC heatmaps only.')
    parser.add_argument('--benchmark',
                        default=['all'],
                        nargs='+',
                        help='Individual benchmark to print heatmaps')
    parser.add_argument('--maxwidth',
                        type=int,
                        default=14,
                        help='Max width in grid for matplotlib')
    parser.add_argument('--figsize',
                        nargs='+',
                        type=int,
                        default=[30, 30],
                        help='Dimension of the figure. Used for ipc only mode.')

    results = parser.parse_args()
    return results


def print_ipc_only(df, benchmarks, maxGridWidth, figsize):
    print(benchmarks)
    fig_tot = plt.figure(figsize=figsize)

    cols = 0
    rows = 1
    for bench in benchmarks:
        width = len(df[df['pair_str'] == bench]['intra'].unique())
        running_sum = cols + width
        if running_sum > maxGridWidth:
            rows += 1
            cols = width
        else:
            cols = running_sum

    gs = mpl.gridspec.GridSpec(rows, maxGridWidth)

    gs_height = 0
    gs_width = 0

    #for ax, bench in zip(axs, benchmarks):
    for bench in benchmarks:
        _df = df[df['pair_str'] == bench]

        gs_height, gs_width = hi.plot_heatmap(_df, x_key='intra', y_key='l2', z_key='norm_ipc', 
                title=bench, scale=1.5, gs=gs, gs_height=gs_height, gs_width=gs_width, gs_width_max= maxGridWidth)

    plt.tight_layout()
    # fig_tot.suptitle('Intra, Normalized IPC', fontsize=18)
    fig_tot.savefig(os.path.join(const.DATA_HOME, 'graphs/total.pdf'))
    plt.close()


def main():
    args = parse_args()

    df_intra = pd.read_pickle(args.pickle)
    df_intra.sort_values('pair_str', inplace=True)

    if args.benchmark[0] == 'all':
        bench_list = df_intra['pair_str'].unique()
        args.benchmark = bench_list

    if args.content == 'metrics':
        for bench in args.benchmark:
            print_intra(df_intra, bench)
    elif args.content == 'ipc':
        print_ipc_only(df_intra, args.benchmark, args.maxwidth, tuple(args.figsize))


if __name__ == '__main__':
    main()
