import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

import data.scripts.common.help_iso as hi
import data.scripts.common.constants as const

mpl.style.use('seaborn-paper')

cmap_green = sns.cubehelix_palette(rot=-.4, dark=0.3, as_cmap=True)
cmap_purple = sns.cubehelix_palette(rot=.1, as_cmap=True)


def plot_heatmap_2d(df, x_key, y_key, z_key, title, scale,
                    gs, gs_height, gs_width, gs_width_max, cmap=cmap_green):
    df = df.sort_values([y_key, x_key], ascending=[False, True])

    num_cols = len(df[y_key].unique())

    data = np.split(df[z_key].values, num_cols)

    if hi.type_table.get(z_key, float) == np.int64:
        fmt = 'd'
    else:
        fmt = '.4f'

    sns.set(font_scale=scale)

    gs_width_end = gs_width + len(df[x_key].unique())
    if gs_width_end > gs_width_max:
        gs_height += 1
        gs_width = 0
        gs_width_end = len(df[x_key].unique())
    axis = plt.subplot(gs[gs_height, gs_width:gs_width_end])
    sns.heatmap(data, ax=axis, linewidth=0.2, linecolor='white',
                square=True, cmap=cmap,  # vmin=cbar_lim[0], vmax=cbar_lim[1],
                xticklabels=df[x_key].unique(), yticklabels=df[y_key].unique(),
                annot=True, fmt=fmt,
                cbar=False,
                cbar_kws={'label': hi.metric_label[z_key]}
                )
    axis.set_xlabel(hi.metric_label[x_key])
    axis.set_ylabel(hi.metric_label[y_key])
    axis.set_title(title)

    return gs_height, gs_width_end


def plot_heatmap_1d(df, x_key, z_key, title, scale,
                    gs, gs_height, gs_width, gs_width_max, cmap=cmap_green):
    df = df.sort_values(x_key, ascending=True)
    data = df[z_key].values.reshape(1, -1)

    if hi.type_table.get(z_key, float) == np.int64:
        fmt = 'd'
    else:
        fmt = '.4f'

    sns.set(font_scale=scale)

    gs_width_end = gs_width + len(df[x_key].unique())
    if gs_width_end > gs_width_max:
        gs_height += 1
        gs_width = 0
        gs_width_end = len(df[x_key].unique())
    axis = plt.subplot(gs[gs_height, gs_width:gs_width_end])

    sns.heatmap(data, ax=axis, linewidth=0.2, linecolor='white',
                square=True, cmap=cmap,  # vmin=cbar_lim[0], vmax=cbar_lim[1],
                xticklabels=df[x_key].unique(),
                annot=True, fmt=fmt,
                cbar=False,
                cbar_kws={'label': hi.metric_label[z_key]}
                )

    axis.set_xlabel(hi.metric_label[x_key])
    axis.set_title(title)

    return gs_height, gs_width_end


def plot_line(df, x_key, y_key, z_key, title, axis, scale):
    # Seaborn is being silly with hue of line plot
    # Have to manually add some garbage string at the end so that it's
    # treated as categorical data
    _df = df.copy()
    legend_category = z_key + '_c'
    _df[legend_category] = \
        _df[z_key].apply(lambda x:
                         '{0} {1}'.format(x, hi.metric_label[z_key])).astype(
            'category')

    sns.set(font_scale=scale)

    sns.lineplot(x=x_key, y=y_key, hue=legend_category, data=_df, ax=axis)
    axis.set_xlabel(hi.metric_label[x_key])
    axis.set_ylabel(hi.metric_label[y_key])
    axis.set_title(title)
    axis.xaxis.grid()


def plot_page_intra_inter(df_intra, df_inter, metric_key, benchmark, pdf):
    _df_intra = df_intra[df_intra['pair_str'] == benchmark]
    _df_inter = df_inter[df_inter['pair_str'] == benchmark]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 30))
    fig.suptitle(benchmark + ': ' + hi.metric_label[metric_key])

    # plotting heat maps
    kwargs = {'df': _df_intra, 'x_key': 'intra', 'y_key': 'l2',
              'z_key': metric_key, 'title': 'Intra-SM', 'axis': ax1,
              'scale': 1.2, 'cmap': cmap_green}

    plot_heatmap_2d(**kwargs)

    kwargs['df'] = _df_inter
    kwargs['x_key'] = 'inter'
    kwargs['title'] = 'Inter-SM'
    kwargs['axis'] = ax2
    kwargs['cmap'] = cmap_purple

    plot_heatmap_2d(**kwargs)

    # plotting line graphs
    kwargs['df'] = _df_intra
    kwargs['x_key'] = 'l2'
    kwargs['y_key'] = metric_key
    kwargs['z_key'] = 'intra'
    kwargs['title'] = 'Intra-SM'
    kwargs['axis'] = ax3
    kwargs.pop('cmap')

    plot_line(**kwargs)

    kwargs['df'] = _df_inter
    kwargs['z_key'] = 'inter'
    kwargs['title'] = 'Inter-SM'
    kwargs['axis'] = ax4

    plot_line(**kwargs)

    pdf.savefig(fig)
    plt.close()


def plot_page_intra(df, metric_key, benchmark, pdf):
    _df = df[df['pair_str'] == benchmark]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
    fig.suptitle(benchmark + ': ' + hi.metric_label[metric_key])

    plot_heatmap_2d(_df.copy(), x_key='intra', y_key='l2', z_key=metric_key,
                    title='Intra-SM Heatmap', axis=ax1,
                    scale=1.4, cmap=cmap_green,
                    )

    plot_line(_df.copy(), x_key='l2', y_key=metric_key, z_key='intra',
              title='Intra-SM Line', axis=ax2,
              scale=1.4
              )

    pdf.savefig(fig)
    plt.close()


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
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/intra.pkl'),
                        help='Pickle that stores all the intra info.')
    parser.add_argument('--content', choices=['metrics', 'ipc'],
                        default='ipc',
                        help='metrics: print all relevant metrics per '
                             'benchmark. ipc: print IPC heatmaps only.')
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
    parser.add_argument('--output',
                        default=os.path.join(const.DATA_HOME,
                                             'graphs/heatmap.pdf'),
                        help='Path to output file.')

    results = parser.parse_args()
    return results


def print_ipc_only(df, benchmarks, maxGridWidth, figsize, outfile):
    print(benchmarks)
    fig_tot = plt.figure(figsize=figsize)
    bench_keys = df[['pair_str', 'kidx']].apply(tuple, axis=1)

    cols = 0
    rows = 1
    for bench in benchmarks:
        width = len(df[bench_keys == bench]['intra'].unique())
        running_sum = cols + width
        if running_sum > maxGridWidth:
            rows += 1
            cols = width
        else:
            cols = running_sum

    gs = mpl.gridspec.GridSpec(rows, maxGridWidth)

    gs_height = 0
    gs_width = 0

    # for ax, bench in zip(axs, benchmarks):
    for bench in benchmarks:
        _df = df[bench_keys == bench]

        gs_height, gs_width = plot_heatmap_1d(_df, x_key='intra',
                                              z_key='norm_ipc', title=bench,
                                              scale=1.5, gs=gs,
                                              gs_height=gs_height,
                                              gs_width=gs_width,
                                              gs_width_max=maxGridWidth)

    plt.tight_layout()
    # fig_tot.suptitle('Intra, Normalized IPC', fontsize=18)
    fig_tot.savefig(outfile)
    plt.close()


def main():
    args = parse_args()

    df_intra = pd.read_pickle(args.pickle)
    df_intra.sort_values(['pair_str', 'kidx'], inplace=True)

    if args.benchmark[0] == 'all':
        pairs = df_intra.apply(lambda row: (row['pair_str'], row['kidx']),
                               axis=1)
        bench_list = pairs.unique()
        args.benchmark = bench_list
    else:
        # Turn benchmarks into tuple (bench, kidx)
        tuple_bench = []
        for app in args.benchmark:
            split_app = app.split(':')
            if len(split_app) > 1:
                tuple_bench.append((split_app[0], int(split_app[1])))
            else:
                tuple_bench.append((app, 0))

        args.benchmark = tuple_bench

    if args.content == 'metrics':
        for bench in args.benchmark:
            print_intra(df_intra, bench)
    elif args.content == 'ipc':
        print_ipc_only(df_intra, args.benchmark, args.maxwidth,
                       tuple(args.figsize), args.output)


if __name__ == '__main__':
    main()
