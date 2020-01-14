import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from tabulate import tabulate
import re
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

mpl.style.use('seaborn-paper')

# bench -> C(0)/M(1)
bench_dict = {'cut_sgemm-0':0, 'cut_sgemm-1':0, 'cut_wmma-0': 0, 'cut_wmma-1': 0,
              'parb_stencil-0': 1, 'parb_sgemm-0': 0,
              'parb_lbm-0': 1, 'parb_spmv-0': 1, 'parb_cutcp-0': 0}

# each tuple contains: regex, dtype
regex_table = {'intra': r'INTRA_0:(.*):[0-9]+_CTA',
               'inter': r'INTER_0:(.*):[0-9]+_SM',
               'l2': r'PARTITION_L2_0:(.*):[0-9|\.]+',
               '1_intra': r'INTRA_0:(.*):[0-9]+_CTA',
               '1_inter': r'INTER_0:(.*):[0-9]+_SM',
               '1_l2': r'PARTITION_L2_0:(.*):[0-9|\.]+',
               '2_intra': r'INTRA_0:[0-9]+:(.*)_CTA',
               '2_inter': r'INTER_0:[0-9]+:(.*)_SM',
               '2_l2': r'PARTITION_L2_0:[0-9|\.]+:([0-9|\.]+)',
               }

type_table = {'intra': int,
              'inter': int,
              'l2': float,
              'norm_ipc': float,
              'avg_dram_bw': float,
              }

metric_label = {'intra': 'Concurrent CTAs/SM',
                'inter': '# of SMs',
                'l2': 'Norm. L2 Partition Size',
                'norm_ipc': 'Normalized IPC',
                'avg_dram_bw': 'DRAM Bandwidth Utilization',
                'dram_idleness': 'DRAM IDLE RATE',
                'l2_miss_rate': 'L2 Miss Rate',
                'l2_BW': 'L2 Bandwidth',
                'l2_total_accesses': 'L2 Total Accesses',
                'l1D_miss_rate': 'L1 Miss Rate',
                'avg_mem_lat': 'Average Memory Latency',
                }

cmap_green = sns.cubehelix_palette(rot=-.4, dark=0.3, as_cmap=True)
cmap_purple = sns.cubehelix_palette(rot=.1, as_cmap=True)


# map a panda column from vector cell to scalar cell by taking average
def avg_array(s):
    result = [np.average(np.array(v[1:-1].split(' ')).astype(float)) for v in s]
    return np.array(result)


def std_array(s):
    result = [np.std(np.array(v[1:-1].split(' ')).astype(float)) for v in s]
    return np.array(result)


def process_config_column(*configs, df):
    for c in configs:
        df[c] = df['config'].apply(lambda x: re.search(regex_table[c], x).group(1)).astype(type_table.get(c, float))


def normalize(df, index, metric, value, inverse):
    if inverse:
        return df.loc[index, metric] / value
    else:
        return value / df.loc[index, metric]


def plot_heatmap(df, x_key, y_key, z_key, title, scale, 
        gs, gs_height, gs_width, gs_width_max, cmap=cmap_green):
    df = df.sort_values([y_key, x_key], ascending=[False, True])

    num_cols = len(df[y_key].unique())

    data = np.split(df[z_key].values, num_cols)

    if type_table.get(z_key, float) == np.int64:
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
                cbar = False,
                cbar_kws={'label': metric_label[z_key]}
                )
    axis.set_xlabel(metric_label[x_key])
    axis.set_ylabel(metric_label[y_key])
    axis.set_title(title)

    return gs_height, gs_width_end


def plot_line(df, x_key, y_key, z_key, title, axis, scale):
    # Seaborn is being silly with hue of line plot
    # Have to manually add some garbage string at the end so that it's treated as categorical data
    _df = df.copy()
    legend_category = z_key + '_c'
    _df[legend_category] = _df[z_key].apply(lambda x: '{0} {1}'.format(x, metric_label[z_key])) \
                                        .astype('category')

    sns.set(font_scale=scale)

    sns.lineplot(x=x_key, y=y_key, hue=legend_category, data=_df, ax=axis)
    axis.set_xlabel(metric_label[x_key])
    axis.set_ylabel(metric_label[y_key])
    axis.set_title(title)
    axis.xaxis.grid()


def plot_page_intra(df, metric_key, benchmark, pdf):
    _df = df[df['pair_str'] == benchmark]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
    fig.suptitle(benchmark + ': ' + metric_label[metric_key])

    plot_heatmap(_df.copy(), x_key='intra', y_key='l2', z_key=metric_key,
                 title='Intra-SM Heatmap', axis=ax1,
                 scale=1.4, cmap=cmap_green,
                 )

    plot_line(_df.copy(), x_key='l2', y_key=metric_key, z_key='intra',
              title='Intra-SM Line', axis=ax2,
              scale=1.4
              )

    pdf.savefig(fig)
    plt.close()


def plot_page_intra_inter(df_intra, df_inter, metric_key, benchmark, pdf):
    _df_intra = df_intra[df_intra['pair_str'] == benchmark]
    _df_inter = df_inter[df_inter['pair_str'] == benchmark]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 30))
    fig.suptitle(benchmark + ': ' + metric_label[metric_key])

    # plotting heat maps
    kwargs = {'df': _df_intra, 'x_key': 'intra', 'y_key': 'l2', 'z_key': metric_key, 'title': 'Intra-SM', 'axis': ax1,
              'scale': 1.2, 'cmap': cmap_green}

    plot_heatmap(**kwargs)

    kwargs['df'] = _df_inter
    kwargs['x_key'] = 'inter'
    kwargs['title'] = 'Inter-SM'
    kwargs['axis'] = ax2
    kwargs['cmap'] = cmap_purple

    plot_heatmap(**kwargs)

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



