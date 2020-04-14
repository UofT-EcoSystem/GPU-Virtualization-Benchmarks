import argparse
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math


def parse_args():
    parser = argparse.ArgumentParser('Generate utilization over time graph '
                                     'from nsight compute outputs.')
    parser.add_argument('--csv',
                        required=True,
                        help='Nsight compute CSV file to parse')

    parser.add_argument('--points', type=int,
                        help='Number of points in timeline')

    parser.add_argument('--outfile',
                        help='Output file path for the graph')

    results = parser.parse_args()
    return results


def read_csv(csv):
    n_skip = 0
    with open(csv) as f:
        for num, line in enumerate(f):
            if '\"ID\",\"Process ID\"' in line:
                n_skip = num

    df = pd.read_csv(csv, skiprows=n_skip, delimiter=',', thousands=',')
    return df


def draw_timeline(df, num_points, outfile):
    # See how many subplots we need
    list_metrics = list(df['Metric Name'].unique())

    if 'Duration' not in list_metrics:
        print('You must profile duration of kernels to plot timeline.')
        sys.exit(1)

    df_duration = df[df['Metric Name'] == 'Duration'].copy()

    # Check if units are all nsecond
    units = list(df_duration['Metric Unit'].unique())
    if len(units) > 1 or units[0] != 'nsecond':
        print('Duration unit is not consistent in nsecond.')
        sys.exit(2)

    # convert duration to usecond
    df_duration['Metric Value'] = df_duration['Metric Value'] / 1e3

    df_duration.sort_values('ID', inplace=True, ascending=True)
    total_duration_us = df_duration['Metric Value'].sum()
    precision = total_duration_us / (num_points - 1)

    xs = np.linspace(0, total_duration_us, num=num_points)
    # ys stores id of kernels
    ys = np.zeros((num_points, 1))
    running_idx = 0

    for index, row in df_duration.iterrows():
        # calculate how many points this kernel needs
        kernel_span = math.floor(row['Metric Value'] / precision)
        if kernel_span == 0:
            print('Number of points are too few.')
            sys.exit(3)

        ys[running_idx: running_idx + kernel_span] = row['ID']
        running_idx = running_idx + kernel_span

    list_metrics.remove('Duration')
    list_metrics.remove('Block Size')
    list_metrics.remove('Grid Size')

    subplot_nrows = len(list_metrics)
    subplot_ncols = 1
    f, axs = plt.subplots(subplot_nrows, subplot_ncols,
                          figsize=(15, 8*len(list_metrics)))
    plt.style.use('seaborn-talk')
    plt.subplots_adjust(hspace=0.3)

    for idx, metric in enumerate(list_metrics):
        plt.subplot(subplot_nrows, subplot_ncols, idx+1)
        df_metric = df[df['Metric Name'] == metric]
        df_metric = df_metric.set_index('ID')

        def util_from_id(x):
            return df_metric.loc[x, 'Metric Value']
        vfunc = np.vectorize(util_from_id)

        y_metric = vfunc(ys)
        plt.plot(xs, y_metric)
        plt.title(metric)
        plt.xlabel('usec')

        if df_metric.loc[0, 'Metric Unit'] == '%':
            plt.ylim([0, 100])
            plt.ylabel('%')

    plt.savefig(outfile, bbox_inches='tight')


def main():
    # Parse arguments
    args = parse_args()

    # Parse csv file into dataframe
    df = read_csv(args.csv)

    draw_timeline(df, args.points, args.outfile)


if __name__ == "__main__":
    main()
