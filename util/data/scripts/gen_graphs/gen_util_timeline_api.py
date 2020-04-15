import argparse
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math


# ID map
API_ID = {
    '[CUDA memset]': 'C1',
    '[CUDA memcpy HtoD]': 'C2',
    '[CUDA memcpy DtoD]': 'C3',
    '[CUDA memcpy DtoH]': 'C4',
}

def parse_args():
    parser = argparse.ArgumentParser('Generate utilization over time graph '
                                     'from nsight compute outputs.')
    parser.add_argument('--csv',
                        required=True,
                        help='Nsight compute CSV file to parse')

    parser.add_argument('--time_csv',
                        required=True,
                        help='Nvprof GPU trace')

    parser.add_argument('--points', type=int,
                        help='Number of points in timeline')

    parser.add_argument('--outfile',
                        help='Output file path for the graph')

    results = parser.parse_args()
    return results


def read_nsight_csv(csv):
    n_skip = 0
    with open(csv) as f:
        for num, line in enumerate(f):
            if '\"ID\",\"Process ID\"' in line:
                n_skip = num

    df = pd.read_csv(csv, skiprows=n_skip, delimiter=',', thousands=',')
    return df


def read_nvprof_csv(csv):
    n_skip = 0

    with open(csv) as f:
        for num, line in enumerate(f):
            if '\"Start\",\"Duration\"' in line:
                n_skip = num

    def skip_row(x):
        return (x < n_skip) or (x == n_skip + 1)

    df = pd.read_csv(csv, skiprows=skip_row, delimiter=',', thousands=',')
    return df


def build_trace(df_time, num_points):
    end = df_time.iloc[-1]['Start'] + df_time.iloc[-1]['Duration']
    start = df_time.at[0, 'Start']
    total_duration_us = end - start
    precision = total_duration_us / (num_points - 1)

    xs = np.linspace(0, total_duration_us, num=num_points)
    # ys stores id of kernels
    ys = np.zeros((num_points, 1))
    # color -> array of x ranges
    api_color = {}
    for c in API_ID.values():
        api_color[c] = []

    running_idx = 0
    previous_end = 0

    for index, row in df_time.iterrows():
        # calculate starting idx
        start_idx = math.floor((row['Start'] - start) / precision)

        if start_idx < previous_end:
            print('Duration runs into previous block.')
            sys.exit(3)

        # calculate how many points this kernel needs
        span = math.floor(row['Duration'] / precision)
        if span == 0:
            print('Number of points are too few.')
            sys.exit(4)

        previous_end = start_idx + span

        # Check if it's a kernel or CUDA API
        if pd.isna(row['Grid X']):
            # This is a CUDA API
            ys[start_idx: previous_end] = -1
            color = API_ID[row['Name']]
            api_color[color].append([start_idx, previous_end])
        else:
            # This is a kernel row
            ys[start_idx: previous_end] = running_idx
            running_idx += 1

    return xs, ys, api_color


def draw_timeline(df_pc, xs, ys, api_color, outfile):
    list_metrics = list(df_pc['Metric Name'].unique())

    # if 'Duration' not in list_metrics:
    #     print('You must profile duration of kernels to plot timeline.')
    #     sys.exit(1)

    list_metrics.remove('Duration')
    list_metrics.remove('Block Size')
    list_metrics.remove('Grid Size')

    subplot_nrows = len(list_metrics)
    subplot_ncols = 1
    f, axs = plt.subplots(subplot_nrows, subplot_ncols,
                          figsize=(35, 8 * len(list_metrics)))
    plt.style.use('seaborn-talk')
    plt.subplots_adjust(hspace=0.3)

    for idx, metric in enumerate(list_metrics):
        plt.subplot(subplot_nrows, subplot_ncols, idx + 1)
        df_metric = df_pc[df_pc['Metric Name'] == metric]
        df_metric = df_metric.set_index('ID')

        def util_from_id(x):
            if x < 0:
                # CUDA API, fill color and take value zero
                return 0
            else:
                return df_metric.loc[x, 'Metric Value']

        vfunc = np.vectorize(util_from_id)

        if (df_metric['Metric Value'] == 0).all():
            y_metric = np.zeros(ys.shape)
        else:
            y_metric = vfunc(ys)

        plt.plot(xs, y_metric)
        plt.title(metric)
        plt.xlabel('usec')

        # Color different CUDA API regions
        for color in api_color:
            for x_range in api_color[color]:
                plt.axvspan(xs[x_range[0]], xs[x_range[1]], color=color,
                            alpha=0.5, ymax=0.2)

        plt.ylim(bottom=0)
        if df_metric.loc[0, 'Metric Unit'] == '%':
            plt.ylim([0, 100])
            plt.ylabel('%')

    plt.savefig(outfile, bbox_inches='tight')


def main():
    # Parse arguments
    args = parse_args()

    # Parse csv file into dataframe
    df_pc = read_nsight_csv(args.csv)
    df_time = read_nvprof_csv(args.time_csv)

    xs, ys, api_color = build_trace(df_time, args.points)
    draw_timeline(df_pc, xs, ys, api_color, args.outfile)


if __name__ == "__main__":
    main()
