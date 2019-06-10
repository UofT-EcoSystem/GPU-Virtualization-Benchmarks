import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import argparse
import cxxfilt
import re
import difflib

from primitives.draw_bar import *


def parse_csv(directory, graph_type):
    results = {}
    types = graph_type + ['time']

    for t in types:
        full_filename = os.path.join(directory, t+'.txt')

        # find out number of lines to skip in file
        n_skip = 0
        output_exists = False

        with open(full_filename) as search:
            for num, line in enumerate(search, 1):
                # Sketchy, might break
                if '\"Metric Value\"' in line or '\"Duration\"' in line:
                    n_skip = num - 1
                    output_exists = True
                    
        if not output_exists:
            print(full_filename, "went wrong!")
            continue

        print("Parsing", full_filename)

        # read data using pandas
        if t == 'time':
            skip_call = lambda x: x in range(n_skip) or x == (n_skip + 1)
        else:
            skip_call = lambda x: x in range(n_skip) 

        data = pd.read_csv(full_filename, delimiter=',', skiprows=skip_call, thousands=',')
        results[t] = data

    # return a map of metric type to pandas dataframe
    return results

def print_kernel(kernels):
    # list out the kernel names
    plt.subplots_adjust(bottom=0.62)
    for i in np.arange(kernels.shape[0]):
        dm = cxxfilt.demangle(kernels[i])
        name = dm
        name = re.sub(">\((.+)\)$", '>', dm, 1)
        #name = re.sub("^([a-z]+)\s", '', name, 1)
        plt.text(0.1, 0.6-i/kernels.shape[0]*0.4, '{}: {}'.format(i, name), transform=plt.gcf().transFigure)

def time(time_df):
    # get the total runtime
    total = time_df.iloc[-1]['Start'] + time_df.iloc[-1]['Duration'] - time_df.iloc[0]['Start']

    # drop rows with NaN (cudamemset has NaN grid size)
    time_df = time_df.dropna(subset=['Grid X'])
    print('num invoks:{}'.format(time_df.shape))

    # group duration of the kernels
    time_df = time_df[['Name', 'Duration']]
    result = time_df.groupby(['Name'], as_index=False).sum()
    result['Duration'] = result['Duration'] / total * 100
    result.rename(columns={'Name': 'Kernel Name', 'Duration':'Importance'}, inplace=True)
    print('kernels in time df: {}'.format(result['Kernel Name'].unique().shape))

    return result

def join(df_time, df):
    # kernel names produced by nvprof and nsight are slightly different 
    # need to manually equate them
    col_import = []
    matches = []

    for index, row in df.iterrows():
        # append importance to list
        matched = df_time[df_time['Kernel Name'].str.contains(row['Kernel Name'])]
        if matched.size == 0:
            print('No matching kernel in time df!')
            print(row['Kernel Name'])
            exit(1)
            col_import.append(0)
        else:
            if matched.shape[0] != 1:
                print('{} has {} matches'.format(row['Kernel Name'], matched.shape[0]))
            matches.append(matched)
            col_import.append(matched['Importance'].sum())

    result = df
    result['Importance'] = col_import
    result = result.sort_values(by='Importance', ascending=False)

    # debug: find out which ones are leftover from time
    match_time = pd.concat(matches)
    diff = pd.concat([match_time, df_time]).drop_duplicates(keep=False)
    print(diff['Kernel Name'])

    return result

def instmix(args, df_time, df):
    bench_name = args.name
    outdir = args.outdir if args.outdir else args.indir

    # divide by 1 million
    df['Metric Value'] /= 1000000

    # process each benchmark individually
    k_metrics = df.groupby(['Kernel Name','Metric Name'], as_index=False)[['Metric Value']].sum()
    #k_metrics = k_metrics.reset_index(level=['Kernel Name','Metric Name'])

    df_map = {}
    df_map['Kernel Name'] = k_metrics['Kernel Name'].unique()
    print('kernels in inst:{} '.format(df_map['Kernel Name'].size))

    metrics = ['FP16', 'FMA', 'FP64', 'INT', 
            'SPU', 'Tensor', 'Load/Store', 'EXEC' ]

    # I really hope no one sees this: re-organize the table
    for metric in metrics:
       metric_col = k_metrics.loc[k_metrics['Metric Name'] == metric]['Metric Value'].array
       df_map[metric] = metric_col

    trans_df = pd.DataFrame(df_map)

    # inner join the two tables: importance and metric values
    # also sort the table rows by descending importance
    df_join = join(df_time, trans_df)

    # normalize each benchmark mixture and scale it by importance
    df_join[metrics] = df_join[metrics].div(df_join['EXEC'], axis=0) \
                                       .multiply(df_join['Importance'], axis=0)
    metrics.remove('EXEC')
    df_join['Other'] = df_join['EXEC'].subtract(df_join[metrics].sum(axis=1))

    cols = ['Kernel Name', 'FP16', 'FMA', 'FP64', 'INT', 
            'SPU', 'Tensor', 'Load/Store', 'Other' ]
    df_join = df_join[cols]

    results = df_join.values

    # set graph size
    plt.rcParams["figure.figsize"] = (25, 40)

    # print the list of kernels
    print_kernel(results[:, 0])

    legends = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Ld/St', 'Other']
    draw_stack(results, legends, bench_name.capitalize(), 'Kernel ID', 
            'Normalized Instruction Count Scaled by Runtime Weight',
            os.path.join(outdir, 'inst.pdf'))

def comp(args, df_time, df):
    # parse inputs
    df_read = parse_csv(args.indir)
    outdir = args.outdir if args.outdir else args.indir

    df_time = time(df_read['time'])

    df = df_read['util']
    bench_name = args.name

    # process each benchmark individually
    k_metrics = df.groupby(['Kernel Name','Metric Name'], as_index=False)[['Metric Value']].mean()
    #k_metrics = k_metrics.reset_index(level=['Kernel Name','Metric Name'])

    df_map = {}
    df_map['Kernel Name'] = k_metrics['Kernel Name'].unique()

    metrics = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Load/Store']

    # I really hope no one sees this: re-organize the table
    for metric in metrics:
        metric_col = k_metrics.loc[k_metrics['Metric Name'] == metric]['Metric Value'].array
        df_map[metric] = metric_col

    trans_df = pd.DataFrame(df_map)
    cols = ['Kernel Name'] + metrics
    trans_df = trans_df[cols]

    df_join = join(df_time, trans_df)

    results = df_join.values

    legends = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Ld/St']
    draw_bar(results, legends, bench_name, 'Kernel ID', 
            'Utilization during Active Cycles (%)',
            os.path.join(outdir, 'util', 'plot.pdf'), pwidth=25)

def mem(args, df_time, df):
    # parse inputs
    df_read = parse_csv(args.indir)
    outdir = args.outdir if args.outdir else args.indir

    aggr_result = pd.DataFrame()

    for benchmark in df_read:
        df = df_read[benchmark]
        bench_name = benchmark.split('.')[0]

        if args.aggregate:
            groups = df.groupby('Metric Name')[['Metric Value']].mean().T
            groups['Benchmark'] = bench_name
            #groups['Control'] = groups['ADU'] + groups['CBU']

            aggr_result = aggr_result.append(groups)
            print('unimplemented error')

        else:
            # process each benchmark individually
            k_metrics = df.groupby(['Kernel Name','Metric Name'], as_index=False)[['Metric Value']].mean()
            #k_metrics = k_metrics.reset_index(level=['Kernel Name','Metric Name'])

            df_map = {}
            df_map['Kernel Name'] = k_metrics['Kernel Name'].unique()

            metrics = ['DRAM_UTIL_PCT', 'REGISTER_USAGE', 'DYNAMIC_SHARED_USAGE', 
                    'STATIC_SHARED_USAGE', 'L2 Hit Rate', 'OCCUPANCY', 
                    'L1 Hit Rate', 'Block Size', 'Shm Config Size']

            # I really hope no one sees this: re-organize the table
            for metric in metrics:
                metric_col = k_metrics.loc[k_metrics['Metric Name'] == metric]['Metric Value'].array
                df_map[metric] = metric_col

            trans_df = pd.DataFrame(df_map)
            cols = ['Kernel Name'] + metrics
            trans_df = trans_df[cols]
            trans_df['REGISTER_USAGE'] = trans_df['REGISTER_USAGE'] * 1024 * trans_df['OCCUPANCY'] / (64 * 1024)
            blocks_per_sm = trans_df['OCCUPANCY'] * 32 * 32 / trans_df['Block Size']
            trans_df['SHARED_USAGE'] = (trans_df['DYNAMIC_SHARED_USAGE'] 
                                       + trans_df['STATIC_SHARED_USAGE'] ) * (blocks_per_sm) / trans_df['Shm Config Size']

            #'L2 Hit Rate' is bogus, bug in nsight
            cols = ['DRAM_UTIL_PCT', 'REGISTER_USAGE', 'SHARED_USAGE', 
                     'OCCUPANCY', 'L1 Hit Rate']
            trans_df = trans_df[['Kernel Name'] + cols]
            results = trans_df.values

            draw_bar(results, cols, bench_name, 'Kernel ID', 
                    'Memory Utilization during Active Cycles (%)',
                    os.path.join(outdir, bench_name+'-util.pdf'), pwidth=25)
 
def main():
    # cmd parser
    parser = argparse.ArgumentParser('Parse nsight compute raw csv output.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    types = ['inst', 'comp', 'mem']
    parser.add_argument('--types', required=True, nargs='+', 
            choices=types, help='Types of plots to make.')

    parser.add_argument('--indir', required=True, 
            help='Input directory of nsight compute dumps')
    parser.add_argument('--outdir', required=False, 
            help='Output directory of graphs. Default is the same as input dir.')

    parser.add_argument('--name', required=True,
            help='Title of the generated graph.')

    # parse arguments
    args = parser.parse_args()

    # parse all text files including timeline information
    df_read = parse_csv(args.indir, args.types)

    # get kernel importance from runtime info
    df_time = time(df_read['time'])

    if 'inst' in args.types:
        # generate plot for instruction mix
        instmix(args, df_time, df_read['inst'])
    elif 'comp' in args.types:
        # generate plot for compute unit utilization 
        comp(args, df_time, df_read['comp'])
    elif 'mem' in args.types:
        # generate plot for memory unit utilization 
        mem(args, df_time, df_read['mem'])

if __name__ == '__main__':
    main()


