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

from primitives.draw import *


def parse_csv(directory, graph_type):
    type_dic = {'comptime': 'comp', 'memtime': 'mem'}
    results = {}
    types = graph_type + ['time']

    for t in types:
        if t in type_dic:
            t = type_dic[t]
        if t == 'sheet':
            continue

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

def print_kernel(kernels, name_len):
    # list out the kernel names
    plt.subplots_adjust(bottom=name_len)
    for i in np.arange(kernels.shape[0]):
        dm = cxxfilt.demangle(kernels[i])
        name = dm
        name = re.sub(">\((.+)\)$", '>', dm, 1)
        #name = re.sub("^([a-z]+)\s", '', name, 1)
        plt.text(0.1, name_len-0.06-i/kernels.shape[0]*(name_len), 
                '{}: {}'.format(i, name), transform=plt.gcf().transFigure, fontsize=16)

def time(time_df):
    # drop rows with NaN (cudamemset has NaN grid size)
    time_df = time_df.dropna(subset=['Grid X'])

    # get the total runtime
    #total = time_df.iloc[-1]['Start'] + time_df.iloc[-1]['Duration'] - time_df.iloc[0]['Start']
    total = time_df['Duration'].sum()

    # group duration of the kernels
    time_df = time_df[['Name', 'Duration']]
    result = time_df.groupby(['Name'], as_index=False).sum()
    result['Duration'] = result['Duration'] / total * 100
    result.rename(columns={'Name': 'Kernel Name', 'Duration':'Importance'}, inplace=True)
    print('kernels in time df: {}'.format(result['Kernel Name'].unique().shape))

    return result

def nsight2nvprof(df_nsight, df_nvprof):
    kmap = {}

    nsight = df_nsight['Kernel Name'].unique()
    nvprof = df_nvprof['Kernel Name'].unique()

    for name_nsight in nsight:
        regex = '([^a-zA-Z]|^)' + name_nsight + '([^a-zA-Z]|$)'
        for name_nvprof in nvprof:
            if re.search(regex, name_nvprof):
                # one name in nsight tool can appear in multiple names in nvprof :/
                if name_nsight not in kmap:
                    kmap[name_nsight] = []
                kmap[name_nsight].append(name_nvprof)

    return kmap

def join(kmap, df_time, df, drop_threshold, util=False):
    # kernel names produced by nvprof and nsight are slightly different 
    # need to manually equate them
    col_import = []
    matches = []

    for index, row in df.iterrows():
        match_df = []

        for prof_name in kmap[row['Kernel Name']]:
            match_df.append(df_time[df_time['Kernel Name'] == prof_name])

        matched = pd.concat(match_df)

        # append importance to list
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
    if diff.size != 0:
        print(diff['Kernel Name'])

    # xticks with importance
    xticks = ['{}\n{:.2f}%'.format(i, result.iloc[i]['Importance']) for i in range(result.shape[0])]

    # drop rows with importance below 1%
    neglible = result[ result['Importance'] < drop_threshold ].index
    result.drop(neglible, inplace=True)

    # drop the importance column
    result = result.drop(columns='Importance')

    return result, xticks

def instmix(args, df_time, df, kmap):
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
    df_join, xticks = join(kmap, df_time, trans_df, args.drop_threshold)

    # normalize each benchmark mixture and scale it by importance
    df_join[metrics] = df_join[metrics].div(df_join['EXEC'], axis=0)
    #.multiply(df_join['Importance'], axis=0)
                                       
    metrics.remove('EXEC')
    df_join['Other'] = df_join['EXEC'].subtract(df_join[metrics].sum(axis=1))

    cols = ['Kernel Name', 'FP16', 'FMA', 'FP64', 'INT', 
            'SPU', 'Tensor', 'Load/Store', 'Other' ]
    df_join = df_join[cols]

    results = df_join.values

    # set graph size
    plt.rcParams["figure.figsize"] = (args.width, args.height)

    # print the list of kernels
    print_kernel(results[:, 0], args.name_len)

    legends = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Ld/St', 'Other']
    draw_stack(results, legends, bench_name.capitalize(), 'Kernel ID', 
            'Normalized Instruction Count Scaled by Runtime Weight',
            xticks=xticks,
            outfile=os.path.join(outdir, 'inst.pdf'))

def comp(args, df_time, df, kmap):
    bench_name = args.name
    outdir = args.outdir if args.outdir else args.indir

    # process each benchmark individually
    k_metrics = df.groupby(['Kernel Name','Metric Name'], as_index=False)[['Metric Value']].mean()
    #k_metrics = k_metrics.reset_index(level=['Kernel Name','Metric Name'])

    df_map = {}
    df_map['Kernel Name'] = k_metrics['Kernel Name'].unique()

    metrics = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Load/Store', 'Eligible']

    # I really hope no one sees this: re-organize the table
    for metric in metrics:
        metric_col = k_metrics.loc[k_metrics['Metric Name'] == metric]['Metric Value'].array
        df_map[metric] = metric_col

    trans_df = pd.DataFrame(df_map)
    cols = ['Kernel Name'] + metrics
    trans_df = trans_df[cols]

    df_join, xticks = join(kmap, df_time, trans_df, args.drop_threshold)

    #df_join = df_join[cols]

    results = df_join.values

    # set graph size
    plt.rcParams["figure.figsize"] = (args.width, args.height)

    legends = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Ld/St', 'Eligible']
    draw_bar(results, legends, bench_name, 'Kernel ID', 
            'Compute Utilization during Active Cycles (%)',
            xticks=xticks,
            outfile=os.path.join(outdir, 'comp.pdf'))

    return results, legends

def mem(args, df_time, df, kmap):
    bench_name = args.name
    outdir = args.outdir if args.outdir else args.indir

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

    # adjust memory usage according to total available amount
    trans_df['REGISTER_USAGE'] = trans_df['REGISTER_USAGE'] * 1024 * trans_df['OCCUPANCY'] / (64 * 1024)
    blocks_per_sm = trans_df['OCCUPANCY'] * 32 * 32 / trans_df['Block Size']
    trans_df['SHARED_USAGE'] = (trans_df['DYNAMIC_SHARED_USAGE'] 
                               + trans_df['STATIC_SHARED_USAGE'] ) * (blocks_per_sm) / trans_df['Shm Config Size']

    #'L2 Hit Rate' is bogus, bug in nsight
    cols = ['DRAM_UTIL_PCT', 'REGISTER_USAGE', 'SHARED_USAGE', 
             'OCCUPANCY', 'L1 Hit Rate']
    trans_df = trans_df[['Kernel Name'] + cols]

    df_join, xticks = join(kmap, df_time, trans_df, args.drop_threshold)
    #df_join = df_join.drop(columns='Kernel Name')

    results = df_join.values

    plt.rcParams["figure.figsize"] = (args.width, args.height)

    # rename dram util
    cols[0] = 'DRAM_BANDWIDTH_UTIL'

    draw_bar(results, cols, bench_name, 'Kernel ID', 
            'Memory Utilization during Active Cycles (%)',
            xticks=xticks,
            outfile=os.path.join(outdir, 'mem.pdf'))

    return results, cols

def preprocess_util2time(df_time, df, kmap, metrics, legends, util_type):
    num_points = 1000000 # 1M points to represent the entire timeline
    start = math.floor(float(df_time.iloc[0]['Start']))
    end = math.ceil(float(df_time.iloc[-1]['Start']) + float(df_time.iloc[-1]['Duration']))
    prec = (end - start) / (num_points-1)
    xs = np.linspace(start, end, num=num_points) / 1000
    ys = np.zeros((num_points, len(legends)))

    df_time['left'] = (np.floor((df_time['Start'] - start)/prec)).values
    df_time['right'] = (np.ceil((df_time['Start'] + df_time['Duration'] - start)/prec)).values

    # for each kernel name, update ys
    for name in kmap:
        comp_values = df[(df['Kernel Name'] == name) ]['Metric Value'].values
        if comp_values.shape[0] % len(metrics) != 0:
            print(comp_values.shape[0])
            print(len(metrics))
            print(df.head(10))
            print('comp_values shape mismatch')
            exit(1)

        slice_size = comp_values.shape[0] / len(metrics)
        comp_values = np.array(np.split(comp_values, slice_size))

        if util_type == 'mem':
                #    # adjust memory usage according to total available amount
    #    trans_df['REGISTER_USAGE'] = trans_df['REGISTER_USAGE'] * 1024 * trans_df['OCCUPANCY'] / (64 * 1024)
    #    blocks_per_sm = trans_df['OCCUPANCY'] * 32 * 32 / trans_df['Block Size']
    #    trans_df['SHARED_USAGE'] = (trans_df['DYNAMIC_SHARED_USAGE'] 
    #            + trans_df['STATIC_SHARED_USAGE'] ) * (blocks_per_sm) / trans_df['Shm Config Size']

            # handle register usage
            comp_values[:, metrics.index('REGISTER_USAGE')] = comp_values[:, metrics.index('REGISTER_USAGE')] * 1024 \
                    * comp_values[:, metrics.index('OCCUPANCY')] / (64 * 1024)

            # handle shared memory usage
            blocks_per_sm = comp_values[:, metrics.index('OCCUPANCY')] * 32 * 32 / comp_values[:, metrics.index('Block Size')]
            comp_values[:,metrics.index('Shm Config Size')] =  \
                    (comp_values[:, metrics.index('DYNAMIC_SHARED_USAGE')] + comp_values[:,metrics.index('STATIC_SHARED_USAGE')]) * \
                    blocks_per_sm / comp_values[:, metrics.index('Shm Config Size')]

            #print('before', comp_values)
            comp_values = np.delete(comp_values, 1, axis=1)
            comp_values = np.delete(comp_values, 3, axis=1)
            comp_values = np.delete(comp_values, 3, axis=1)
            #print('after', comp_values)
            #exit(1)

        # get start and end index of the kernel
        idx_df = []
        for prof_name in kmap[name]:
            idx_df.append(df_time[df_time['Name'] == prof_name])
        idx_df = pd.concat(idx_df)

        if comp_values.shape[0] != idx_df.shape[0]:
            print(name)
            print(comp_values.shape[0], idx_df.shape[0])
            print("Error: nsight and nvprof shape mismatches")
            exit()
            #print(kmap[name])
            #print(comp_values.shape, idx_df.shape)

        i = 0
        for index, row in idx_df.iterrows():
            l = int(row['left'])
            r = int(row['right'])
            v = comp_values[i, :]

            ys[l:r, :] = ys[l:r, :] + v


            i += 1

    return xs, ys

def comp2time(args, df_time, df, kmap):
    bench_name = args.name
    outdir = args.outdir if args.outdir else args.indir

    # constants
    metrics = ['ALU', 'FMA', 'FP16', 'FP64', 'Load/Store', 'Tensor', 'SPU', 'Eligible']
    legends = metrics

    xs, ys = preprocess_util2time(df_time, df, kmap, metrics, legends, 'comp')

    plt.rcParams["figure.figsize"] = (args.width, args.height)

    draw_plot(xs, ys, metrics, bench_name, 'Time', 
            'Compute Utilization during Active Cycles (%)', 
            outfile=os.path.join(outdir, 'comp-time.pdf'))

def mem2time(args, df_time, df, kmap):
    bench_name = args.name
    outdir = args.outdir if args.outdir else args.indir

    df = df.drop(df[df['Metric Name'] == 'L2 Hit Rate'].index)

    metrics = ['DRAM_UTIL_PCT', 'Block Size', 'REGISTER_USAGE', 'Shm Config Size', 
            'DYNAMIC_SHARED_USAGE', 
            'STATIC_SHARED_USAGE', 'OCCUPANCY', 
            'L1 Hit Rate']

    legends = ['DRAM_UTIL_PCT', 'REGISTER_USAGE', 
            'SHARED_USAGE', 'OCCUPANCY', 'L1 Hit Rate']


    print('invoks in mem: {}'.format(df['ID'].unique().shape[0]))
    print('invoks in time: {}'.format(df_time.dropna(subset=['Grid X']).shape[0]))
    xs, ys = preprocess_util2time(df_time, df, kmap, metrics, legends, 'mem')

    plt.rcParams["figure.figsize"] = (args.width, args.height)

    # rename dram util
    legends[0] = 'DRAM_BANDWIDTH_UTIL'

    draw_plot(xs, ys, legends, bench_name, 'Time', 
            'Memory Utilization during Active Cycles (%)', 
            outfile=os.path.join(outdir, 'mem-time.pdf'))



def main():
    # cmd parser
    parser = argparse.ArgumentParser('Parse nsight compute raw csv output.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    types = ['inst', 'comp', 'mem', 'comptime', 'memtime', 'sheet']
    parser.add_argument('--types', required=True, nargs='+', 
            choices=types, help='Types of plots to make.')

    parser.add_argument('--indir', required=True, 
            help='Input directory of nsight compute dumps')
    parser.add_argument('--outdir', required=False, 
            help='Output directory of graphs. Default is the same as input dir.')

    parser.add_argument('--name', required=True,
            help='Title of the generated graph.')

    parser.add_argument('--drop_threshold', type=float, default=0, 
            help='Drop kernels that have a weight less than this threshold.')

    parser.add_argument('--width', type=int, default=30,
            help='Width of the the graph.')

    parser.add_argument('--height', type=int, default=20,
            help='Width of the the graph.')

    parser.add_argument('--name_len', type=float, default=0.62,
            help='Portion of inst graph for kernel name.')

 
    # parse arguments
    args = parser.parse_args()

    # parse all text files including timeline information
    df_read = parse_csv(args.indir, args.types)

    # get kernel importance from runtime info
    df_time = time(df_read['time'])

    # get kernel name map
    kmap = nsight2nvprof(next(iter(df_read.items()))[1], df_time)

    if 'inst' in args.types:
        # generate plot for instruction mix
        instmix(args, df_time, df_read['inst'], kmap)

    if 'comp' in args.types:
        # generate plot for compute unit utilization 
        comp_val, comp_legends = comp(args, df_time, df_read['comp'], 
                kmap)

    if 'mem' in args.types:
        # generate plot for memory unit utilization 
        mem_val, mem_legends = mem(args, df_time, df_read['mem'], kmap)

    if 'comptime' in args.types:
        comp2time(args, df_read['time'], df_read['comp'], kmap)

    if 'memtime' in args.types:
        mem2time(args, df_read['time'], df_read['mem'], kmap)

    if 'sheet' in args.types:
        if comp_val is None or mem_val is None:
            print("Comp and mem must be selected when using sheet.")
        else:
            # combine values
            dump = np.concatenate((mem_val, comp_val[:, 1:]), axis=1)
            dump_col = ['Kernel'] + mem_legends + comp_legends
            dump = pd.DataFrame(data=dump, columns=dump_col)
            dump = dump.drop('L1 Hit Rate', axis=1)
            dump = dump.drop('Eligible', axis=1)

            dump.insert(loc=0, column='Benchmark_Set', value=args.name)

            dump.to_csv(os.path.join(args.indir, 'sheet.csv'), sep=',', 
                    index=False)
if __name__ == '__main__':
    main()


