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


def parse_csv(filename, t):
    # find out number of lines to skip in file
    n_skip = 0
    output_exists = False

    with open(filename) as search:
        for num, line in enumerate(search, 1):
            # Sketchy, might break
            if '\"Metric Value\"' in line or '\"Duration\"' in line:
                n_skip = num - 1
                output_exists = True
                
    if not output_exists:
        print(filename, "went wrong!")
        return False, None

    print("Parsing", filename)

    # read data using pandas
    if t == 'time':
        skip_call = lambda x: x in range(n_skip) or x == (n_skip + 1)
    else:
        skip_call = lambda x: x in range(n_skip) 

    data = pd.read_csv(filename, delimiter=',', skiprows=skip_call, thousands=',')

    # return a map of metric type to pandas dataframe
    return True, data

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
    #total = time_df.iloc[-1]['Start'] + time_df.iloc[-1]['Duration'] - time_df.iloc[0]['Start']
    total = time_df['Duration'].sum()

    # drop rows with NaN (cudamemset has NaN grid size)
    time_df = time_df.dropna(subset=['Grid X'])

    # group duration of the kernels
    time_df = time_df[['Name', 'Duration']]
    result = time_df.groupby(['Name'], as_index=False).sum()
    result['Duration'] = result['Duration'] / total * 100
    result.rename(columns={'Name': 'Kernel Name', 'Duration':'Importance'}, inplace=True)

    return result


def join(df_time, df, bench_name, scale=False):
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
            #exit(1)
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
    if diff['Kernel Name'].size != 0:
        print(diff['Kernel Name'])

    # xticks with importance
    xticks = ['{}\n{:.1f}%'.format(i, result.iloc[i]['Importance']) for i in range(result.shape[0])]

    result = result.drop(columns='Kernel Name')

    if scale:
        result = result.multiply(result['Importance']/100, axis=0)

    result = result.sum(axis=0)

    result = pd.concat([pd.Series([bench_name], index=['Kernel Name']), result])

    # drop rows with importance below 1%
    #neglible = result[ result['Importance'] < 1 ].index
    #result.drop(neglible, inplace=True)

    # drop the importance column
    result = result.drop(labels='Importance')

    return result, xticks

def instmix(df_time, df, bench_name):
    #bench_name = args.name
    #outdir = args.outdir if args.outdir else args.indir

    # divide by 1 million
    df['Metric Value'] /= 1000000

    # process each benchmark individually
    k_metrics = df.groupby(['Kernel Name','Metric Name'], as_index=False)[['Metric Value']].sum()
    #k_metrics = k_metrics.reset_index(level=['Kernel Name','Metric Name'])

    df_map = {}
    df_map['Kernel Name'] = k_metrics['Kernel Name'].unique()

    metrics = ['FP16', 'FMA', 'FP64', 'INT', 
            'SPU', 'Tensor', 'Load/Store', 'EXEC' ]

    # I really hope no one sees this: re-organize the table
    for metric in metrics:
       metric_col = k_metrics.loc[k_metrics['Metric Name'] == metric]['Metric Value'].array
       df_map[metric] = metric_col

    trans_df = pd.DataFrame(df_map)

    # inner join the two tables: importance and metric values
    # also sort the table rows by descending importance
    df_join, xticks = join(df_time, trans_df, bench_name)

    # normalize each benchmark mixture and scale it by importance
    df_join[metrics] = df_join[metrics].div(df_join['EXEC'], axis=0)
    #.multiply(df_join['Importance'], axis=0)
                                       
    metrics.remove('EXEC')
    df_join['Other'] = df_join['EXEC'] - (df_join[metrics].sum())

    cols = ['Kernel Name', 'FP16', 'FMA', 'FP64', 'INT', 
            'SPU', 'Tensor', 'Load/Store', 'Other' ]
    df_join = df_join[cols]
    
    results = df_join.values

    # set graph size
    #plt.rcParams["figure.figsize"] = (30, 40)

    # print the list of kernels
    #print_kernel(results[:, 0])

    #draw_stack(results, legends, bench_name.capitalize(), 'Kernel ID', 
            #'Normalized Instruction Count Scaled by Runtime Weight',
            #xticks=xticks,
            #outfile=os.path.join(outdir, 'inst.pdf'))
    return results

def comp(df_time, df, bench_name):

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

    df_join, xticks = join(df_time, trans_df, bench_name, scale=True)

    #df_join = df_join[cols]

    results = df_join.values

    return results

    # set graph size
    #plt.rcParams["figure.figsize"] = (30, 20)

    '''
    draw_bar(results, legends, bench_name, 'Kernel ID', 
            'Utilization during Active Cycles (%)',
            xticks=xticks,
            outfile=os.path.join(outdir, 'comp.pdf'))
    '''

def mem(df_time, df, bench_name):
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

    df_join, xticks = join(df_time, trans_df, bench_name, scale=True)
    #df_join = df_join.drop(columns='Kernel Name')

    results = df_join.values

    return results 

    #plt.rcParams["figure.figsize"] = (35, 20)

    #draw_bar(results, cols, bench_name, 'Kernel ID', 
            #'Memory Utilization during Active Cycles (%)',
            #xticks=xticks,
            #outfile=os.path.join(outdir, 'mem.pdf'))

def main():
    # cmd parser
    parser = argparse.ArgumentParser('Parse nsight compute raw csv output for grouped microbenchmarks.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--indir', required=True, 
            help='Input directory of nsight compute dumps')

    # parse arguments
    args = parser.parse_args()

    dfs = {}
    # iterate over inst folder
    for filename in os.listdir(os.path.join(args.indir, 'inst')):
        full_filename = os.path.join(args.indir, 'inst', filename)
        bench_name = filename.split('.')[0]
        success, df = parse_csv(full_filename, 'inst')

        if success:
            dfs[bench_name] = {}
            dfs[bench_name]['inst'] = df

    def parse_all(subfolder):
        for filename in os.listdir(os.path.join(args.indir, subfolder)):
            full_filename = os.path.join(args.indir, subfolder, filename)
            bench_name = filename.split('.')[0]

            if bench_name not in dfs:
                continue

            success, df = parse_csv(full_filename, subfolder)

            if success:
                dfs[bench_name][subfolder] = df
            else:
                dfs.pop(bench_name, None)

    parse_all('time')
    parse_all('mem')
    parse_all('comp')

    inst_val = []
    mem_val = []
    comp_val = []
    # process data
    for bench in dfs:
        df_time = time(dfs[bench]['time'])

        result = instmix(df_time, dfs[bench]['inst'], bench)
        inst_val.append(result)

        result = mem(df_time, dfs[bench]['mem'], bench)
        mem_val.append(result)

        result = comp(df_time, dfs[bench]['comp'], bench)
        comp_val.append(result)


    ########### Instruction Mix ###########

    inst_val = np.array(inst_val)

    legends = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Ld/St', 'Other']
    # set graph size
    plt.rcParams["figure.figsize"] = (25, 20)
    draw_stack(inst_val, legends, 'Instruction Mix', 'Benchmark', 
            'Instruction Type Percentage', inst_val[:, 0], 
            xtick_rot=20, outfile=os.path.join(args.indir, 'inst.pdf'))

    ############ Memory Usage ##############
    mem_val = np.array(mem_val)

    legends = ['DRAM_UTIL_PCT', 'REGISTER_USAGE', 'SHARED_USAGE', 
            'OCCUPANCY', 'L1 Hit Rate']

    # set graph size
    plt.rcParams["figure.figsize"] = (15, 25)
    draw_bar(mem_val, legends, 'Memory Utilization', 'Benchmark', 
            'Memory Utilization during Active Cycles (%)', 
            xticks=mem_val[:, 0], xtick_rot=20, outfile=os.path.join(args.indir, 'mem.pdf'))

    ############### Compute Usage ############
    comp_val = np.array(comp_val)
    legends = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Ld/St']

    # set graph size
    plt.rcParams["figure.figsize"] = (15, 25)
    draw_bar(comp_val, legends, 'Compute Utilization', 'Benchmark', 
            'Compute Utilization during Active Cycles (%)',
            xticks=comp_val[:, 0], xtick_rot=20,
            outfile=os.path.join(args.indir, 'comp.pdf'))



if __name__ == '__main__':
    main()


