import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import argparse
import cxxfilt
import re



def parse_csv(directory):
    results = {}

    for filename in os.listdir(directory):
        full_filename = os.path.join(directory, filename)

        # find out number of lines to skip in file
        n_skip = 0
        output_exists = False

        if not filename.endswith(".txt"):
            continue

        with open(full_filename) as search:
            for num, line in enumerate(search, 1):
                # Sketchy, might break
                if '\"Metric Value\"' in line:
                    n_skip = num - 1
                    output_exists = True
                    

        # exit if ==prof== is on the last line
        if not output_exists:
            print(filename, "went wrong!")
            continue

        print("Parsing", filename)

        # read data using pandas
        data = pd.read_csv(full_filename, delimiter=',', skiprows=n_skip, thousands=',')

        results[filename] = data

    # return a map of benchmark/filename to pandas dataframe
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

def draw_stack(results, legends, title, xlabel, ylabel, outfile='inst.pdf', pwidth=20):
    plt.rcParams["figure.figsize"] = (pwidth, 40)

    # draw stack bars
    idx = np.arange(results.shape[0])
    width = 0.35

    accum = np.zeros(results.shape[0], dtype=np.float32)

    p = []
    # colors = sns.color_palette("husl", len(cols)-1)
    colors = sns.color_palette("Set2", len(legends))

    for i, col in enumerate(results.T):
        if i == 0 or i == len(legends):
            continue
        height = np.divide(col, results[:,-1])
        p.append(plt.bar(idx, height, width, bottom=accum, color=colors[i]))
        accum += height.astype(np.float32)

    # add the other uncaptured types
    p.append(plt.bar(idx, 1-accum, width, bottom=accum))

    #plt.xticks(idx, results[:, 0], fontsize=16, rotation=30)
    plt.xticks(idx, np.arange(results.shape[0]), fontsize=16, rotation=30)
    plt.yticks(fontsize=16)
    
    plt.title(title, fontsize=22)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    plt.ylim([0, 1])
    # flip legends
    p.reverse()
    legends.reverse()

    plt.legend(p, legends, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
    print_kernel(results[:, 0])

    plt.savefig(outfile,  bbox_inches='tight')
    plt.close()

def instmix(args):
    # parse inputs
    df_read = parse_csv(args.indir)
    outdir = args.outdir if args.outdir else args.indir

    aggr_result = pd.DataFrame()

    for benchmark in df_read:
        df = df_read[benchmark]
        bench_name = benchmark.split('.')[0]

        # divide by 1 million
        df['Metric Value'] /= 1000000

        if args.aggregate:
            groups = df.groupby('Metric Name')[['Metric Value']].sum().T
            groups['Benchmark'] = bench_name
            #groups['Control'] = groups['ADU'] + groups['CBU']

            aggr_result = aggr_result.append(groups)
            print('unimplemented error')

        else:
            # process each benchmark individually
            k_metrics = df.groupby(['Kernel Name','Metric Name'], as_index=False)[['Metric Value']].sum()
            #k_metrics = k_metrics.reset_index(level=['Kernel Name','Metric Name'])

            df_map = {}
            df_map['Kernel Name'] = k_metrics['Kernel Name'].unique()

            metrics = ['FP16', 'FMA', 'FP64', 'INT', 
                    'SPU', 'Tensor', 'Load/Store', 'EXEC' ]

            print(k_metrics.loc[k_metrics['Kernel Name'] == 'wgrad_alg0_engine'])
            # I really hope no one sees this: re-organize the table
            for metric in metrics:
               metric_col = k_metrics.loc[k_metrics['Metric Name'] == metric]['Metric Value'].array
               df_map[metric] = metric_col

            trans_df = pd.DataFrame(df_map)
            cols = ['Kernel Name'] + metrics
            trans_df = trans_df[cols]
            print(trans_df.head())

            results = trans_df.values

            legends = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Ld/St', 'Other']
            draw_stack(results, legends, bench_name.capitalize(), 'Kernel ID', 
                    'Percentage of Total Executed Instructions',
                    os.path.join(outdir, bench_name+'-inst.pdf'), pwidth=25)
         
def draw_bar(results, legends, title, xlabel, ylabel, outfile='util.pdf', pwidth=20):
    plt.rcParams["figure.figsize"] = (pwidth, 20)

    num_metrics = results.shape[1]
    idx = np.arange(results.shape[0])

    p = []
    colors = sns.color_palette("husl", len(legends))
    #colors = sns.color_palette("Set2", len(legends))

    ax1 = None
    for i, col in enumerate(results.T):
        if i == 0:
            continue

        if i == 1:
            ax = ax1 = plt.subplot(len(legends), 1, i)
            plt.bar(idx, results[:, i], color=colors[i], label=legends[i-1])
            plt.title(title, fontsize=22)
        else:
            ax = plt.subplot(len(legends), 1, i, sharex=ax1, sharey=ax1)
            plt.bar(idx, results[:, i], color=colors[i], label=legends[i-1])
            

        #plt.setp(ax.get_xticklabels(), visible=False)
        plt.legend(fontsize=18)
        plt.ylim([0, 100])
        plt.yticks(fontsize=16)


        if i == num_metrics//2:
            plt.ylabel(ylabel, fontsize=18)
        if i == num_metrics - 1:
            plt.xlabel(xlabel, fontsize=18)

    plt.xticks(idx, idx, fontsize=14)
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
     

def util(args):
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

            metrics = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Load/Store']

            # I really hope no one sees this: re-organize the table
            for metric in metrics:
                metric_col = k_metrics.loc[k_metrics['Metric Name'] == metric]['Metric Value'].array
                df_map[metric] = metric_col

            trans_df = pd.DataFrame(df_map)
            cols = ['Kernel Name'] + metrics
            trans_df = trans_df[cols]

            results = trans_df.values

            legends = ['FP16', 'FMA', 'FP64', 'ALU', 'SPU', 'Tensor', 'Ld/St', 'Other']
            draw_bar(results, legends, bench_name, 'Kernel ID', 
                    'Utilization during Active Cycles (%)',
                    os.path.join(outdir, bench_name+'-util.pdf'), pwidth=25)
 
def mem(args):
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

            metrics = ['DRAM_UTIL_PCT', 'REGISTER_USAGE', 'DYNAMIC_SHARED_USAGE', 'STATIC_SHARED_USAGE', 'L2 Hit Rate', 'OCCUPANCY', 'L1 Hit Rate']

            # I really hope no one sees this: re-organize the table
            for metric in metrics:
                metric_col = k_metrics.loc[k_metrics['Metric Name'] == metric]['Metric Value'].array
                df_map[metric] = metric_col

            trans_df = pd.DataFrame(df_map)
            cols = ['Kernel Name'] + metrics
            trans_df = trans_df[cols]
            trans_df['REGISTER_USAGE'] = trans_df['REGISTER_USAGE'] * 1024 * trans_df['OCCUPANCY'] / (64 * 1024)

            results = trans_df.values

            legends = ['DRAM_UTIL_PCT', 'REGISTER_USAGE', 'DYNAMIC_SHARED_USAGE', 'STATIC_SHARED_USAGE', 
                    'L2 Hit Rate', 'OCCUPANCY', 'L1 Hit Rate', 'Other']
            draw_bar(results, legends, bench_name, 'Kernel ID', 
                    'Memory Utilization during Active Cycles (%)',
                    os.path.join(outdir, bench_name+'-util.pdf'), pwidth=25)
 

def main():

    def add_io_dir(par):
        par.add_argument('--indir', required=True, 
                help='Input directory of nsight compute dumps')
        par.add_argument('--outdir', required=False, 
                help='Output directory of graphs. Default is the same as input dir.')

    def add_aggregate_flag(par):
        par.add_argument('--aggregate', action='store_true', default=False,
                help='''Produce averaged results for all benchmarks in the input dir. 

                Otherwise, output verbose results for each benchmark.
                Default is false.''')

    # cmd parser
    parser = argparse.ArgumentParser('Parse nsight compute raw csv output.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    subparsers = parser.add_subparsers(title='Nsight compute process data types.',
            help='additional help')

    instmix_parser = subparsers.add_parser('instmix')
    add_io_dir(instmix_parser)
    add_aggregate_flag(instmix_parser)
    instmix_parser.set_defaults(func=instmix)

    util_parser = subparsers.add_parser('util')
    add_io_dir(util_parser)
    add_aggregate_flag(util_parser)
    util_parser.set_defaults(func=util)

    mem_parser = subparsers.add_parser('mem')
    add_io_dir(mem_parser)
    add_aggregate_flag(mem_parser)
    mem_parser.set_defaults(func=mem)

    # parse arguments
    args = parser.parse_args()

    # call function of each subparser
    args.func(args)









       

if __name__ == '__main__':
    main()


