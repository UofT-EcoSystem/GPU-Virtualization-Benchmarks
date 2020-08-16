import matplotlib as mpl
import matplotlib.pyplot as plt
from random import randint
import argparse
import pandas as pd
import numpy as np

plt.rcParams["figure.figsize"] = [10, 6]

# parse input params
def parse_args():
    parser = argparse.ArgumentParser("Graph sensetivity results for GPUPool.")

    parser.add_argument('--data_file', type=str, required=True, 
                        help='File from which to extract data.')
    parser.add_argument('--type', required=True,
                        choices=['ws', 'gpus', 'viol'],
                        help='Type of graph that we want to produce.\
                                One of: ws, gpus, violations')

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data_file)
    fig, ax = plt.subplots()

    x = df['num_jobs']
    width = 15  # the width of the bars
    ax.set_xlabel('Number of Jobs')
    ax.set_xticks(x)

    if args.type == 'ws':
        # plot the weighted speedup vs number of jobs
        pool = df['gpupool_ws']
        mig = df['mig_ws']
        dram = df['dram_ws']
        rects1 = ax.bar(x - width, pool, width, label='GPUPool')
        rects2 = ax.bar(x, mig, width, label='MIG')
        rects3 = ax.bar(x + width, dram, width, label='DRAM_based')
        ax.set_ylabel('Weighted Speedup')
        ax.set_title('Weighted Speedup: GPUPool vs MIG vs DRAM_Based')
        plt.ylim([0, max(pool.max(), mig.max(), dram.max()) + 0.5])
        ax.legend(loc=2)
        plt.savefig('/home/golikovp/graph_sens_num_ws.pdf')

    elif args.type == 'gpus':
        # plot the number of gpus on y-axis
        pool = df['gpupool_required_gpus']
        mig = df['mig_required_gpus']
        rects1 = ax.bar(x - width/2, pool, width, label='GPUPool')
        rects2 = ax.bar(x + width/2, mig, width, label='MIG')
        ax.set_ylabel('Number of GPUs')
        ax.set_title('Number of GPUs: GPUPool vs MIG')
        ax.legend(loc=2)
        plt.savefig('/home/golikovp/graph_sens_num_gpus.pdf')

    elif args.type == 'viol':
        pool = df['gpupool_violations']
        random = df['random_violations']
        dram = df['dram_bw_based_violations']
        rects1 = ax.bar(x - width, pool, width, label='GPUPool')
        rects2 = ax.bar(x, random, width, label='Random')
        rects3 = ax.bar(x + width, dram, width, label='DRAM_based')
        ax.set_ylabel('Number of Violations')
        ax.set_title('Number of Violations: GPUPool vs Random vs DRAM_Based')
        plt.ylim([0, max(pool.max(), random.max(), dram.max()) + 0.5])
        ax.legend(loc=2)
        plt.savefig('/home/golikovp/graph_sens_num_viol.pdf')





if __name__ == '__main__':
    main()





