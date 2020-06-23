import numpy as np
import pandas as pd
import re
import ast
import sys

# bench -> C(0)/M(1)
bench_dict = {'cut_sgemm-0':0, 'cut_sgemm-1':0, 'cut_wmma-0': 0, 'cut_wmma-1': 0,
              'parb_stencil-0': 1, 'parb_sgemm-0': 0,
              'parb_lbm-0': 1, 'parb_spmv-0': 1, 'parb_cutcp-0': 0}

# each tuple contains: regex, dtype
regex_table = {'intra': r'INTRA_0:(.*):[0-9]+_CTA',
               'inter': r'INTER_0:(.*):[0-9]+_SM',
               'kidx': r'MIX_(.*)_KIDX',
               'l2': r'PARTITION_L2_0:(.*):[0-9|\.]+',
               '1_intra': r'INTRA_0:(.*):[0-9]+_CTA',
               '1_inter': r'INTER_0:(.*):[0-9]+_SM',
               '1_l2': r'PARTITION_L2_0:(.*):[0-9|\.]+',
               '1_kidx': r'MIX_0:(.*):[0-9]+_KIDX',
               '1_ctx': r'INTRA_0:(.*):.*_RATIO',
               '2_intra': r'INTRA_0:[0-9]+:(.*)_CTA',
               '2_inter': r'INTER_0:[0-9]+:(.*)_SM',
               '2_l2': r'PARTITION_L2_0:[0-9|\.]+:([0-9|\.]+)',
               '2_kidx': r'MIX_0:[0-9]+:(.*)_KIDX',
               '2_ctx': r'INTRA_0:.*:(.*)_RATIO',
               }

type_table = {'intra': int,
              'inter': int,
              'l2': float,
              'norm_ipc': float,
              'avg_dram_bw': float,
              'kidx': int,
              '1_kidx': int,
              '2_kidx': int,
              '1_ctx': float,
              '2_ctx': float,
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


# map a panda column from vector cell to scalar cell by taking average
def avg_array(s):
    result = [np.nanmean(np.array(v[1:-1].split(' ')).astype(float)) for v in s]
    return np.array(result)


def std_array(s):
    result = [np.std(np.array(v[1:-1].split(' ')).astype(float)) for v in s]
    return np.array(result)


def process_config_column(*configs, df):

    for c in configs:
        def parse_cfg(str):
            # split config string into trunks
            list_config = str.split('-')

            # match config name
            for substring in list_config:
                match = re.search(regex_table[c], substring)
                if match:
                    return match.group(1)

            return 0

        df[c] = df['config'].apply(parse_cfg).astype(type_table.get(c, float))


def normalize(df, index, metric, value, inverse):
    if inverse:
        return df.loc[index, metric] / value
    else:
        return value / df.loc[index, metric]


def parse_multi_array(text):
    text = text.replace(' ', ',')

    # In case there's any inf and ast literal_eval will fail
    text = text.replace('inf', '-1')
    arr = ast.literal_eval(text)

    return arr


def multi_array_col_seq(df):
    def singular(*metrics):
        for m in metrics:
            if df[m].dtype != np.int64:
                df[m] = df[m].transform(parse_multi_array)
                df[m] = df[m].transform(lambda x: x[1][0])

    singular('runtime', 'instructions', 'l2_bw', 'avg_mem_lat',
             'avg_core_to_l2', 'avg_l2_to_core', 'avg_mrq_latency')


def calculate_sld_short(shared_runtime, isolated_runtime):
    runtime = np.array(shared_runtime)
    sld = []

    # 1. Find out which stream has a shorter runtime (single iteration)
    sum_runtime = np.array([sum(arr_run) for arr_run in runtime])
    min_runtime = np.min(sum_runtime[sum_runtime > 0])

    # 2. For every stream, find the max number of full iterations
    # executed before the shortest stream ended
    for stream_id, stream in enumerate(runtime):
        sld_stream = 0
        stream = np.array(stream)

        if len(stream) > 0:
            num_kernels = len(isolated_runtime[stream_id])
            num_iters = int(len(stream) / num_kernels)
            tot_time = sum(stream[0:num_iters * num_kernels])

            while tot_time > min_runtime:
                num_iters -= 1
                tot_time = sum(stream[0:num_iters * num_kernels])

            if tot_time <= 0:
                print("calculate_sld_short: tot_time is invalid.")
                sys.exit(1)

            iter_time = sum(isolated_runtime[stream_id])
            sld_stream = num_iters * iter_time / tot_time

        sld.append(sld_stream)

    return sld
