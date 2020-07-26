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
               'bypass_l2': r'BYPASS_L2D_S1',
               'lut': r'INTRA_LUT',
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
               'cap': r'CAP_(.*)_CYCLE',
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
              'cap': int,
              }

metric_label = {'intra': 'Concurrent TBs/SM',
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


def process_config_column(*configs, df, default=0):
    for c in configs:
        df[c] = df['config'].apply(lambda x: check_config(c, x, default))


def check_config(config, config_str, default=0):
    # split config string into trunks
    list_config = config_str.split('-')

    # match config name
    for substring in list_config:
        match = re.search(regex_table[config], substring)
        if match:
            if len(match.groups()) == 1:
                convert = type_table.get(config, float)
                return convert(match.group(1))
            else:
                return True

    return default


def normalize(df, index, metric, value, inverse):
    if inverse:
        return df.loc[index, metric] / value
    else:
        return value / df.loc[index, metric]


def parse_multi_array(text):
    text = text.replace(' ', ',')

    # In case there's any inf and ast literal_eval will fail
    text = text.replace('inf', '-1')
    text = text.replace('-nan', '0')
    text = text.replace('nan', '0')

    try:
        arr = ast.literal_eval(text)
    except:
        print("parse_multi_array error: ", text)
        sys.exit()

    return arr


def multi_array_col_seq(df):
    def singular(*metrics):
        for m in metrics:
            if df[m].dtype != np.int64:
                df[m] = df[m].transform(parse_multi_array)
                df[m] = df[m].transform(lambda x: x[1][0])

    singular('runtime', 'instructions', 'l2_bw', 'avg_mem_lat',
             'avg_core_to_l2', 'avg_l2_to_core', 'avg_mrq_latency',
             'barrier_cycles', 'inst_empty_cycles', 'branch_cycles',
             'scoreboard_cycles', 'stall_sp_cycles', 'stall_dp_cycles',
             'stall_int_cycles', 'stall_tensor_cycles', 'stall_sfu_cycles',
             'stall_mem_cycles', 'not_selected_cycles', 'cycles_per_issue',
             )


def calculate_sld_short(shared_end_stamp, isolated_runtime, start=None):
    shared_end_stamp = [np.array(arr_end) for arr_end in shared_end_stamp]
    sld = []

    # 1. Find out which stream has a shorter runtime (single iteration)
    sum_runtime = np.array([arr_end[-1] for arr_end in shared_end_stamp])
    min_runtime = np.min(sum_runtime[sum_runtime > 0])

    # 2. For every stream, find the max number of full iterations
    # executed before the shortest stream ended
    for stream_id, stream in enumerate(shared_end_stamp):
        sld_stream = 0
        stream = np.array(stream)

        if len(stream) > 0:
            num_kernels = len(isolated_runtime[stream_id])
            num_iters = int(len(stream) / num_kernels)
            tot_time = stream[num_iters * num_kernels - 1]

            while tot_time > min_runtime:
                num_iters -= 1
                assert(num_iters > 0)
                tot_time = stream[num_iters * num_kernels - 1]

            if tot_time <= 0:
                print("calculate_sld_short: tot_time is invalid.")
                sys.exit(1)

            iter_time = sum(isolated_runtime[stream_id])

            if start:
                relative_start = start[stream_id]

            sld_stream = num_iters * iter_time / (tot_time - relative_start)

        sld.append(sld_stream)

    return sld
