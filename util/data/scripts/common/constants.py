import os
import oyaml as yaml
import math
from collections import OrderedDict
import numpy as np
import sys
import pandas as pd

DATA_HOME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

max_cta_volta = 32
max_thread_volta = 2048
max_smem = 96 * 1024
max_register = 64 * 1024
num_sm_volta = 80
num_mem_channels_volta = 32

base_config = "V100-PAE-CONCURRENT-SEP_RW"
pair_base_config = base_config + "-LSRR" + "-PRINT_DEVICE_SYNC"

l2d_bypass_threshold = 0.95

# Specify kernel launch latency in cycles
# This is useful to enforce which stream gets to launch its kernel first in
# CUDA stream simulation
LAUNCH_LATENCY = 10

kernel_yaml = yaml.load(
    open(os.path.join(DATA_HOME, 'scripts/common/', 'kernel.yml')),
    Loader=yaml.FullLoader)
syn_yaml = yaml.load(
    open(os.path.join(DATA_HOME, 'scripts/common/', 'synthetic.yml')),
    Loader=yaml.FullLoader)

# benchmark -> kernel sequence in one iteration
multi_kernel_app = OrderedDict(
    [('parb_sad-0', [1, 2, 3]),
     ('parb_sad-1', [1, 2, 3]),
     # ('parb_histo-0', [1, 2, 3, 4]),
     # ('parb_histo-1', [1, 2, 3, 4]),
     ('parb_mriq-0', [1, 2, 2]),
     ('nvd_conv-0', [1, 2]),
     ('rod_cfd-0', [1, 1, 1, 4, 5, 6, 5, 6, 5, 6]),
     ('rod_cfd-1', [1, 1, 1, 4, 5, 6, 5, 6, 5, 6]),
     ])


def get_single_kernel_app():
    return [app for app in kernel_yaml if app not in multi_kernel_app]


def get_kernel_stat(kernel, stat, kidx):
    sp_kernel = kernel.split(':')
    kidx = int(kidx)
    if len(sp_kernel) > 1:
        bench = sp_kernel[0]
        kidx = int(sp_kernel[1])
        return kernel_yaml[bench][kidx][stat]
    elif kernel in multi_kernel_app:
        return kernel_yaml[kernel][kidx][stat]
    else:
        # Ignore kidx if this is a single-kernel benchmark
        return kernel_yaml[kernel][stat]


def get_max_cta_per_sm(bench, kidx=1):
    return get_kernel_stat(bench, 'max_cta', kidx)


def calc_cta_quota(bench, ctx):
    k_quota = []
    if bench in multi_kernel_app:
        for k in multi_kernel_app[bench]:
            q = math.floor(get_max_cta_per_sm(bench, k) * ctx)
            q_grid = math.ceil(get_grid_size(bench, k) / num_sm_volta)
            k_quota.append(min(q, q_grid))

    return k_quota


def get_grid_size(bench, kidx=1):
    return get_kernel_stat(bench, 'grid', kidx)


def get_achieved_cta(kernel):
    split_kernel = kernel.split(':')
    bench = split_kernel[0]
    kidx = split_kernel[1] if len(split_kernel) > 1 else 0

    result = np.minimum(np.ceil(get_grid_size(bench, kidx) / num_sm_volta),
                        get_max_cta_per_sm(bench, kidx)).astype('int32')

    return result


def get_block_size(bench, kidx=1, padded=True):
    block_size = get_kernel_stat(bench, 'block', kidx)
    if padded:
        block_size = math.ceil(block_size / 32) * 32

    return block_size


def get_block_ratio(bench, kidx=1):
    block_size = get_block_size(bench, kidx)

    return block_size / max_thread_volta


def get_regs(bench, kidx):
    return get_kernel_stat(bench, 'regs', kidx)


def get_smem(bench, kidx):
    return get_kernel_stat(bench, 'smem', kidx)


def get_num_kernels(app):
    def get_num_kernels_bench(bench):
        if bench in multi_kernel_app:
            return len(multi_kernel_app[bench])
        else:
            return 1

    if app in syn_yaml:
        list_bench = syn_yaml[app]

        # Replace "repeat" benchmark with previous benchmark in list
        for idx, bench in enumerate(list_bench):
            if bench == 'repeat':
                list_bench[idx] = list_bench[idx - 1]

        result = sum([get_num_kernels_bench(bench) for bench in list_bench])
        return result
    else:
        return get_num_kernels_bench(app)


def get_primary_kidx(bench, idx):
    return multi_kernel_app[bench][idx]


def get_pickle(pickle_name):
    path_pickle = os.path.join(DATA_HOME, 'pickles', pickle_name)

    if not os.path.exists(path_pickle):
        print("Cannot find {0}.".format(pickle_name))
        sys.exit(3)

    df = pd.read_pickle(path_pickle)
    return df


# Return a list of isolated runtime in app
def get_seq_cycles(app):
    # Cap cycles are calculated based on seq run
    df_seq_multi = get_pickle('seq-multi.pkl')
    df_seq_multi.set_index(['pair_str', '1_kidx'], inplace=True)

    df_seq = get_pickle('seq.pkl')
    df_seq.set_index(['pair_str'], inplace=True)

    def get_seq_cycles_bench(bench):
        if bench in multi_kernel_app:
            cycles = [df_seq_multi.loc[(bench, kidx)]['runtime']
                      for kidx in multi_kernel_app[bench]]
        else:
            cycles = [df_seq.loc[bench]['runtime']]

        return cycles

    result = []
    if app in syn_yaml:
        # Synthetic workloads
        list_bench = syn_yaml[app]
        for idx, benchmark in enumerate(list_bench):
            if benchmark == 'repeat':
                result += get_seq_cycles_bench(list_bench[idx - 1])
            else:
                result += get_seq_cycles_bench(benchmark)
    else:
        result += get_seq_cycles_bench(app)

    return result


def get_dominant_usage(app):
    df_seq = get_pickle('seq.pkl')
    df_seq.set_index(['pair_str'], inplace=True)

    def get_seq_dominant_bench(bench, df_seq):
        if bench in multi_kernel_app:
            print("Unimplemented error: get_dominant_usage in constants.py")
            sys.exit(1)

        block_size = get_block_size(bench)
        block_regs = ((df_seq.loc[bench, 'regs'] + 3) & ~3) * block_size
        block_smem = df_seq.loc[bench, 'smem']

        usage = [("thread_ratio", block_size / max_thread_volta),
                 ("regs_ratio", block_regs / max_register),
                 ("smem_ratio", block_smem / max_smem),
                 ("thread_block_ratio", 1 / max_cta_volta)]

        return max(usage, key=lambda x: x[1])

    if app in syn_yaml:
        # Synthetic workloads
        list_bench = syn_yaml[app]
        list_usage = [get_seq_dominant_bench(bench, df_seq) for bench in
                      list_bench]

        return list_usage
    else:
        return [get_seq_dominant_bench(app, df_seq)]


def get_cta_setting_from_ctx(rsrc_usage, ctx):
    list_quota = [math.floor(ctx / usage[1]) for usage in rsrc_usage]

    return list_quota


# For kernels that simply repeat the primary kernel, return the kidx key of
# the primary kernel
# Input kidx starts at 0 (GPUSim output)
def translate_gpusim_kidx(bench, kidx):
    num_kernels = get_num_kernels(bench)
    idx = kidx % num_kernels

    return get_primary_kidx(bench, idx)


def gen_kernel_headers(app):
    result = ["{}:{}".format(app, idx)
              for idx in range(get_num_kernels(app))]

    return result


# Assume kernels are launched back to back
def get_from_to(duration):
    time_series = [0]

    for d in duration:
        new_elem = time_series[-1] + d
        time_series.append(new_elem)

    time_series = np.array(time_series)
    return time_series[0:-1], time_series[1:]

