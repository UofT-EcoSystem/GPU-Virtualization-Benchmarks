import os
import oyaml as yaml
import math
from collections import OrderedDict
import numpy as np

DATA_HOME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

max_cta_volta = 32
max_thread_volta = 2048
max_smem = 96 * 1024
max_register = 64 * 1024
num_sm_volta = 80
num_mem_channels_volta = 24

base_config = 'TITANV-PAE-CONCURRENT-SEP_RW-LSRR'

kernel_yaml = yaml.load(
    open(os.path.join(DATA_HOME, 'scripts/common/', 'kernel.yml')),
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


def get_kernel_stat(kernel, stat, kidx):
    sp_kernel = kernel.split(':')
    kidx = int(kidx)
    if len(sp_kernel) > 1:
        bench = sp_kernel[0]
        kidx = int(sp_kernel[1])
        return kernel_yaml[bench][kidx][stat]
    elif kidx != 0:
        return kernel_yaml[kernel][kidx][stat]
    else:
        return kernel_yaml[kernel][stat]


def get_max_cta_per_sm(bench, kidx=0):
    return get_kernel_stat(bench, 'max_cta', kidx)


def calc_cta_quota(bench, ctx):
    k_quota = []
    if bench in multi_kernel_app:
        for k in multi_kernel_app[bench]:
            q = math.floor(get_max_cta_per_sm(bench, k) * ctx)
            q_grid = math.ceil(get_grid_size(bench, k) / num_sm_volta)
            k_quota.append(min(q, q_grid))

    return k_quota


def get_grid_size(bench, kidx=0):
    return get_kernel_stat(bench, 'grid', kidx)


def get_achieved_cta(kernel):
    split_kernel = kernel.split(':')
    bench = split_kernel[0]
    kidx = split_kernel[1] if len(split_kernel) > 1 else 0

    result = np.minimum(np.ceil(get_grid_size(bench, kidx) / num_sm_volta),
                        get_max_cta_per_sm(bench, kidx)).astype('int32')

    return result


def get_block_size(bench, kidx=0, padded=True):
    block_size = get_kernel_stat(bench, 'block', kidx)
    if padded:
        block_size = math.ceil(block_size / 32) * 32

    return block_size


def get_block_ratio(bench, kidx=0):
    block_size = get_block_size(bench, kidx)

    return block_size / max_thread_volta


def get_regs(bench, kidx):
    return get_kernel_stat(bench, 'regs', kidx)


def get_smem(bench, kidx):
    return get_kernel_stat(bench, 'smem', kidx)


def get_num_kernels(bench):
    if bench in multi_kernel_app:
        return len(multi_kernel_app[bench])
    else:
        return 1


def get_primary_kidx(bench, idx):
    return multi_kernel_app[bench][idx]


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
