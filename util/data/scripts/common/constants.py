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

base_config = 'TITANV-PAE-CONCURRENT-SEP_RW-LSRR'

kernel_yaml = yaml.load(
    open(os.path.join(DATA_HOME, 'scripts/common/', 'kernel.yml')),
    Loader=yaml.FullLoader)

# benchmark -> number of kernels
multi_kernel_app = OrderedDict([('parb_sad-0', 3),
                                ('parb_sad-1', 3),
                                ('parb_histo-0', 4),
                                ('parb_histo-1', 4),
                                ('parb_mriq-0', 3),
                                ('nvd_conv-0', 2),
                                ('rod_cfd-0', 6),
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
    if bench in multi_kernel_app.keys():
        for k in kernel_yaml[bench]:
            q = math.floor(get_max_cta_per_sm(bench, k) * ctx)
            q_grid = math.ceil(get_grid_size(bench, k) / num_sm_volta)
            k_quota.append(min(q, q_grid))
            assert (k == len(k_quota))
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
    if bench in multi_kernel_app.keys():
        return multi_kernel_app[bench]
    else:
        return 1
