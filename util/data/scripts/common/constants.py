import os
import oyaml as yaml
from collections import OrderedDict

DATA_HOME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

max_cta_volta = 32
max_thread_volta = 2048
max_smem = 96 * 1024
max_register = 64 * 1024
num_sm_volta = 80

kernel_yaml = yaml.load(
    open(os.path.join(DATA_HOME, 'scripts/common/', 'kernel.yml')),
    Loader=yaml.FullLoader)


def get_kernel_stat(kernel, stat, kidx):
    sp_kernel = kernel.split(':')
    if len(sp_kernel) > 1:
        bench = sp_kernel[0]
        kidx = int(sp_kernel[1])
        return kernel_yaml[bench][kidx][stat]
    elif kidx != 0:
        return kernel_yaml[kernel][kidx][stat]
    else:
        return kernel_yaml[kernel][stat]


def get_max_cta_per_sm(kernel, kidx=0):
    return get_kernel_stat(kernel, 'max_cta', kidx)


def get_grid_size(kernel, kidx=0):
    return get_kernel_stat(kernel, 'grid', kidx)


def get_block_size(kernel, kidx=0):
    return get_kernel_stat(kernel, 'block', kidx)


def get_regs(kernel, kidx):
    return get_kernel_stat(kernel, 'regs', kidx)


def get_smem(kernel, kidx):
    return get_kernel_stat(kernel, 'smem', kidx)


# benchmark -> number of unique kernels
multi_kernel_app = OrderedDict([('parb_sad-0', 3),
                                ('parb_sad-1', 3),
                                ('parb_histo-0', 4),
                                ('parb_histo-1', 4),
                                ('nvd_conv-0', 2),
                                ])
