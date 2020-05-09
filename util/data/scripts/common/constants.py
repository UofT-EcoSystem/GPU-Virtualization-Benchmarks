import os
from collections import OrderedDict

DATA_HOME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

max_cta_volta = 32
max_thread_volta = 2048
max_smem = 96 * 1024
max_register = 64 * 1024
num_sm_volta = 80

# elements in value:
# 1. max ctas according to resource constraints,
# 2. grid size,
# 3. is compute intensive?
kernel_dict = OrderedDict([
    # ('cut_sgemm-0', [2, 128, 1]),
    ('cut_sgemm-1', [2, 512, 1]),
    ('cut_wmma-0', [4, 128, 1]),
    # ('cut_wmma-1', [4, 1024, 1]),
    # ('parb_sgemm-0', [11, 528, 1]),
    ('parb_cutcp-0', [16, 121, 0]),
    ('parb_stencil-0', [16, 1024, 0]),
    ('parb_lbm-0', [12, 18000, 0]),
    ('parb_spmv-0', [16, 1147, 0]),
    # ('rod_mummer-0', [8, 5727]),
    ('rod_heartwall-0', [5, 51, 0]),
    ('rod_hotspot-0', [8, 1849, 1]),
    ('rod_hotspot3d-0', [8, 1024, 0]),
    ('rod_streamcluster-0', [3, 128, 0]),
    ('rod_pathfinder-0', [8, 463, 1]),
    ('rod_lavamd-0', [9, 1000, 1]),
    ('nvd_binomial-0', [16, 1024, 1]),
    ('nvd_blackscholes-0', [16, 15625, 0]),
    ('nvd_fdtd3d-0', [1, 288, 0]),
    ('nvd_interval-0', [12, 1024, 1]),
    ('nvd_sobol-0', [32, 51200, 1]),

    ('nvd_conv-0:1', [32, 18432, None]),
    ('nvd_conv-0:2', [16, 18432, None]),

    ('parb_sad-0:1', [17, 1584, None]),
    ('parb_sad-0:2', [16, 99, None]),
    ('parb_sad-0:3', [32, 99, None]),

    ('parb_sad-1:1', [17, 128640, None]),
    ('parb_sad-1:2', [16, 8040, None]),
    ('parb_sad-1:3', [32, 8040, None]),
])


def get_max_cta_per_sm(kernel):
    return kernel_dict[kernel][0]


def get_grid_size(kernel):
    return kernel_dict[kernel][1]


pair_ignore = ['cut_sgemm-0', 'cut_wmma-1', 'parb_sgemm-0', 'rod_mummer-0']
app_for_pair = [app for app in kernel_dict if app not in pair_ignore]

multi_kernel_app = OrderedDict([('parb_sad-0', 3),
                                ('parb_sad-1', 3),
                                ('parb_histo-0', 4),
                                ('parb_histo-1', 4),
                                ])
