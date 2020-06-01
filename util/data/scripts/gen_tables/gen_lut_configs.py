import numpy as np
import pandas as pd
import math

import data.scripts.common.constants as const
import scheduler.scheduler as scheduler


def get_lut_matrix(apps, df_dynamic, weighted=True):
    if weighted:
        # Want to minimize weighted runtime increase (sum_increase)
        df_best = df_dynamic.sort_values('sum_increase', ascending=True) \
            .drop_duplicates(['1_bench', '1_kidx', '2_bench', '2_kidx']) \
            .set_index(['1_bench', '1_kidx', '2_bench', '2_kidx'])
    else:
        df_best = df_dynamic.sort_values('ws', ascending=False) \
            .drop_duplicates(['1_bench', '1_kidx', '2_bench', '2_kidx']) \
            .set_index(['1_bench', '1_kidx', '2_bench', '2_kidx'])

    num_cols = const.get_num_kernels(apps[0])
    num_rows = const.get_num_kernels(apps[1])
    matrix_size = (num_rows, num_cols)

    configs = [np.zeros(matrix_size, dtype=int),
               np.zeros(matrix_size, dtype=int)]
    interference = [np.zeros(matrix_size),
                    np.zeros(matrix_size)]

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            kidx = [const.translate_gpusim_kidx(apps[0], col_idx),
                    const.translate_gpusim_kidx(apps[1], row_idx)]
            best_idx = (apps[0], kidx[0], apps[1], kidx[1])

            cta_1 = df_best.loc[best_idx, '1_intra']
            sld_1 = df_best.loc[best_idx, 'sld'][1]

            cta_2 = df_best.loc[best_idx, '2_intra']
            sld_2 = df_best.loc[best_idx, 'sld'][2]

            matrix_idx = (row_idx, col_idx)

            configs[0][matrix_idx] = int(cta_1)
            configs[1][matrix_idx] = int(cta_2)

            interference[0][matrix_idx] = sld_1
            interference[1][matrix_idx] = sld_2

    return configs, interference


def pretty_print_matrix(apps, title, data, headers):
    for app, app_data in zip(apps, data):
        print(app, title)
        print(pd.DataFrame(app_data, index=headers[1], columns=headers[0]))
        print('-' * 50)


def print_rsrc_usage(configs, headers, df_intra_index):
    print('Execution context utilization:')
    rsrc = ['cta_ratio', 'thread_ratio', 'smem_ratio', 'reg_ratio',
            'avg_dram_bw']

    def get_usage(intra_list, kernel_list, app_idx):
        usage_app = []
        for intra, kernel in zip(intra_list, kernel_list):
            split_kernel = kernel.split(':')
            app = split_kernel[0]
            kidx = const.get_primary_kidx(app, int(split_kernel[1]))
            intra_idx = (app, kidx, intra)

            usage = [df_intra_index.loc[intra_idx, r] for r in rsrc]
            usage_app.append(usage)

        usage_app = np.array(usage_app)
        print('>>> App', app_idx)
        max_usage = np.amax(usage_app, axis=0)
        for u, r in zip(max_usage, rsrc):
            print(r, "{:.3}".format(u))

    # App 1, take max of column in first table
    max_intra_1 = configs[0].max(axis=0)
    get_usage(max_intra_1, headers[0], 1)

    # App 2: take max of each row in second table
    max_intra_2 = configs[1].max(axis=1)
    get_usage(max_intra_2, headers[1], 2)


def predict_app(apps, interference, df_seq_index):
    # Get seq runtimes
    def get_runtime(app):
        result = [df_seq_index.loc[(app, const.translate_gpusim_kidx(app, kidx))]
                  ['runtime'] for kidx in range(const.get_num_kernels(app))]

        return result

    runtimes = [get_runtime(app) for app in apps]

    tot_runtime = [sum(r) for r in runtimes]
    if tot_runtime[0] < tot_runtime[1]:
        iter_lim = [math.inf, 1]
    else:
        iter_lim = [1, math.inf]

    sim_results = scheduler.simulate(runtimes, interference, iter_lim,
                                     finish_remaining=False)
    scaled_runtime = [[int(t) for t in app] for app in sim_results[0]]
    #     print('Predicted runtime:', scaled_runtime)

    norm_ipc = scheduler.calculate_norm_ipc(runtimes, scaled_runtime)
    #     print('Norm IPC:', norm_ipc)

    sld = scheduler.calculate_qos(runtimes, scaled_runtime, short=True,
                                  revert=False)
    print('Predicted weighted speedup:', sum(sld))

    return scaled_runtime, norm_ipc
