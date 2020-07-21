import numpy as np
import pandas as pd
import math
import sys

import data.scripts.common.constants as const
import scheduler.scheduler as scheduler


# Assume we have the configuration matrix and we want to get the interference
# matrix from dynamic or intra. Use interpolation to support "fuzzy logic" where
# no exact dynamic pair exists.
def get_interference_matrix(apps, configs, df_dynamic, df_intra):
    benchmarks = []
    for app in apps:
        if app in const.syn_yaml:
            benchmarks.append([(bench, 1) for bench in const.syn_yaml[app]])
        elif app in const.multi_kernel_app:
            benchmarks.append([(app, kidx)
                               for kidx in const.multi_kernel_app[app]])
        else:
            benchmarks.append([(app, 1)])

    num_cols = const.get_num_kernels(apps[0])
    num_rows = const.get_num_kernels(apps[1])
    matrix_size = (num_rows, num_cols)

    interference = [np.zeros(matrix_size),
                    np.zeros(matrix_size)]

    kernel_columns = ['1_bench', '1_kidx', '2_bench', '2_kidx']

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            matrix_idx = (row_idx, col_idx)

            bench = [benchmarks[0][col_idx], benchmarks[1][row_idx]]
            kernel_configs = [c[matrix_idx] for c in configs]

            if bench[0] == bench[1]:
                # Look at df_intra
                df_bench = df_intra[(df_intra['pair_str'] == bench[0][0]) &
                                    (df_intra['1_kidx'] == bench[0][1])]
                df_bench = df_bench.copy()
                df_bench.sort_values('intra', inplace=True)

                intra_total = sum(kernel_configs)
                df_config = pd.DataFrame([{'intra': intra_total}])

                df_merge = pd.merge_asof(df_config, df_bench,
                                         on='intra',
                                         direction='nearest')

                weight = [intra / intra_total for intra in kernel_configs]
                list_sld = [df_merge['norm_ipc'].iloc[0] * w for w in weight]
            else:
                # Look at dynamic
                sorted_bench = bench.copy()
                sorted_bench.sort(key=lambda x: x[0])

                if sorted_bench != bench:
                    kernel_configs.reverse()

                value = sorted_bench[0] + sorted_bench[1]
                df_bench = df_dynamic[df_dynamic[kernel_columns].isin(
                    value).all(axis=1)].copy()

                if df_bench.empty:
                    # TODO: Need to handle kernel serialization...
                    print("Unimplemented error.")
                    sys.exit(1)

                df_bench['distance'] = np.abs(df_bench['1_intra'] -
                                              kernel_configs[0]) + \
                                       np.abs(df_bench['2_intra'] -
                                              kernel_configs[1])

                df_bench.sort_values('distance', inplace=True, ascending=True)

                list_sld = df_bench['sld'].iloc[0][1:]

                # Flip it back
                if sorted_bench != bench:
                    list_sld.reverse()

            interference[0][matrix_idx] = list_sld[0]
            interference[1][matrix_idx] = list_sld[1]

    return interference


def get_config_matrix_from_ctx(apps, ctx_1):
    ctx = [ctx_1, 1 - ctx_1]
    quota = [const.get_cta_from_ctx(const.get_dominant_usage(app), app_ctx, app)
             for app, app_ctx in zip(apps, ctx)]

    # Build config matrix:
    matrix_size = (const.get_num_kernels(apps[1]),
                   const.get_num_kernels(apps[0]))
    configs = [np.zeros(matrix_size), np.zeros(matrix_size)]

    # config for app 0
    for idx, kernel_quota in enumerate(quota[0]):
        configs[0][:, idx] = kernel_quota

    # config for app 1
    for idx, kernel_quota in enumerate(quota[1]):
        configs[1][idx, :] = kernel_quota

    configs = [c.astype(int) for c in configs]

    return quota, configs


def get_lut_matrix(apps, df_dynamic, df_intra):
    num_cols = const.get_num_kernels(apps[0])
    num_rows = const.get_num_kernels(apps[1])
    matrix_size = (num_rows, num_cols)

    # Calculate total runtime
    seq_cycles = [const.get_seq_cycles(app) for app in apps]
    importance = [[cycles / sum(app_cycles) for cycles in app_cycles]
                  for app_cycles in seq_cycles]

    if num_cols != len(importance[0]) or num_rows != len(importance[1]):
        print('Dimension mismatch between matrix size and importance array')
        sys.exit(1)

    # return list_cta, list_sld, serial?
    def get_cta_sld(bench_idx):
        real_bench = []
        serial = False

        for app, midx in zip(apps, bench_idx):
            if 'syn' in app:
                # FIXME: Here we assume we don't involve multi-kernel bench
                real_bench.append((const.syn_yaml[app][midx], 1))
            else:
                real_bench.append((app, const.translate_gpusim_kidx(app, midx)))

        if real_bench[0] == real_bench[1]:
            # Identical benchmark pair. Get settings from df_intra
            df_bench = df_intra[(df_intra['pair_str'] == real_bench[0][0]) &
                                (df_intra['1_kidx'] == real_bench[0][1])].copy()
            idx_max = df_bench[['intra', 'norm_ipc']].idxmax(axis=0)
            if idx_max['intra'] == idx_max['norm_ipc']:
                # Max setting is the max allowed setting
                max_cta = const.get_max_cta_per_sm(real_bench[0][0],
                                                   real_bench[0][1])
                if df_bench['intra'].max() * 2 > max_cta:
                    cta_setting = max_cta // 2

                    if cta_setting == 0:
                        cta_setting = 1
                        serial = True
                        sld = 1
                    else:
                        # Might be a bit pessimistic here:
                        sld = 0.5
                else:
                    cta_setting = df_bench.loc[idx_max['intra']]['intra']
                    sld = df_bench.loc[idx_max['intra']]['norm_ipc']
            else:
                cta_setting = df_bench.loc[idx_max['norm_ipc']]['intra'] // 2
                sld = 0.5

            return [cta_setting, cta_setting], [sld, sld], serial

        else:
            # Different benchmarks. Get settings from df_dynamic
            # Benchmarks within each pair in dynamic are sorted
            sorted_real_bench = real_bench.copy()
            sorted_real_bench.sort(key=lambda x: x[0])
            bench_importance = [importance[0][bench_idx[0]],
                                importance[1][bench_idx[1]]]

            if real_bench != sorted_real_bench:
                bench_importance.reverse()

            idx_df = (df_dynamic['1_bench'] == sorted_real_bench[0][0]) & \
                     (df_dynamic['1_kidx'] == sorted_real_bench[0][1]) & \
                     (df_dynamic['2_bench'] == sorted_real_bench[1][0]) & \
                     (df_dynamic['2_kidx'] == sorted_real_bench[1][1])

            df_pair = df_dynamic[idx_df].copy()

            if len(df_pair.index) == 0:
                # If no feasible pair dynamic config, let the kernels run
                # serially using its best intra config
                cta_setting = []
                sld = []
                serial = True
                for bench in real_bench:
                    best_idx = df_intra[(df_intra['pair_str'] == bench[0]) &
                                        (df_intra['1_kidx'] == bench[1])
                                        ]['norm_ipc'].idxmax(axis=0)
                    cta_setting.append(df_intra.loc[best_idx]['intra'])
                    sld.append(df_intra.loc[best_idx]['norm_ipc'])
            else:
                # df_pair['sum_increase'] = df_pair['sld'].apply(
                #     lambda list_sld: bench_importance[0] / list_sld[1] +
                #                      bench_importance[1] / list_sld[2]
                # )

                df_pair['sum_increase'] = df_pair['sld'].apply(
                    lambda list_sld: 1 / list_sld[1] +
                                     1 / list_sld[2]
                )

                df_pair.sort_values('sum_increase', inplace=True,
                                    ascending=True)

                series_best = df_pair.iloc[0]
                cta_setting = [series_best['1_intra'], series_best['2_intra']]
                sld = series_best['sld'][1:3]

                if real_bench != sorted_real_bench:
                    cta_setting.reverse()
                    sld.reverse()

            return cta_setting, sld, serial

    configs = [np.zeros(matrix_size, dtype=int),
               np.zeros(matrix_size, dtype=int)]
    interference = [np.zeros(matrix_size),
                    np.zeros(matrix_size)]
    serial_matrix = np.zeros(matrix_size, dtype=int)

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            list_cta, list_sld, pair_serial = get_cta_sld([col_idx, row_idx])

            if len(list_cta) == 0:
                print("LUT config does not exist for {}", apps)
                sys.exit(1)

            matrix_idx = (row_idx, col_idx)

            configs[0][matrix_idx] = int(list_cta[0])
            configs[1][matrix_idx] = int(list_cta[1])

            interference[0][matrix_idx] = list_sld[0]
            interference[1][matrix_idx] = list_sld[1]

            serial_matrix[matrix_idx] = int(pair_serial)

    return configs, interference, serial_matrix


# def get_ctx_matrix(apps, row, df_dynamic_index):
#     # Show predicted timeline based on df_dynamic info
#     matrix_size = (const.get_num_kernels(apps[1]),
#                    const.get_num_kernels(apps[0]))
#     interference_1 = np.zeros(matrix_size)
#     interference_2 = np.zeros(matrix_size)
#
#     # Build interference matrix
#     for col_id in range(matrix_size[1]):
#         for row_id in range(matrix_size[0]):
#             cta_1 = row['cta_quota'][1][col_id]
#             cta_2 = row['cta_quota'][2][row_id]
#
#             kidx_1 = const.translate_gpusim_kidx(apps[0], col_id)
#             kidx_2 = const.translate_gpusim_kidx(apps[1], row_id)
#             dynamic_idx = (apps[0], kidx_1, int(cta_1),
#                            apps[1], kidx_2, int(cta_2))
#
#             if dynamic_idx not in df_dynamic_index.index:
#                 # No prediction for this ctx config
#                 return None
#
#             sld_1 = df_dynamic_index.loc[dynamic_idx, 'sld'][1]
#             sld_2 = df_dynamic_index.loc[dynamic_idx, 'sld'][2]
#
#             interference_1[row_id, col_id] = sld_1
#             interference_2[row_id, col_id] = sld_2
#
#     # Get baseline runtime
#     interference = [interference_1, interference_2]
#
#     return interference


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


# FIXME: this function should take serial matrix to handle kernel serialization
def predict_app(apps, interference, at_least_one=True,
                upper_lim=[math.inf, math.inf], output=False):
    # Get seq runtimes
    runtimes = [const.get_seq_cycles(app) for app in apps]
    sim_results = scheduler.simulate(runtimes, interference,
                                     upper_lim=upper_lim,
                                     at_least_one=at_least_one,
                                     finish_remaining=False)

    scaled_runtime = [[int(t) for t in app] for app in sim_results[0]]
    time_stamps = np.array([const.get_from_to(scaled_app) for scaled_app in
                            scaled_runtime])
    end_stamps = time_stamps[:, 1]
    #     print('Predicted runtime:', scaled_runtime)

    norm_ipc = scheduler.calculate_norm_ipc(runtimes, scaled_runtime)

    sld = scheduler.calculate_qos(runtimes, end_stamps, short=True,
                                  revert=False)

    if output:
        print('Norm IPC:')
        print('app 0: ', norm_ipc[0])
        print('app 1: ', norm_ipc[1])
        print('Predicted weighted speedup:', sum(sld))

    return time_stamps, norm_ipc, sld
