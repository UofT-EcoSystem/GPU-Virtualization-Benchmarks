import numpy as np
import scipy
import pandas as pd
import math

import job_launching.pair as run_pair
import data.scripts.common.constants as const
import data.scripts.common.help_iso as hi
import data.scripts.gen_tables.gen_lut_configs as gen_lut


def _predict_1d_best(apps, at_least_one, upper_limit, num_slice=4):
    configs = run_pair.find_ctx_configs(apps, '', num_slice)

    if len(configs) == 0:
        return None

    ctx = [hi.check_config('1_ctx', c, default=0) for c in configs]

    df_dynamic = const.get_pickle('pair_dynamic.pkl')
    df_intra = const.get_pickle('intra.pkl')

    best_ctx = 0
    best_ws = 0
    best_sld = []
    best_time_stamps = []
    best_norm_ipc = []

    for ctx_value in ctx:
        quota, configs = gen_lut.get_config_matrix_from_ctx(apps, ctx_value)
        interference = gen_lut.get_interference_matrix(apps, configs,
                                                       df_dynamic, df_intra)
        time_stamps, norm_ipc, sld = gen_lut.predict_app(apps, interference,
                                                         at_least_one,
                                                         upper_limit)

        if sum(sld) > best_ws:
            best_ctx = ctx_value
            best_ws = sum(sld)
            best_sld = sld
            best_time_stamps = time_stamps
            best_norm_ipc = norm_ipc

    return best_sld, best_ctx, best_time_stamps, best_norm_ipc


def _predict_3d(apps, at_least_one, upper_limit, allow_serial=False):
    df_dynamic = const.get_pickle('pair_dynamic.pkl')
    df_intra = const.get_pickle('intra.pkl')

    configs, interference, serial_matrix = gen_lut.get_lut_matrix(apps,
                                                                  df_dynamic,
                                                                  df_intra)

    if (not allow_serial) and (1 in serial_matrix):
        return None

    time_stamps, norm_ipc, sld = gen_lut.predict_app(apps, interference,
                                                     at_least_one,
                                                     upper_limit)

    # Find out max dominant resource usage per stream
    s1_usage = [np.max(configs[0][:, idx]) *
                const.get_dominant_usage(const.syn_yaml[apps[0]][idx])[0][1]
                for idx in range(configs[0].shape[1])]
    s2_usage = [np.max(configs[1][idx, :]) *
                const.get_dominant_usage(const.syn_yaml[apps[1]][idx])[0][1]
                for idx in range(configs[1].shape[0])]

    usage = [s1_usage, s2_usage]

    return sld, usage, time_stamps, norm_ipc


def build_df_predict(pairs, at_least_one, upper_limit=None):
    if at_least_one:
        upper_limit = [math.inf, math.inf]

    if not upper_limit:
        upper_limit = [100, 100]

    prediction = []

    for apps in pairs:
        entry = {}
        result_1d = _predict_1d_best(apps, at_least_one=at_least_one,
                                     upper_limit=upper_limit)
        if not result_1d:
            continue

        best_sld, best_ctx, time_stamps, norm_ipc = result_1d
        entry['1d'] = sum(best_sld)
        entry['1d_sld'] = best_sld
        entry['ctx'] = best_ctx
        entry['1d_time_stamps'] = time_stamps
        entry['1d_norm_ipc'] = norm_ipc

        result_3d = _predict_3d(apps, at_least_one=at_least_one,
                                upper_limit=upper_limit,
                                allow_serial=False)

        if not result_3d:
            continue

        sld, usage, time_stamps, norm_ipc = result_3d
        entry['3d'] = sum(sld)
        entry['3d_sld'] = sld
        entry['3d_usage'] = usage
        entry['3d_time_stamps'] = time_stamps
        entry['3d_norm_ipc'] = norm_ipc

        entry['pair'] = '+'.join(apps)

        prediction.append(entry)

    df_predict = pd.DataFrame(prediction)

    geomean = {'pair': 'geomean',
               '1d': scipy.stats.mstats.gmean(df_predict['1d']),
               '3d': scipy.stats.mstats.gmean(df_predict['3d']), }
    df_predict = df_predict.append(geomean, ignore_index=True)

    return df_predict
