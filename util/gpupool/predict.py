import numpy as np
import sys
import pandas as pd
from enum import Enum
import math
import time

import job_launching.pair as run_pair
import data.scripts.common.help_iso as hi
import data.scripts.common.constants as const
import data.scripts.gen_graphs.gen_altair_timeline as gen_altair
import data.scripts.predict.predict_slowdown as tree_model


def debug_print(text, condition):
    if condition:
        print(text)


class Allocation(Enum):
    One_D = 1
    Two_D = 2
    Three_D = 3


class StageOne(Enum):
    GPUSim = 1
    BoostTree = 2


class StageTwo(Enum):
    Full = 1
    Steady = 2
    Weighted = 3
    GPUSim = 4


class Performance:
    def __init__(self, jobs):
        self.jobs = jobs

        self.start_stamps = np.array([])
        self.end_stamps = np.array([])
        self.norm_ipc = np.array([])
        self.sld = np.array([])
        self.steady_iter = None
        self.delta = 0

        # Profiling info
        self.time_stage1 = -1
        self.time_stage2 = -1

    def fill_with_duration(self, shared_runtimes, seq_runtimes, offset_times):
        # This assumes kernels are run back to back
        timestamps = np.array([const.get_from_to(shared, offset)
                               for shared, offset in zip(shared_runtimes,
                                                         offset_times)])
        self.start_stamps = timestamps[:, 0]
        self.end_stamps = timestamps[:, 1]

        circular_runtimes = [np.resize(seq, len(shared))
                             for shared, seq in zip(shared_runtimes,
                                                    seq_runtimes)
                             ]

        self.norm_ipc = [np.divide(circular, shared) for
                         circular, shared in zip(circular_runtimes,
                                                 shared_runtimes)
                         ]

        # Slowdown calculation should exclude the initial offset period
        # The excluded list should be the one that didn't get delayed
        offset_times.reverse()
        adjusted_end_stamps = []
        start = []
        for start_stamps, end_stamps, offset, seq in \
                zip(self.start_stamps, self.end_stamps, offset_times,
                    seq_runtimes):
            if offset > 0:
                # Get rid of the first iteration
                num_kernels = len(seq)
                adjusted_end_stamps.append(end_stamps[num_kernels:])
                start.append(start_stamps[num_kernels])
            else:
                adjusted_end_stamps.append(end_stamps)
                start.append(0)

        self.sld = hi.calculate_sld_short(adjusted_end_stamps, seq_runtimes,
                                          start)

    def fill_with_slowdown(self, sld, steady_iter=None):
        self.sld = sld

        if steady_iter:
            self.steady_iter = np.array(steady_iter)

    def visualize(self, chart_title, width=900, xmax=None):
        def get_kernels(idx):
            kernels = np.arange(1,
                                self.jobs[idx].get_num_benchmarks() + 1)
            kernels = np.resize(kernels, len(self.start_stamps[idx]))
            kernels = ['{}:{}'.format(idx + 1, k) for k in kernels]

            return kernels

        stream = [np.full_like(stamps, stream_id + 1)
                  for stream_id, stamps in enumerate(self.start_stamps)]

        kernel = [get_kernels(stream_id)
                  for stream_id in range(len(self.jobs))]

        data = pd.DataFrame()

        data["start"] = np.concatenate(self.start_stamps)
        data["end"] = np.concatenate(self.end_stamps)

        data["stream"] = np.concatenate(stream).astype(int).astype('str')
        data["kernel"] = np.concatenate(kernel)

        data['position'] = (data['start'] + data['end']) / 2
        data['norm'] = np.concatenate(self.norm_ipc)
        data['norm'] = data['norm'].round(2)

        return gen_altair.draw_altair(data, chart_title, width, xmax)

    def weighted_speedup(self):
        return sum(self.sld)

    def __eq__(self, other):
        return sum(self.sld) == sum(other.sld)

    def __gt__(self, other):
        return sum(self.sld) > sum(other.sld)


class RunOption:
    EQUALITY_ERROR = 0.0001
    STEADY_STEP = 20
    QOS_LOSS_ERROR = 0.01

    TRAINING_SET_RATIO = 0.8

    df_dynamic = const.get_pickle('pair_dynamic.pkl')
    df_dynamic_best = df_dynamic.sort_values(
        'ws', ascending=False).drop_duplicates(
        ['1_bench', '1_kidx', '2_bench', '2_kidx']).set_index(
        ['1_bench', '1_kidx', '2_bench', '2_kidx'], drop=True)

    df_intra = const.get_pickle('intra.pkl').set_index(
        ['pair_str', '1_kidx', 'intra']).sort_index()

    model = None

    def __init__(self, jobs, offset_ratio=0):
        self.jobs = jobs
        self.job_names = [job.name for job in self.jobs]
        self.offset_ratio = offset_ratio

        num_cols = const.get_num_kernels(self.job_names[0])
        num_rows = const.get_num_kernels(self.job_names[1])
        matrix_size = (num_rows, num_cols)

        self.config_matrix = [np.zeros(matrix_size, dtype=int),
                              np.zeros(matrix_size, dtype=int)]
        self.interference_matrix = [np.zeros(matrix_size),
                                    np.zeros(matrix_size)]
        self.interference_hit = np.zeros(matrix_size, dtype=int)
        self.serial = np.zeros(matrix_size, dtype=int)

    # Return Performance
    def app_wise_full_and_steady(self, steady=False, at_least_once=False):
        seq_runtimes = [job.get_seq_cycles() for job in self.jobs]

        # to keep track of iterations completed by apps
        iter_ = [0, 0]
        # scaled_runtimes array is in the form of
        # [[k1 time, k2 time...], [k1 time, k2 time, ...]]
        shared_runtimes = [[0], [0]]
        # indeces of kernels for two apps - by default 0 and 0
        kidx = [0, 0]
        # by default the two kernels launch simultaneously
        remaining_runtimes = [seq_runtimes[0][kidx[0]],
                              seq_runtimes[1][kidx[1]]]
        # app that has remaining kernels after the other app finished
        remaining_app = 0
        # index of current kernel in remaining app
        remaining_app_kidx = 0

        # past and total accumulated runtimes of apps
        past_qos_loss = [0, 0]
        # list to keep track of estimated qos using steady state estimate
        steady_state_qos = [-1, -1]
        steady_state_iter = [0, 0]

        # Clear the interference_hit matrix just in case
        self.interference_hit = np.zeros(self.interference_hit.shape,
                                         dtype=int)

        # initialize starting offset by fast forward app 0
        forward_cycles = self.offset_ratio * sum(seq_runtimes[0])
        remaining_cycles = forward_cycles
        while True:
            if seq_runtimes[0][kidx[0]] <= remaining_cycles:
                shared_runtimes[0][-1] += seq_runtimes[0][kidx[0]]
                remaining_cycles -= seq_runtimes[0][kidx[0]]

                # advance to the next kernel
                kidx[0] = (kidx[0] + 1) % len(seq_runtimes[0])
                remaining_runtimes[0] = seq_runtimes[0][kidx[0]]
                shared_runtimes[0].append(0)
            else:
                shared_runtimes[0][-1] += remaining_cycles
                remaining_runtimes[0] -= remaining_cycles
                break

        def find_qos_loss(scaled_runtime, num_iter, _isolated_runtime):
            return (sum(_isolated_runtime) * num_iter) / sum(scaled_runtime)

        def handle_completed_kernel(app_idx):
            can_exit = False
            other_app_idx = (app_idx + 1) % len(self.jobs)

            kidx[app_idx] += 1

            # if app has finished an iteration
            if kidx[app_idx] == len(seq_runtimes[app_idx]):
                # re-assignment of outer scope variables
                nonlocal remaining_app, remaining_app_kidx
                remaining_app = other_app_idx
                remaining_app_kidx = kidx[other_app_idx]

                kidx[app_idx] = 0

                # app has completed an iteration of all kernels
                iter_[app_idx] += 1

                # evaluate steady state
                if iter_[app_idx] % RunOption.STEADY_STEP == 0:
                    # compare qos to the past qos
                    qos_loss = find_qos_loss(shared_runtimes[app_idx],
                                             iter_[app_idx],
                                             seq_runtimes[app_idx])

                    if abs(past_qos_loss[app_idx] - qos_loss) \
                            < RunOption.QOS_LOSS_ERROR \
                            and steady_state_qos[app_idx] == -1:
                        steady_state_qos[app_idx] = qos_loss
                        steady_state_iter[app_idx] = iter_[app_idx]

                        # Check if we reach steady state for all apps
                        can_exit = all([qos != -1 for qos in steady_state_qos])

                    # update past qos loss to the current qos loss
                    past_qos_loss[app_idx] = qos_loss

            remaining_runtimes[app_idx] = seq_runtimes[app_idx][kidx[app_idx]]

            shared_runtimes[app_idx].append(0)
            return can_exit

        limit = [job.num_iters for job in self.jobs]
        if at_least_once:
            limit = [math.inf, math.inf]

        # main loop of the simulation
        while iter_[0] < limit[0] and iter_[1] < limit[1]:
            # figure out who finishes first by scaling the runtimes by the
            # slowdowns
            kidx_pair = (kidx[1], kidx[0])
            self.interference_hit[kidx_pair] = 1
            app0_ker_scaled = \
                remaining_runtimes[0] / self.interference_matrix[0][kidx_pair]
            app1_ker_scaled = \
                remaining_runtimes[1] / self.interference_matrix[1][kidx_pair]
            # logger.debug("app1 kernel scaled runtime is: {}".format(
            # app1_ker_scaled))

            # Advance scaled runtime by the shorter scaled kernel
            short_scaled = min(app0_ker_scaled, app1_ker_scaled)
            shared_runtimes[0][-1] += short_scaled
            shared_runtimes[1][-1] += short_scaled

            diff_scaled = abs(app0_ker_scaled - app1_ker_scaled)

            steady_exit = False
            if diff_scaled <= RunOption.EQUALITY_ERROR:
                # both kernels finished at the same time update the total
                # runtime of both kernels to either kernel runtime since they
                # have finished together
                steady_exit |= handle_completed_kernel(0)
                steady_exit |= handle_completed_kernel(1)
            elif app0_ker_scaled < app1_ker_scaled:
                # app0 kernel will finish before app1 kernel update the total
                # runtime of both kernels to app0 kernel runtime since it
                # finished first

                # compute raw remaining runtime of app1
                remaining_runtimes[1] = diff_scaled * \
                                        self.interference_matrix[1][kidx_pair]

                steady_exit |= handle_completed_kernel(0)
            else:
                # app1 kernel will finish before app0 kernel update the total
                # runtime of both kernels to app1 kernel runtime since it
                # finished first

                # compute raw remaining runtime of app0
                remaining_runtimes[0] = diff_scaled * \
                                        self.interference_matrix[0][kidx_pair]

                steady_exit |= handle_completed_kernel(1)

            if steady and steady_exit:
                break

            if at_least_once:
                if self.offset_ratio == 0:
                    if iter_[0] >= 1 and iter_[1] >= 1:
                        break
                else:
                    if iter_[0] >= 2 and iter_[1] >= 1:
                        break
        # end of loop

        # finish off the last iteration of remaining app in isolation
        shared_runtimes[remaining_app][-1] += \
            remaining_runtimes[remaining_app]
        for idx in range(remaining_app_kidx + 1,
                         len(seq_runtimes[remaining_app])):
            shared_runtimes[remaining_app].append(
                seq_runtimes[remaining_app][idx])

        iter_[remaining_app] += 1

        # Handle app that did not get a steady state estimation
        for app_idx in range(len(self.jobs)):
            if steady_state_iter[app_idx] == 0:
                steady_state_iter[app_idx] = iter_[app_idx]
                steady_state_qos[app_idx] = find_qos_loss(
                    shared_runtimes[app_idx], iter_[app_idx],
                    seq_runtimes[app_idx])

        if steady:
            steady_perf = Performance(self.jobs)
            steady_perf.fill_with_slowdown(sld=steady_state_qos,
                                           steady_iter=steady_state_iter)
            return steady_perf
        else:
            if not at_least_once:
                # complete the rest of the required iterations of
                # remaining app in isolation
                remaining_iter = self.jobs[remaining_app].num_iters - iter_[
                    remaining_app]
                isolated_runtime = np.resize(
                    seq_runtimes[remaining_app],
                    remaining_iter * len(seq_runtimes[remaining_app])
                )
                shared_runtimes[remaining_app] += list(isolated_runtime)

            # Get rid of tailing zero
            shared_runtimes = [array[0:-1] if array[-1] == 0 else array
                               for array in shared_runtimes]

            # Build performance instances for full calculation and steady state
            full_perf = Performance(self.jobs)
            full_perf.fill_with_duration(shared_runtimes, seq_runtimes,
                                         offset_times=[0, forward_cycles])

            return full_perf

    def app_wise_weighted(self):
        seq_cycles = [job.get_seq_cycles() for job in self.jobs]
        kernel_weight = [seq_cycles[0] / np.sum(seq_cycles[0]),
                         seq_cycles[1] / np.sum(seq_cycles[1])]

        qos_loss = [0, 0]

        def weight_geom_mean(weights, data):
            exponent = 1 / sum(weights)
            prod = 1
            for weight_idx in range(weights.size):
                prod = prod * (data[weight_idx] ** weights[weight_idx])

            return prod ** exponent

        # find initial prediction for app0
        # weighted geom mean collapse to a vector (wrt app0)
        vector_app0 = []
        for idx in range(kernel_weight[1].size):
            vector_app0.append(
                weight_geom_mean(kernel_weight[0],
                                 self.interference_matrix[0][idx, :]))
        # collapse the vector wrt app1
        qos_loss[0] = weight_geom_mean(kernel_weight[1], vector_app0)

        # find initial prediction for app1
        # weighted geom mean collapse to a vector (wrt app0)
        vector_app1 = []
        for idx in range(kernel_weight[0].size):
            vector_app1.append(
                weight_geom_mean(kernel_weight[1],
                                 self.interference_matrix[1][:, idx]))
        # collapse the vector wrt app0
        qos_loss[1] = weight_geom_mean(kernel_weight[0], vector_app1)

        # This is commented out because we only calculate QoS short
        # find out which app finished first
        # tot_est_runtimes = [qos_loss[0] * sum(seq_cycles[0]) * iter_lim[0],
        #                     qos_loss[1] * sum(seq_cycles[1]) * iter_lim[1]]
        # longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
        # # find how long rem app was executing in isolation
        # rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
        # rem_app_isol_ratio = rem_app_isol_runtime /
        # tot_est_runtimes[longer_app]
        # # computed new QOS knowing the ratio of isolated runtime
        # qos_loss[longer_app] = rem_app_isol_ratio * 1 + (
        #             1 - rem_app_isol_ratio) * qos_loss[longer_app]

        perf = Performance(self.jobs)
        perf.fill_with_slowdown(qos_loss)

        return perf

    def app_wise_gpusim(self):
        # TODO
        pass

    @classmethod
    def train_boosting_tree(cls):
        x, y = tree_model.prepare_datasets(cls.df_dynamic)

        offset = int(x.shape[0] * cls.TRAINING_SET_RATIO)
        x_train, y_train = x[:offset], y[:offset]
        x_test, y_test = x[offset:], y[offset:]

        cls.model = tree_model.train(x_train, y_train)

        y_predicted = cls.model.predict(x_test)
        delta = np.average(np.abs((y_predicted - y_test) / y_predicted)) * 100

        print("Average abs relative error in test set: {:.2f}%.".format(delta))

    def kernel_wise_prediction(self):

        for matrix_idx, value in np.ndenumerate(self.interference_matrix[0]):
            row_idx = matrix_idx[0]
            col_idx = matrix_idx[1]

            benchmarks = [(self.jobs[0].benchmarks[col_idx], 1),
                          (self.jobs[1].benchmarks[row_idx], 1)]

            dfs = [self.df_intra.xs(bench).reset_index().rename(
                columns={"index": "intra"})
                   for bench in benchmarks]

            df_prod = dfs[0].assign(key=1).merge(
                dfs[1].assign(key=1), on="key").drop("key", axis=1)

            for rsrc in const.EXEC_CTX:
                df_prod = df_prod[
                    (df_prod[rsrc + '_x'] + df_prod[rsrc + '_y']) <= 1.0
                    ]

            def time_multiplex():
                # Simply run at best intra config and get 0.5 sld
                for i in range(len(self.jobs)):
                    self.config_matrix[i][matrix_idx] = \
                        dfs[i]['norm_ipc'].idxmax(axis=0)
                    self.interference_matrix[i][matrix_idx] = 0.5

            if len(df_prod.index) == 0:
                time_multiplex()
            else:
                xs = tree_model.prepare_datasets(df_prod,
                                                 training=False)
                config = df_prod[['intra_x', 'intra_y']]

                sld = np.array([RunOption.model.predict(x) for x in xs])
                ws = np.add(sld[0], sld[1])

                best_ws = max(ws)

                if best_ws < 1.0:
                    time_multiplex()
                else:
                    idx_ws = np.argmax(ws)
                    self.config_matrix[0][matrix_idx] = \
                        int(config.iloc[idx_ws]['intra_x'])
                    self.config_matrix[1][matrix_idx] = \
                        int(config.iloc[idx_ws]['intra_y'])

                    self.interference_matrix[0][matrix_idx] = sld[0][idx_ws]
                    self.interference_matrix[1][matrix_idx] = sld[1][idx_ws]

    def kernel_wise_gpusim(self):
        sys.exit(1)
        pass

    def _pretty_print_matrix(self, title, data):
        for job, job_data in zip(self.jobs, data):
            print(job.name, title)
            print(pd.DataFrame(job_data, index=self.jobs[1].benchmarks,
                               columns=self.jobs[0].benchmarks))
            print('-' * 100)

    def print_interference(self):
        self._pretty_print_matrix("Interference", self.interference_matrix)

    def print_config(self):
        self._pretty_print_matrix("Config", self.config_matrix)

    def serial_required(self):
        return 1 in self.serial

    def interference_hit_percentage(self):
        return np.count_nonzero(self.interference_hit) / \
               self.interference_hit.size


class RunOption1D(RunOption):
    def __init__(self, ctx, jobs):
        super(RunOption1D, self).__init__(jobs)

        # 1D config is fixed based on ctx value, calculate now
        self.ctx = [ctx, 1 - ctx]
        self.quota = [const.get_cta_from_ctx(const.get_dominant_usage(job),
                                             job_ctx, job)
                      for job, job_ctx in zip(self.job_names, self.ctx)]

        # config for job 0
        for idx, kernel_quota in enumerate(self.quota[0]):
            self.config_matrix[0][:, idx] = int(kernel_quota)

        # config for job 1
        for idx, kernel_quota in enumerate(self.quota[1]):
            self.config_matrix[1][idx, :] = int(kernel_quota)

    def kernel_wise_gpusim(self):
        benchmarks = [[(bench, 1) for bench in job.benchmarks]
                      for job in self.jobs]

        kernel_columns = ['1_bench', '1_kidx', '2_bench', '2_kidx']

        for matrix_idx, value in np.ndenumerate(self.interference_matrix[0]):
            row_idx = matrix_idx[0]
            col_idx = matrix_idx[1]

            bench = [benchmarks[0][col_idx], benchmarks[1][row_idx]]
            kernel_configs = [c[matrix_idx] for c in self.config_matrix]

            if bench[0] == bench[1]:
                # Look at df_intra
                df_bench = self.df_intra.xs(bench)

                intra_total = sum(kernel_configs)
                nearest_intra = df_bench.index.get_loc(intra_total,
                                                       method='nearest')

                weight = [intra / intra_total for intra in kernel_configs]
                list_sld = [df_bench[nearest_intra]['norm_ipc'] * w
                            for w in weight]
            else:
                # Look at dynamic
                sorted_bench = bench.copy()
                sorted_bench.sort(key=lambda x: x[0])

                if sorted_bench != bench:
                    kernel_configs.reverse()

                value = sorted_bench[0] + sorted_bench[1]
                df_bench = self.df_dynamic[
                    self.df_dynamic[kernel_columns].isin(
                        value).all(axis=1)].copy()

                if df_bench.empty:
                    # TODO: Need to handle kernel serialization...
                    # FIXME: time multiplexing for now
                    # print("RunOption1D unimplemented error.")
                    # print(bench)
                    # sys.exit(1)
                    list_sld = [0.5, 0.5]

                else:
                    # FIXME: this needs to be removed when all dynamic pairs
                    #  are available
                    df_bench['distance'] = np.abs(df_bench['1_intra'] -
                                                  kernel_configs[0]) + \
                                           np.abs(df_bench['2_intra'] -
                                                  kernel_configs[1])

                    df_bench.sort_values('distance', inplace=True,
                                         ascending=True)

                    list_sld = df_bench['sld'].iloc[0][1:]

                # Flip it back
                if sorted_bench != bench:
                    list_sld.reverse()

            self.interference_matrix[0][matrix_idx] = list_sld[0]
            self.interference_matrix[1][matrix_idx] = list_sld[1]


class RunOption3D(RunOption):
    def __init__(self, job, offset_ratio):
        super(RunOption3D, self).__init__(job, offset_ratio)

    def kernel_wise_gpusim(self):
        # return list_cta, list_sld, serial?
        def get_cta_sld(bench_idx):
            serial = False

            real_bench = [(job.benchmarks[idx], 1)
                          for job, idx in zip(self.jobs, bench_idx)]
            same_bench = real_bench[0] == real_bench[1]

            if same_bench:
                # Identical benchmark pair. Get settings from df_intra
                df_bench = self.df_intra.xs(real_bench[0])

                intra_max = df_bench.index.max()
                intra_ipc_max = df_bench['norm_ipc'].idxmax(axis=0)
                if intra_max == intra_ipc_max:
                    # Max setting is the max allowed setting
                    max_cta = const.get_max_cta_per_sm(real_bench[0][0],
                                                       real_bench[0][1])
                    if intra_max * 2 > max_cta:
                        cta_setting = max(max_cta // 2, 1)
                        # Might be a bit pessimistic here:
                        sld = 0.5
                    else:
                        cta_setting = intra_max

                        # FIXME: sketchy?
                        if intra_max * 2 in df_bench.index:
                            sld = df_bench.loc[intra_max * 2]['norm_ipc'] / 2
                        else:
                            sld = 0.5
                else:
                    cta_setting = intra_ipc_max // 2
                    sld = 0.5

                return [cta_setting, cta_setting], [sld, sld], serial

            else:
                # Different benchmarks. Get settings from df_dynamic
                # Benchmarks within each pair in dynamic are sorted
                sorted_real_bench = real_bench.copy()
                sorted_real_bench.sort(key=lambda x: x[0])

                pair_index = sorted_real_bench[0] + sorted_real_bench[1]

                if pair_index in self.df_dynamic_best.index:
                    series_best = self.df_dynamic_best.loc[pair_index]
                    cta_setting = [series_best['1_intra'],
                                   series_best['2_intra']]
                    sld = series_best['sld'][1:3]

                    if real_bench != sorted_real_bench:
                        cta_setting.reverse()
                        sld.reverse()
                else:
                    # If no feasible pair dynamic config, let the kernels
                    # time multiplex using its best intra config
                    cta_setting = []
                    sld = []
                    # FIXME: get rid of serial
                    # serial = True
                    for bench in sorted_real_bench:
                        best_intra = self.df_intra.xs(bench)['norm_ipc'].idxmax(
                            axis=0)
                        cta_setting.append(best_intra)
                        sld.append(0.5)

                return cta_setting, sld, serial

        for matrix_idx, value in np.ndenumerate(self.interference_matrix[0]):
            row_idx = matrix_idx[0]
            col_idx = matrix_idx[1]

            list_cta, list_sld, pair_serial = get_cta_sld(
                [col_idx, row_idx])

            if len(list_cta) == 0:
                print("LUT config does not exist for {}", self.job_names)
                sys.exit(1)

            self.config_matrix[0][matrix_idx] = int(list_cta[0])
            self.config_matrix[1][matrix_idx] = int(list_cta[1])

            self.interference_matrix[0][matrix_idx] = list_sld[0]
            self.interference_matrix[1][matrix_idx] = list_sld[1]

            self.serial[matrix_idx] = int(pair_serial)


class PairJob:
    NUM_SLICES_1D = 4

    def __init__(self, jobs: list):
        self.jobs = jobs
        self.job_names = [job.name for job in self.jobs]

    # return RunOption, Performance, ws
    def get_gpupool_performance(self, predictor_config):
        # print("Getting performance for", self.job_names)

        option_col = predictor_config.get_option()
        perf_col = predictor_config.get_perf()
        ws_col = predictor_config.get_ws()
        result = {option_col: None, perf_col: None, ws_col: 0}

        # Create RunOptions for allocation design
        if predictor_config.alloc.name == Allocation.One_D.name:
            # Find all possible 1D allocations
            configs = run_pair.find_ctx_configs(self.job_names,
                                                self.NUM_SLICES_1D)

            if len(configs) == 0:
                return result

            ctx = [hi.check_config('1_ctx', c, default=0) for c in configs]

            run_options = [RunOption1D(ctx_value, self.jobs)
                           for ctx_value in ctx]
        elif predictor_config.alloc.name == Allocation.Three_D.name:
            # step = 1.0 / num_slices
            # run_options = [RunOption3D(self.jobs, ratio)
            #                for ratio in np.arange(0, 1, step)]
            run_options = [RunOption3D(self.jobs, 0)]
        else:
            # 2D is unimplemented
            print("PairJob: 2D is unimplemented.")
            sys.exit(1)

        # Profiling
        start_stage1 = time.perf_counter()

        # Run Stage 1 to get kernel-wise matrices
        for option in run_options:
            if predictor_config.stage_1.name == StageOne.GPUSim.name:
                option.kernel_wise_gpusim()
            else:
                # 'BOOST_TREE'
                option.kernel_wise_prediction()

        # Profiling
        time_stage1 = time.perf_counter() - start_stage1
        start_stage2 = time.perf_counter()

        # Run Stage 2 to get app-wise matrices
        performance = []
        for option in run_options:
            if predictor_config.stage_2.name == StageTwo.Full.name:
                performance.append(
                    option.app_wise_full_and_steady(
                        at_least_once=predictor_config.at_least_once)
                )
            elif predictor_config.stage_2.name == StageTwo.Steady.name:
                performance.append(
                    option.app_wise_full_and_steady(
                        steady=True,
                        at_least_once=predictor_config.at_least_once)
                )
            elif predictor_config.stage_2.name == StageTwo.Weighted.name:
                performance.append(option.app_wise_weighted())
            else:
                performance.append(option.app_wise_gpusim())

        time_stage2 = time.perf_counter() - start_stage2

        # only keep the best one (only matter for 1D)
        best_perf = max(performance)
        best_perf.delta = max(performance).weighted_speedup() - \
                          min(performance).weighted_speedup()

        # Add profile info
        best_perf.time_stage1 = time_stage1
        best_perf.time_stage2 = time_stage2

        best_idx = performance.index(best_perf)
        best_option = run_options[best_idx]

        result[option_col] = best_option
        result[perf_col] = best_perf
        result[ws_col] = best_perf.weighted_speedup()

        return result

    def get_best_effort_performance(self):
        option = RunOption3D(self.jobs, 0)
        # Get ground truth for kernels
        option.kernel_wise_gpusim()
        perf = option.app_wise_full_and_steady(
            at_least_once=False)

        return perf

    def name(self):
        return "+".join(self.job_names)
