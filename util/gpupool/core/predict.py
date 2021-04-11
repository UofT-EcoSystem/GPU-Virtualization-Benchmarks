import numpy as np
import sys
import pandas as pd
import math
import time

from gpupool.core.helper import *
import job_launching.pair as run_pair
import data.scripts.common.help_iso as hi
import data.scripts.common.constants as const
import data.scripts.gen_graphs.gen_altair_timeline as gen_altair
import data.scripts.predict.predict_slowdown as tree_model
from gpupool.core.configs import *


def debug_print(text, condition=True):
    if condition:
        print(text)


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
                                                         offset_times)],
                               dtype=object)
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

    # Prepare df_dynamic #
    df_dynamic = const.get_pickle('pair_dynamic.pkl')
    df_dynamic = df_dynamic.astype({'1_intra': 'int32', '2_intra': 'int32'})
    df_dynamic_best = df_dynamic.sort_values(
        'ws', ascending=False).drop_duplicates(
        ['1_bench', '1_kidx', '2_bench', '2_kidx']).set_index(
        ['1_bench', '1_kidx', '2_bench', '2_kidx'], drop=True)
    df_dynamic_index = df_dynamic.set_index(
        ['1_bench', '2_bench'], drop=True)
    df_dynamic_index_all = df_dynamic.set_index(
        ['1_bench', '2_bench', '1_intra', '2_intra'], drop=True
    ).sort_index()

    # Prepare df_intra
    df_intra = const.get_pickle('intra.pkl').set_index(
        ['pair_str', '1_kidx', 'intra']).sort_index()
    # Shave off unused column
    USED_COLS = list(set().union(list(tree_model.metric_dict.values()),
                     const.EXEC_CTX))
    df_intra = df_intra[USED_COLS]

    def __init__(self, jobs, offset_ratio=0):
        self.jobs = jobs
        self.job_names = [job.name for job in self.jobs]
        self.offset_ratio = offset_ratio

        num_cols = const.get_num_kernels(self.job_names[0])
        num_rows = const.get_num_kernels(self.job_names[1])
        matrix_size = (num_rows, num_cols, 2)

        self.config_matrix = np.zeros(matrix_size, dtype=int)
        self.interference_matrix = np.zeros(matrix_size)

        # self.interference_hit = np.zeros(matrix_size, dtype=int)
        # self.serial = np.zeros(matrix_size, dtype=int)

    # Return Performance
    def app_wise_full_and_steady(self, steady=False, at_least_once=False):
        # # Dump input for C++ implementation
        # with open("seq_cycles.csv", "w+") as f:
        #     for job in self.jobs:
        #         string_cycles = [str(c) for c in job.get_seq_cycles()]
        #         f.write(",".join(string_cycles))
        #         f.write("\n")
        #
        # np.savetxt("inter1.csv", self.interference_matrix[0], delimiter=",")
        # np.savetxt("inter2.csv", self.interference_matrix[1], delimiter=",")
        #
        # print("job 0 limit", self.jobs[0].num_iters)
        # print("job 1 limit", self.jobs[1].num_iters)

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
        # self.interference_hit = np.zeros(self.interference_hit.shape,
        #                                  dtype=int)

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
                if iter_[app_idx] % STEADY_STEP == 0:
                    # compare qos to the past qos
                    qos_loss = find_qos_loss(shared_runtimes[app_idx],
                                             iter_[app_idx],
                                             seq_runtimes[app_idx])

                    if abs(past_qos_loss[app_idx] - qos_loss) \
                            < QOS_LOSS_ERROR \
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
            # self.interference_hit[kidx_pair] = 1
            app0_ker_scaled = \
                remaining_runtimes[0] / self.interference_matrix[kidx_pair][0]
            app1_ker_scaled = \
                remaining_runtimes[1] / self.interference_matrix[kidx_pair][1]
            # logger.debug("app1 kernel scaled runtime is: {}".format(
            # app1_ker_scaled))

            # Advance scaled runtime by the shorter scaled kernel
            short_scaled = min(app0_ker_scaled, app1_ker_scaled)
            shared_runtimes[0][-1] += short_scaled
            shared_runtimes[1][-1] += short_scaled

            diff_scaled = abs(app0_ker_scaled - app1_ker_scaled)

            steady_exit = False
            if diff_scaled <= EQUALITY_ERROR:
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
                                        self.interference_matrix[kidx_pair][1]

                steady_exit |= handle_completed_kernel(0)
            else:
                # app1 kernel will finish before app0 kernel update the total
                # runtime of both kernels to app1 kernel runtime since it
                # finished first

                # compute raw remaining runtime of app0
                remaining_runtimes[0] = diff_scaled * \
                                        self.interference_matrix[kidx_pair][0]

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
            # print("stage 2 took seconds", time.perf_counter() - start_time)
            # print("qos", steady_state_qos)
            # print("iter", steady_state_iter)
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
    def train_boosting_tree(cls, train_all=True, df_train=None):
        if train_all:
            x, y = tree_model.prepare_datasets(cls.df_dynamic)

            offset = int(x.shape[0] * cls.TRAINING_SET_RATIO)
            x_train, y_train = x[:offset], y[:offset]
            x_test, y_test = x[offset:], y[offset:]

            model = tree_model.train(x_train, y_train)

            y_predicted = model.predict(x_test)
            delta = np.average(np.abs((y_predicted - y_test) / y_predicted))
            print("Average abs relative error in test set: {:.2f}%."
                  .format(delta * 100))

        else:
            x_train, y_train = tree_model.prepare_datasets(df_train)
            model = tree_model.train(x_train, y_train, cross_validation=False)

        return model

    def kernel_wise_prediction(self, accuracy_mode, model=None):
        # Cross product of kernels from each job
        df_prod_tot = self.jobs[0].df_benchmarks.merge(
            self.jobs[1].df_benchmarks, how='cross')

        # Drop the ones that exceed execution context resource
        for rsrc in const.EXEC_CTX:
            df_prod_tot = df_prod_tot[
                (df_prod_tot[rsrc + '_x'] + df_prod_tot[rsrc + '_y']) <= 1.0
                ]

        # Prep for inference data
        xs = tree_model.prepare_datasets(df_prod_tot, training=False)

        # Run inference on all pairs
        # sld = [model.predict(x) for x in xs]
        sld = model.predict(xs)

        half_len = int(len(sld) / 2)
        df_prod_tot['sld_x'] = sld[0:half_len]
        df_prod_tot['sld_y'] = sld[half_len:]
        df_prod_tot['ws'] = df_prod_tot['sld_x'] + df_prod_tot['sld_y']

        # accuracy mode, drop configs not in df_dynamic
        # super hacky
        if accuracy_mode:
            def process_pair(metrics: tuple):
                if metrics[0] > metrics[1]:
                    return metrics[1], metrics[0], metrics[3], metrics[2]
                else:
                    return metrics

            columns = ['pair_str_x', 'pair_str_y', 'intra_x', 'intra_y']
            prod_data = df_prod_tot[columns].values
            prod_data = [tuple(x) for x in prod_data]

            dynamic_data = self.df_dynamic_index_all.index.tolist()

            bool_index = [(process_pair(row) in dynamic_data)
                          for row in prod_data]
            df_prod_tot = df_prod_tot[bool_index]

        # Append default time multiplexing performance
        id_pairs = [(x, y) for x in range(len(self.jobs[0].benchmarks))
                    for y in range(len(self.jobs[1].benchmarks))]
        df_default = pd.DataFrame(id_pairs,
                                  columns=['kernel_id_x', 'kernel_id_y'])
        df_default['intra_x'] = df_default['intra_y'] = -1
        df_default['sld_x'] = df_default['sld_y'] = 0.5
        df_default['ws'] = 1.0
        df_prod_tot = df_prod_tot.append(df_default, ignore_index=True)

        # Only keep rows with max weighted speedup
        # Sort by kernel ids for vectorized write
        df_prod_tot = df_prod_tot.sort_values('ws', ascending=False)\
            .drop_duplicates(['kernel_id_y', 'kernel_id_x'])\
            .sort_values(['kernel_id_y', 'kernel_id_x'])

        self.interference_matrix = \
            df_prod_tot[['sld_x', 'sld_y']].values.reshape(
                self.interference_matrix.shape)

        self.config_matrix = \
            df_prod_tot[['intra_x', 'intra_y']].values.reshape(
                self.config_matrix.shape)

    def kernel_wise_gpusim(self):
        print("Parent kernel_wise_gpusim function prototype called.")
        sys.exit(1)
        pass

    def verify_kernel_wise(self, accuracy_mode, model=None):
        from copy import deepcopy
        self.kernel_wise_prediction(accuracy_mode, model)
        prediction = deepcopy(self.interference_matrix)

        self.kernel_wise_gpusim()
        ground_truth = deepcopy(self.interference_matrix)

        delta = [np.abs(p - g) / g
                 for p, g in zip(prediction, ground_truth)]

        delta = np.array([matrix.reshape((matrix.size,))
                          for matrix in delta]).flatten()

        return sum(delta) / len(delta), len(delta)

    def get_real_performance(self):
        # Update interference matrix
        for matrix_idx in np.ndindex(*(self.interference_matrix.shape[:2])):
            # if the predicted config was time multiplex, skip
            if np.all(self.config_matrix[matrix_idx] == [-1, -1]):
                continue

            # Look up real performance from df_dynamic
            row_idx = matrix_idx[0]
            col_idx = matrix_idx[1]
            benchmarks = [self.jobs[0].benchmarks[col_idx],
                          self.jobs[1].benchmarks[row_idx]]
            ctas = self.config_matrix[matrix_idx]

            sorted_bench = sorted(benchmarks)

            if sorted_bench != benchmarks:
                # Flip everything
                ctas = ctas[::-1]

            index = tuple(sorted_bench) + tuple(ctas)
            if index in self.df_dynamic_index_all.index:
                df_bench_pair = self.df_dynamic_index_all.loc[index]
                sld = df_bench_pair['sld'].iloc[0][1:]
            else:
                print("Non time-multiplex pair_str not in df_dynamic")
                exit(-1)

            if sorted_bench != benchmarks:
                sld.reverse()

            self.interference_matrix[matrix_idx] = np.array(sld)

        # Run second stage full
        perf = self.app_wise_full_and_steady(steady=False, at_least_once=False)
        return perf

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
            def time_multiplex(sorted_bench):
                # If no feasible pair dynamic config, let the kernels
                # time multiplex using its best intra config
                _cta_setting = []
                _sld = []
                for bench in sorted_bench:
                    best_intra = self.df_intra.xs(bench)['norm_ipc'].idxmax(
                        axis=0)
                    _cta_setting.append(best_intra)
                    _sld.append(0.5)

                return _cta_setting, _sld

            serial = False

            real_bench = [(job.benchmarks[idx], 1)
                          for job, idx in zip(self.jobs, bench_idx)]
            same_bench = real_bench[0] == real_bench[1]

            if same_bench:
                # Identical benchmark pair. Get settings from df_intra
                # df_bench = self.df_intra.xs(real_bench[0])
                #
                # intra_max = df_bench.index.max()
                # intra_ipc_max = df_bench['norm_ipc'].idxmax(axis=0)
                # if intra_max == intra_ipc_max:
                #     # Max setting is the max allowed setting
                #     max_cta = const.get_max_cta_per_sm(real_bench[0][0],
                #                                        real_bench[0][1])
                #     if intra_max * 2 > max_cta:
                #         cta_setting = max(max_cta // 2, 1)
                #         # Might be a bit pessimistic here:
                #         sld = 0.5
                #     else:
                #         cta_setting = intra_max
                #
                #         # FIXME: sketchy?
                #         if intra_max * 2 in df_bench.index:
                #             sld = df_bench.loc[intra_max * 2]['norm_ipc'] / 2
                #         else:
                #             sld = 0.5
                # else:
                #     cta_setting = intra_ipc_max // 2
                #     sld = 0.5

                # return [cta_setting, cta_setting], [sld, sld], serial
                cta_setting, sld = time_multiplex(real_bench)
                return cta_setting, sld, serial

            else:
                # Different benchmarks. Get settings from df_dynamic
                # Benchmarks within each pair in dynamic are sorted
                sorted_real_bench = real_bench.copy()
                sorted_real_bench.sort(key=lambda x: x[0])

                pair_index = sorted_real_bench[0] + sorted_real_bench[1]

                if pair_index in self.df_dynamic_best.index:
                    series_best = self.df_dynamic_best.loc[pair_index]

                    if sum(series_best['sld']) < 1.0:
                        cta_setting, sld = time_multiplex(sorted_real_bench)
                    else:
                        cta_setting = [series_best['1_intra'],
                                       series_best['2_intra']]
                        sld = series_best['sld'][1:3]
                else:
                    cta_setting, sld = time_multiplex(sorted_real_bench)

                if real_bench != sorted_real_bench:
                    cta_setting.reverse()
                    sld.reverse()

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
        # if predictor_config.alloc.name == Allocation.One_D.name:
        #     # Find all possible 1D allocations
        #     configs = run_pair.find_ctx_configs(self.job_names,
        #                                         self.NUM_SLICES_1D)
        #
        #     if len(configs) == 0:
        #         return result
        #
        #     ctx = [hi.check_config('1_ctx', c, default=0) for c in configs]
        #
        #     run_options = [RunOption1D(ctx_value, self.jobs)
        #                    for ctx_value in ctx]
        # elif predictor_config.alloc.name == Allocation.Three_D.name:
        #     # step = 1.0 / num_slices
        #     # run_options = [RunOption3D(self.jobs, ratio)
        #     #                for ratio in np.arange(0, 1, step)]
        #     run_options = [RunOption3D(self.jobs, 0)]
        # else:
        #     # 2D is unimplemented
        #     print("PairJob: 2D is unimplemented.")
        #     sys.exit(1)

        # Hard-code to 3D #
        option = RunOption3D(self.jobs, 0)

        # Profiling
        start_stage1 = time.perf_counter()

        # Run Stage 1 to get kernel-wise matrices
        if predictor_config.stage_1.name == StageOne.GPUSim.name:
            option.kernel_wise_gpusim()
        else:
            # 'BOOST_TREE'
            option.kernel_wise_prediction(
                accuracy_mode=predictor_config.accuracy_mode,
                model=predictor_config.get_model())

        # Profiling
        time_stage1 = time.perf_counter() - start_stage1
        start_stage2 = time.perf_counter()

        # Run Stage 2 to get app-wise matrices
        if predictor_config.stage_2.name == StageTwo.Full.name:
            performance = option.app_wise_full_and_steady(
                at_least_once=predictor_config.at_least_once)
        elif predictor_config.stage_2.name == StageTwo.Steady.name:
            performance = option.app_wise_full_and_steady(
                steady=True,
                at_least_once=predictor_config.at_least_once)
        elif predictor_config.stage_2.name == StageTwo.Weighted.name:
            performance = option.app_wise_weighted()
        else:
            performance = option.app_wise_gpusim()

        time_stage2 = time.perf_counter() - start_stage2

        # Add profile info
        performance.time_stage1 = time_stage1
        performance.time_stage2 = time_stage2

        result[option_col] = option
        result[perf_col] = performance
        result[ws_col] = performance.weighted_speedup()

        return result

    def get_best_effort_performance(self):
        option = RunOption3D(self.jobs, 0)
        # Get ground truth for kernels
        option.kernel_wise_gpusim()
        perf = option.app_wise_full_and_steady(
            at_least_once=False)

        return perf

    def verify_boosting_tree(self, predictor_config):
        option = RunOption3D(self.jobs, 0)
        delta = option.verify_kernel_wise(
            predictor_config.accuracy_mode,
            predictor_config.get_model()
        )

        return delta

    def name(self):
        return "+".join(self.job_names)
