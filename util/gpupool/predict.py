import numpy as np
import sys
import pandas as pd

from gpupool.workload import *
import job_launching.pair as run_pair
import data.scripts.common.help_iso as hi


ALLOCATION = ['1D', '2D', '3D']
STAGE_ONE = ['GPUSIM', 'BOOST_TREE']
STAGE_TWO = ['FULL', 'STEADY', 'WEIGHTED', 'GPUSIM']


class Performance:
    def __init__(self):
        self.timestamps = np.array()
        self.norm_ipc = np.array()
        self.sld = np.array()
        self.steady_iter = -1

    def fill_with_duration(self, shared_runtimes, seq_runtimes):
        # This assumes kernels are run back to back
        self.timestamps = const.get_from_to(shared_runtimes)

        circular_runtimes = \
            [np.resize(seq_runtimes[idx], len(app_time))
             for idx, app_time in enumerate(shared_runtimes)
             ]

        self.norm_ipc = np.divide(circular_runtimes, shared_runtimes)
        self.sld = hi.calculate_sld_short(self.timestamps[:, 1], seq_runtimes)

    def fill_with_slowdown(self, sld, steady_iter):
        self.sld = sld
        self.steady_iter = steady_iter

    def visualize(self):
        # TODO: call gen altair to draw timeline
        pass

    def __eq__(self, other):
        return sum(self.sld) == sum(other.sld)

    def __gt__(self, other):
        return sum(self.sld) > sum(self.sld)


class RunOption:
    EQUALITY_ERROR = 0.0001
    STEADY_STEP = 20
    QOS_LOSS_ERROR = 0.01

    def __init__(self, jobs):
        self.jobs = jobs
        self.job_names = [job.name() for job in self.jobs]

        self.config_matrix = np.array()
        self.interference_matrix = np.array()
        self.serial = np.array()

    # Return Performance
    # FIXME: handle kernel serialization
    def app_wise_full_and_steady(self, at_least_one=False):
        seq_runtimes = [job.get_seq_cycles() for job in self.jobs]

        # to keep track of iterations completed by apps
        iter_ = [0, 0]
        # scaled_runtimes array is in the form of
        # [[k1 time, k2 time...], [k1 time, k2 time, ...]]
        shared_runtimes = [[0], [0]]
        # indeces of kernels for two apps - by default 0 and 0
        kidx = [0, 0]
        # by default the two kernels launch simultaneously
        remaining_runtimes = [seq_runtimes[0][kidx[0]], seq_runtimes[1][kidx[1]]]
        # app that has remaining kernels after the other app finished
        remaining_app = 0
        # index of current kernel in remaining app
        remaining_app_kidx = 0
        # variable to keep track of offsets - used to detect completed period
        offsets = []

        # past and total accumulated runtimes of apps
        past_qos_loss = [0, 0]
        # list to keep track of estimated qos using steady state estimate
        steady_state_qos = [-1, -1]
        steady_state_iter = [0, 0]

        # initialize starting offset
        # start_offset(apps, rem_runtimes, ker, scaled_runtimes,
        # offsets, offset, offset_app)

        def find_qos_loss(scaled_runtime, num_iter, isolated_runtime):
            return sum(scaled_runtime) / (sum(isolated_runtime) * num_iter)

        def handle_completed_kernel(app_idx):
            other_app_idx = (app_idx + 1) % len(self.jobs)

            kidx[app_idx] += 1

            # if app0 has finished an iteration
            if kidx[app_idx] == len(seq_runtimes[app_idx]):
                # re-assignment of outer scope variables
                nonlocal remaining_app, remaining_app_kidx
                remaining_app = other_app_idx
                remaining_app_kidx = kidx[other_app_idx]

                kidx[app_idx] = 0

                # app0 has completed an iteration of all kernels
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

                # update past qos loss to the current qos loss
                past_qos_loss[app_idx] = qos_loss

            remaining_runtimes[app_idx] = seq_runtimes[app_idx][kidx[app_idx]]

            if iter_[app_idx] < self.jobs[app_idx].num_iters:
                shared_runtimes[app_idx].append(0)

        # main loop of the simulation
        while iter_[0] < self.jobs[0].num_iters \
                and iter_[1] < self.jobs[1].num_iters:
            # figure out who finishes first by scaling the runtimes by the
            # slowdowns
            kidx_pair = (kidx[1], kidx[0])
            app0_ker_scaled = \
                remaining_runtimes[0] / self.interference_matrix[0][kidx_pair]
            # logger.debug("app0 kernel scaled runtime is: {}".format(
            # app0_ker_scaled))
            app1_ker_scaled = \
                remaining_runtimes[1] / self.interference_matrix[1][kidx_pair]
            # logger.debug("app1 kernel scaled runtime is: {}".format(
            # app1_ker_scaled))

            # Advance scaled runtime by the shorter scaled kernel
            short_scaled = min(app0_ker_scaled, app1_ker_scaled)
            shared_runtimes[0][-1] += short_scaled
            shared_runtimes[1][-1] += short_scaled

            diff_scaled = abs(app0_ker_scaled - app1_ker_scaled)

            if diff_scaled <= RunOption.EQUALITY_ERROR:
                # both kernels finished at the same time update the total
                # runtime of both kernels to either kernel runtime since they
                # have finished together
                handle_completed_kernel(0)
                handle_completed_kernel(1)
            elif app0_ker_scaled < app1_ker_scaled:
                # app0 kernel will finish before app1 kernel update the total
                # runtime of both kernels to app0 kernel runtime since it
                # finished first

                # compute raw remaining runtime of app1
                remaining_runtimes[1] = diff_scaled * \
                                        self.interference_matrix[1][kidx_pair]

                handle_completed_kernel(0)
            else:
                # app1 kernel will finish before app0 kernel update the total
                # runtime of both kernels to app1 kernel runtime since it
                # finished first

                # compute raw remaining runtime of app0
                remaining_runtimes[0] = diff_scaled * \
                                        self.interference_matrix[0][kidx_pair]

                handle_completed_kernel(1)

            if at_least_one and iter_[0] >= 1 and iter_[1] >= 1:
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

        # complete the rest of the required iterations of
        # remaining app in isolation
        remaining_iter = self.jobs[remaining_app].num_iters - iter_[
            remaining_app]
        isolated_runtime = np.resize(seq_runtimes[remaining_app],
                                     remaining_iter)
        shared_runtimes[remaining_app] += list(isolated_runtime)

        # Get rid of tailing zero
        shared_runtimes = [array[0:-1] if array[-1] == 0 else array
                           for array in shared_runtimes]

        # Build performance instances for full calculation and steady state
        full_perf = Performance()
        full_perf.fill_with_duration(shared_runtimes, seq_runtimes)

        steady_perf = Performance()
        steady_perf.fill_with_slowdown(sld=steady_state_qos,
                                       steady_iter=steady_state_iter)

        return full_perf, steady_perf

    def app_wise_weighted(self):
        # TODO
        pass

    def app_wise_gpusim(self, df):
        # TODO
        pass


class RunOption1D(RunOption):
    def __init__(self, ctx, jobs):
        super(RunOption1D, self).__init__(jobs)
        self.ctx = [ctx, 1 - ctx]
        self.quota = [const.get_cta_from_ctx(const.get_dominant_usage(job),
                                             job_ctx, job)
                      for job, job_ctx in zip(self.job_names, ctx)]

        # 1D config is fixed based on ctx value, calculate now
        # Build config matrix:
        matrix_size = (const.get_num_kernels(self.job_names[1]),
                       const.get_num_kernels(self.job_names[0]))
        configs = [np.zeros(matrix_size), np.zeros(matrix_size)]

        # config for job 0
        for idx, kernel_quota in enumerate(self.quota[0]):
            configs[0][:, idx] = kernel_quota

        # config for job 1
        for idx, kernel_quota in enumerate(self.quota[1]):
            configs[1][idx, :] = kernel_quota

        self.config_matrix = [c.astype(int) for c in configs]

    def kernel_wise_prediction(self, df_intra, model):
        # TODO
        pass

    def kernel_wise_gpusim(self, df_dynamic, df_intra):
        benchmarks = []
        for app in self.job_names:
            if app in const.syn_yaml:
                benchmarks.append([(bench, 1) for bench in const.syn_yaml[app]])
            elif app in const.multi_kernel_app:
                benchmarks.append([(app, kidx)
                                   for kidx in const.multi_kernel_app[app]])
            else:
                benchmarks.append([(app, 1)])

        num_cols = const.get_num_kernels(self.job_names[0])
        num_rows = const.get_num_kernels(self.job_names[1])
        matrix_size = (num_rows, num_cols)

        interference = [np.zeros(matrix_size),
                        np.zeros(matrix_size)]

        kernel_columns = ['1_bench', '1_kidx', '2_bench', '2_kidx']

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                matrix_idx = (row_idx, col_idx)

                bench = [benchmarks[0][col_idx], benchmarks[1][row_idx]]
                kernel_configs = [c[matrix_idx] for c in self.config_matrix]

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
                    list_sld = [df_merge['norm_ipc'].iloc[0] * w for w in
                                weight]
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

                    df_bench.sort_values('distance', inplace=True,
                                         ascending=True)

                    list_sld = df_bench['sld'].iloc[0][1:]

                    # Flip it back
                    if sorted_bench != bench:
                        list_sld.reverse()

                interference[0][matrix_idx] = list_sld[0]
                interference[1][matrix_idx] = list_sld[1]

        self.interference_matrix = interference


class RunOption3D(RunOption):
    def __init__(self, job):
        super(RunOption3D, self).__init__(job)

    def kernel_wise_prediction(self, df_intra, model):
        # TODO
        pass

    def kernel_wise_gpusim(self, df_dynamic, df_intra):
        # Generate both config and interference matrices
        num_cols = const.get_num_kernels(self.job_names[0])
        num_rows = const.get_num_kernels(self.job_names[1])
        matrix_size = (num_rows, num_cols)

        # Calculate total runtime
        seq_cycles = [const.get_seq_cycles(app) for app in self.job_names]
        importance = [[cycles / sum(app_cycles) for cycles in app_cycles]
                      for app_cycles in seq_cycles]

        if num_cols != len(importance[0]) or num_rows != len(importance[1]):
            print('Dimension mismatch between matrix size and importance array')
            sys.exit(1)

        # return list_cta, list_sld, serial?
        def get_cta_sld(bench_idx):
            real_bench = []
            serial = False

            for app, midx in zip(self.job_names, bench_idx):
                if 'syn' in app:
                    # FIXME: Here we assume we don't involve multi-kernel bench
                    real_bench.append((const.syn_yaml[app][midx], 1))
                else:
                    real_bench.append(
                        (app, const.translate_gpusim_kidx(app, midx)))

            if real_bench[0] == real_bench[1]:
                # Identical benchmark pair. Get settings from df_intra
                df_bench = df_intra[(df_intra['pair_str'] == real_bench[0][0]) &
                                    (df_intra['1_kidx'] == real_bench[0][
                                        1])].copy()
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
                    cta_setting = df_bench.loc[idx_max['norm_ipc']][
                                      'intra'] // 2
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
                    cta_setting = [series_best['1_intra'],
                                   series_best['2_intra']]
                    sld = series_best['sld'][1:3]

                    if real_bench != sorted_real_bench:
                        cta_setting.reverse()
                        sld.reverse()

                return cta_setting, sld, serial

        self.config_matrix = [np.zeros(matrix_size, dtype=int),
                              np.zeros(matrix_size, dtype=int)]
        self.interference_matrix = [np.zeros(matrix_size),
                                    np.zeros(matrix_size)]
        self.serial = np.zeros(matrix_size, dtype=int)

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                list_cta, list_sld, pair_serial = get_cta_sld(
                    [col_idx, row_idx])

                if len(list_cta) == 0:
                    print("LUT config does not exist for {}", self.job_names)
                    sys.exit(1)

                matrix_idx = (row_idx, col_idx)

                self.config_matrix[0][matrix_idx] = int(list_cta[0])
                self.config_matrix[1][matrix_idx] = int(list_cta[1])

                self.interference_matrix[0][matrix_idx] = list_sld[0]
                self.interference_matrix[1][matrix_idx] = list_sld[1]

                self.serial[matrix_idx] = int(pair_serial)


class PairJob:
    df_dynamic = const.get_pickle('pair_dynamic.pkl')
    df_intra = const.get_pickle('intra.pkl')
    # FIXME: load model from pickle
    model = None

    def __init__(self, jobs):
        self.jobs = jobs
        self.job_names = [job.name() for job in self.jobs]

    # return RunOption, Performance
    def get_performance(self, num_slices, alloc='1D',
                        stage_1='GPUSIM', stage_2='FULL',
                        at_least_once=False):

        # Create RunOptions for allocation design
        if alloc == '1D':
            # Find all possible 1D allocations
            configs = run_pair.find_ctx_configs(self.job_names, num_slices)

            if len(configs) == 0:
                return None

            ctx = [hi.check_config('1_ctx', c, default=0) for c in configs]
            run_options = [RunOption1D(ctx_value, self.jobs)
                           for ctx_value in ctx]
        elif alloc == '3D':
            run_options = [RunOption3D(self.jobs)]
        else:
            # 2D is unimplemented
            print("2D is unimplemented.")
            sys.exit(1)

        # Run Stage 1 to get kernel-wise matrices
        for option in run_options:
            if stage_1 == 'GPUSIM':
                option.kernel_wise_gpusim(PairJob.df_dynamic, PairJob.df_intra)
            else:
                # 'BOOST_TREE'
                option.kernel_wise_prediction(PairJob.df_intra, PairJob.model)

        # Run Stage 2 to get app-wise matrices
        performance = []
        for option in run_options:
            if stage_2 == 'FULL':
                performance.append(
                    option.app_wise_full_and_steady(at_least_once)[0])
            elif stage_2 == 'STEADY':
                performance.append(
                    option.app_wise_full_and_steady(at_least_once)[1])
            elif stage_2 == 'WEIGHTED':
                performance.append(option.app_wise_weighted())
            else:
                performance.append(option.app_wise_gpusim())

        if len(performance) > 1:
            # only keep the best one
            best_perf = max(performance)
            best_idx = performance.index(best_perf)
            best_option = run_options[best_idx]

            return best_option, best_perf
        else:
            return run_options[0], performance[0]







