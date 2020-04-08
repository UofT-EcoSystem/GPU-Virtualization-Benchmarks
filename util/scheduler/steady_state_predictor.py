


"""
Helper function to find qos loss.
"""
def find_qos(scaled_runtime, num_iter, isolated_total):
    # logger.debug("{} {} {}".format(sum(scaled_runtime), num_iter, isolated_total))
    return (isolated_total * num_iter) / sum(scaled_runtime)


"""
Complete the iterations of longer app in isolation
"""
def isolated_finish(apps, qos, iter_lim):
    # find out which app we estimate will finished first
    tot_est_runtimes = [qos[0] * sum(apps[0]) * iter_lim[0], qos[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    # find how long rem app was executing in isolation
    # print("Estimate that apps will co-run for ", min(tot_est_runtimes))
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    # print("estimate remaining runtime of longer app is ", rem_app_isol_runtime)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    final_pred = [0, 0]
    final_pred[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * qos[0]
    final_pred[abs(1 - longer_app)] = qos[abs(1 - longer_app)]
    return final_pred


"""
Estimate the QOS loss of apps when co-running using steady state estimate.
"""
def estimate_steady_states(apps, interference, equality_error, steady_step, qos_error):
    # initialize values that will be returend
    steady_state_qos = [-1, -1]
    steady_state_iter = [0, 0]
    # to keep track of iterations completed by apps
    iter_ = [0, 0]
    # scaled_runtimes array will be filled up as we compute
    scaled_runtimes = [[0 for x in apps[0]], [0 for y in apps[1]]]
    # indeces of kernels for two apps - by default 0 and 0
    ker = [0, 0]
    # by default the two kernels launch simultaneously
    rem_runtimes = [apps[0][ker[0]], apps[1][ker[1]]]
    # initializing variables
    rem_app = 0  # app that has remaining kernels after the other app finished
    rem_app_index = 0  # index of current kernel in remaining app
    # variable to keep track of offsets - used to detect completed period
    offsets = []
    # past and total accumulated runtimes of apps
    past_qos = [0, 0]

    # initialize starting offset
    # main loop of the estimator
    while steady_state_qos[0] == -1 or steady_state_qos[1] == -1:
        # figure out who finishes first by scaling the runtimes by the slowdowns
        app0_ker_scaled = rem_runtimes[0] / interference[0][ker[0]][ker[1]]
        # logger.debug("app0 kernel scaled runtime is: {}".format(app0_ker_scaled))
        app1_ker_scaled = rem_runtimes[1] / interference[1][ker[0]][ker[1]]
        # logger.debug("app1 kernel scaled runtime is: {}".format(app1_ker_scaled))
        if abs(app0_ker_scaled - app1_ker_scaled) <= equality_error:
            # both kernels finished at the same time
            # update the total runtime of both kernels to either kernel runtime since they have finished together
            scaled_runtimes[0][ker[0]] += app0_ker_scaled
            scaled_runtimes[1][ker[1]] += app0_ker_scaled
            # advance the index for both kernels
            ker[0] += 1
            # if app0 has finished an iteration
            if ker[0] == len(apps[0]):
                rem_app = 1
                rem_app_index = ker[1]
                ker[0] = 0
                # app0 has completed an iteration of all kernels
                iter_[0] += 1
                # evaluate steady state for app 0
                if iter_[0] % steady_step == 0:
                    # compare qos to the past qos
                    curr_qos = find_qos(scaled_runtimes[0], iter_[0], sum(apps[0]))
                    if abs(past_qos[0] - curr_qos) < qos_error and steady_state_qos[0] == -1:
                        steady_state_qos[0] = curr_qos
                        steady_state_iter[0] = iter_[0]
                    # update past qos to the current qos
                    past_qos[0] = curr_qos

            ker[1] += 1
            # if app1 has finished an iteration
            if ker[1] == len(apps[1]):
                rem_app = 0
                rem_app_index = ker[0]
                ker[1] = 0
                # app1 has completed an iteration of all kernels
                iter_[1] += 1
                # evaluate steady state for app 1
                if iter_[1] % steady_step == 0:
                    # compare qos to the past qos
                    curr_qos = find_qos(scaled_runtimes[1], iter_[1], sum(apps[1]))
                    if abs(past_qos[1] - curr_qos) < qos_error and steady_state_qos[1] == -1:
                        steady_state_qos[1] = curr_qos
                        steady_state_iter[1] = iter_[1]
                    # update past qos to the current qos
                    past_qos[1] = curr_qos
            # get new remaining runtimes using new indeces for both kernels
            rem_runtimes[0] = apps[0][ker[0]]
            rem_runtimes[1] = apps[1][ker[1]]

        elif app0_ker_scaled < app1_ker_scaled:
            # app0 kernel will finish before app1 kernel
            # update the total runtime of both kernels to app0 kernel runtime since it finished first
            scaled_runtimes[0][ker[0]] += app0_ker_scaled
            scaled_runtimes[1][ker[1]] += app0_ker_scaled
            # remainder of app1 runtime scaled by slowdown
            rem_runtimes[1] = app1_ker_scaled - app0_ker_scaled
            # compute raw remaining runtime of app1
            rem_runtimes[1] = rem_runtimes[1] * interference[1][ker[0]][ker[1]]
            # move the index to the next kernel for app0
            ker[0] += 1
            if ker[0] == len(apps[0]):
                rem_app = 1
                rem_app_index = ker[1]
                ker[0] = 0
                # app0 has completed an iteration of all kernels
                iter_[0] += 1
                if iter_[0] % steady_step == 0:
                    # compare qos to the past qos
                    curr_qos = find_qos(scaled_runtimes[0], iter_[0], sum(apps[0]))
                    if abs(past_qos[0] - curr_qos) < qos_error and steady_state_qos[0] == -1:
                        steady_state_qos[0] = curr_qos
                        steady_state_iter[0] = iter_[0]
                    # update past qos to the current qos
                    past_qos[0] = curr_qos

            # get new remaining runtime for next kernel of app0
            rem_runtimes[0] = apps[0][ker[0]]
        else:
            # app1 kernel will finish before app0 kernel
            # update the total runtime of both kernels to app1 kernel runtime since it finished first
            scaled_runtimes[0][ker[0]] += app1_ker_scaled
            scaled_runtimes[1][ker[1]] += app1_ker_scaled
            # remainder of app0 runtime scaled by slowdown
            rem_runtimes[0] = app0_ker_scaled - app1_ker_scaled
            # compute raw remaining runtime of app0
            rem_runtimes[0] = rem_runtimes[0] * interference[0][ker[0]][ker[1]]
            # move the index to the next kernel for app1
            ker[1] += 1
            if ker[1] == len(apps[1]):
                rem_app = 0
                rem_app_index = ker[0]
                ker[1] = 0
                # app1 has completed an iteration of all kernels
                iter_[1] += 1
                if iter_[1] % steady_step == 0:
                    # compare qos to the past qos
                    curr_qos = find_qos(scaled_runtimes[1], iter_[1], sum(apps[1]))
                    if abs(past_qos[1] - curr_qos) < qos_error and steady_state_qos[1] == -1:
                        steady_state_qos[1] = curr_qos
                        steady_state_iter[1] = iter_[1]
                    # update past qos to the current qos
                    past_qos[1] = curr_qos

            # get new remaining runtime for next kernel of app1
            rem_runtimes[1] = apps[1][ker[1]]
    # end of loop
    return steady_state_qos, steady_state_iter
