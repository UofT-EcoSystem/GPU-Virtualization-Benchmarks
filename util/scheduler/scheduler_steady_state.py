import random
import numpy as np
import scipy
import logging


"""Initializing the logger"""
logger = logging.getLogger("scheduler logger")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)




"""
Complete the iterations of longer app in isolation
"""
def isolated_finish(apps, qos, iter_lim):
    # find out which app finished first
    tot_est_runtimes = [qos[0] * sum(apps[0]) * iter_lim[0], qos[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    final_pred = [0, 0]
    final_pred[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * qos[0]
    final_pred[abs(1 - longer_app)] = qos[abs(1 - longer_app)]
    return final_pred


"""
Helper function to find qos loss.
"""
def find_qos(scaled_runtime, num_iter, isolated_total):
    return sum(scaled_runtime) / (isolated_total * num_iter)


"""
Estimate the QOS loss of apps when co-running using steady state estimate.
"""
def estimate_steady_states(apps, interference):
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
        app0_ker_scaled = rem_runtimes[0] * interference[0][ker[1]][ker[0]]
        # logger.debug("app0 kernel scaled runtime is: {}".format(app0_ker_scaled))
        app1_ker_scaled = rem_runtimes[1] * interference[1][ker[1]][ker[0]]
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
                    curr_qos = find_qos(scaled_runtimes[0], iter_[0], sum(app_lengths[0]))
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
                    curr_qos = find_qos(scaled_runtimes[1], iter_[1], sum(app_lengths[1]))
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
            rem_runtimes[1] = rem_runtimes[1] / interference[1][ker[1]][ker[0]]
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
                    curr_qos = find_qos(scaled_runtimes[0], iter_[0], sum(app_lengths[0]))
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
            rem_runtimes[0] = rem_runtimes[0] / interference[0][ker[1]][ker[0]]
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
                    curr_qos = find_qos(scaled_runtimes[1], iter_[1], sum(app_lengths[1]))
                    if abs(past_qos[1] - curr_qos) < qos_error and steady_state_qos[1] == -1:
                        steady_state_qos[1] = curr_qos
                        steady_state_iter[1] = iter_[1]
                    # update past qos to the current qos
                    past_qos[1] = curr_qos

            # get new remaining runtime for next kernel of app1
            rem_runtimes[1] = apps[1][ker[1]]
    # end of loop
    return steady_state_qos, steady_state_iter


"""
Simulate co-running of apps.
"""
def simulate(apps, interference, iter_lim, offset, offset_app):
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
    # start_offset(apps, rem_runtimes, ker, scaled_runtimes, offsets, offset, offset_app)
    # main loop of the simulation
    while iter_[0] < iter_lim[0] and iter_[1] < iter_lim[1]:
        # figure out who finishes first by scaling the runtimes by the slowdowns
        app0_ker_scaled = rem_runtimes[0] * interference[0][ker[1]][ker[0]]
        # logger.debug("app0 kernel scaled runtime is: {}".format(app0_ker_scaled))
        app1_ker_scaled = rem_runtimes[1] * interference[1][ker[1]][ker[0]]
        # logger.debug("app1 kernel scaled runtime is: {}".format(app1_ker_scaled))
        if abs(app0_ker_scaled - app1_ker_scaled) <= equality_error:
            # both kernels finished at the same time
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

            ker[1] += 1
            # if app1 has finished an iteration
            if ker[1] == len(apps[1]):
                rem_app = 0
                rem_app_index = ker[0]
                ker[1] = 0
                # app1 has completed an iteration of all kernels
                iter_[1] += 1
            # get new remaining runtimes using new indeces for both kernels
            rem_runtimes[0] = apps[0][ker[0]]
            rem_runtimes[1] = apps[1][ker[1]]

        elif app0_ker_scaled < app1_ker_scaled:
            # app0 kernel will finish before app1 kernel
            scaled_runtimes[0][ker[0]] += app0_ker_scaled
            scaled_runtimes[1][ker[1]] += app0_ker_scaled
            # remainder of app1 runtime scaled by slowdown
            rem_runtimes[1] = app1_ker_scaled - app0_ker_scaled
            # compute raw remaining runtime of app1
            rem_runtimes[1] = rem_runtimes[1] / interference[1][ker[1]][ker[0]]
            # move the index to the next kernel for app0
            ker[0] += 1
            if ker[0] == len(apps[0]):
                rem_app = 1
                rem_app_index = ker[1]
                ker[0] = 0
                # app0 has completed an iteration of all kernels
                iter_[0] += 1
            # get new remaining runtime for next kernel of app0
            rem_runtimes[0] = apps[0][ker[0]]
        else:
            # app1 kernel will finish before app0 kernel
            scaled_runtimes[0][ker[0]] += app1_ker_scaled
            scaled_runtimes[1][ker[1]] += app1_ker_scaled
            # remainder of app0 runtime scaled by slowdown
            rem_runtimes[0] = app0_ker_scaled - app1_ker_scaled
            # compute raw remaining runtime of app0
            rem_runtimes[0] = rem_runtimes[0] / interference[0][ker[1]][ker[0]]
            # move the index to the next kernel for app1
            ker[1] += 1
            if ker[1] == len(apps[1]):
                rem_app = 0
                rem_app_index = ker[0]
                ker[1] = 0
                # app1 has completed an iteration of all kernels
                iter_[1] += 1
            # get new remaining runtime for next kernel of app1
            rem_runtimes[1] = apps[1][ker[1]]
    # end of loop

    # at this point, apps stop co-running and the larger app will continue execution in isolation
    co_run_qos = [0, 0]
    co_run_qos[0] = find_qos(scaled_runtimes[0], iter_[0], sum(app_lengths[0]))
    co_run_qos[1] = find_qos(scaled_runtimes[1], iter_[1], sum(app_lengths[1]))

    # finish off this iteration of remaining app in isolation
    scaled_runtimes[rem_app][rem_app_index] += rem_runtimes[rem_app]
    for ind in range(rem_app_index + 1, len(apps[rem_app])):
        scaled_runtimes[rem_app][ind] += apps[rem_app][ind]

    iter_[rem_app] += 1
    rem_iter = iter_lim[rem_app] - iter_[rem_app]

    # complete the rest of the required iterations of remaining app in isolation
    rem_kern = [x * rem_iter for x in apps[rem_app]]
    iter_[rem_app] += rem_iter
    scaled_runtimes[rem_app] = [sum(x) for x in zip(rem_kern, scaled_runtimes[rem_app])]
    logger.debug("Total isolated runtimes of app 0 {}".format([x * iter_[0] for x in apps[0]]))
    logger.debug("Total isolated runtimes of app 1 {}".format([x * iter_[1] for x in apps[1]]))
    qos = [sum(scaled_runtimes[0]) / (sum(apps[0]) * iter_[0]), sum(scaled_runtimes[1]) / (sum(apps[1]) * iter_[1])]
    return qos, co_run_qos


"""
Initializing data structures and parameters for predictions
"""
logger.setLevel(logging.INFO)
equality_error = 0.00001
steady_step = 100
qos_error = 0.0001


logger.info("Positive error = real QOS is better; Negative Error = real QOS is worse")
logger.info("========================================================================")

perc_errors = np.empty([1, 0])

# main loop
for i in range(1000):

    # testing random sizes:
    app0_size = random.randrange(20, 30)
    app1_size = random.randrange(20, 30)
    # logger.info("App0 has {} kernels, App1 has {} kernels".format(app0_size, app1_size))

    # generated random test data
    interference_0 = np.random.rand(app1_size, app0_size)
    interference_1 = np.random.rand(app1_size, app0_size)
    interference_matrix = [interference_0, interference_1]

    # draw random data from normal distribution
    app_lengths = [np.absolute(np.random.normal(8, 2000, size=app0_size)),
                   np.absolute(np.random.normal(8, 2000, size=app1_size))]

    # logger.info("App0 has {} kernels, App1 has {} kernels".format(app0_size, app1_size))

    # # hand-crafted data
    # interference_matrix = [np.array([[1.3, 2.1, 1.5], [1.8, 1.4, 1.7]]), np.array([[1.7, 1.7, 2.5], [1.3, 1.9, 1.1]])]
    # app_lengths = [np.array([4, 3, 8]), np.array([2, 8])]

    # counter for how many iterations each app has completed
    iter_limits = [10000, 10000]

    # compute the QOS losses for both apps
    qos, co_run_qos = simulate(app_lengths, interference_matrix, iter_limits, 0, 0)
    # logger.info(("Actual times are {}".format(qos)))

    # estimate using steady state prediction
    co_run_qos_estimate, detection_iterations = estimate_steady_states(app_lengths, interference_matrix)
    # logger.info("est. co-run qos of apps is: {}".format(co_run_qos_estimate))
    # logger.info("real co-run qos of apps is: {}".format(co_run_qos))
    # estimate the isolated tail of the longer app
    qos_estimate = isolated_finish(app_lengths, co_run_qos_estimate, iter_limits)
    # logger.info(("QOS predictions are {}".format(qos_estimate)))
    # logger.info(("Iterations to reach steady state are {}".format(co_run_qos_estimate[1])))
    if co_run_qos_estimate[0] == -1 or co_run_qos_estimate[1] == -1:
        break
    # logger.info("========================================================================")

    # assign what we want to be displayed as errors
    real_perc = [x * 100 for x in co_run_qos]
    estimate_perc = [x * 100 for x in co_run_qos_estimate]

    # compute errors from co-run qos
    errors = [((real_perc[0] - estimate_perc[0]) / real_perc[0]), (real_perc[1] - estimate_perc[1]) / real_perc[1]]
    # print(errors)
    perc_errors = np.append(perc_errors, [100 * x for x in errors])

print(perc_errors)

logger.info("=============== Steady Prediction ===============")
max_error = max(perc_errors)
max_index = np.where(perc_errors == max_error)
min_error = min(perc_errors)
min_index = np.where(perc_errors == min_error)
logger.info("Max is {} % at index {}".format(max_error, max_index))
logger.info("Min is {} % at index {}".format(min_error, min_index))
logger.info("Average estimate error is {} %".format(np.average(np.absolute(perc_errors))))
logger.info("Standard deviation is {} %".format(np.std(perc_errors)))


# TODO:
#  1. check if the co-run qos estimate is done correctly - it is, we esentially control the error
#  2. expand the simulator to account for when apps can not be co-run with one another
#  you don't need to actually run all the SMs to saturate the input;


#  how to deal with the tail? should we 



