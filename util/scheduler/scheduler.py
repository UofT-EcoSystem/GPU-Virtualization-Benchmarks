import random
import numpy as np
import scipy
import logging
from scipy.stats.mstats import gmean


"""Initializing the logger"""
logger = logging.getLogger("scheduler logger")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


# def start_offset(apps, rem_runtimes, ker, scaled_runtimes, offsets, offset, offset_app):
#     # figure out the starting offset and factor it into runtimes
#     # inputs:
#     other_app = abs(offset_app - 1)
#     offsets.append(offset)
#
#     ker_index = 0
#     ker_len = apps[other_app][0]
#     while offset > ker_len:
#         # go to next kernel and reduce the offset
#         # this kernel of the other app was executing in isolation
#         scaled_runtimes[other_app][ker_index] += apps[other_app][ker_index]
#         offset -= ker_len
#         ker_index += 1
#         ker_len = apps[other_app][ker_index]
#
#     # part of this kernel for other app was excuting in isolation -> add to scaled runtimes
#     scaled_runtimes[other_app][ker_index] += offset
#     rem_runtimes[other_app] = ker_len - offset
#     ker[other_app] = ker_index
#

def weight_geom_mean(weights, data):
    exponent = 1 / sum(weights)
    prod = 1
    for ind in range(weights.size):
        prod = prod * (data[ind] ** weights[ind])

    return prod ** exponent


def weight_harm_mean(weights, data):

    bottom_sum = 0
    for ind in range(weights.size):
        bottom_sum += weights[ind] / data[ind]

    return sum(weights) / bottom_sum


def estimate_weigh_geom_mean(apps, interference, iter_lim) -> list:
    # normalized apps
    norm_apps = [apps[0] / np.sum(apps[0]), apps[1] / np.sum(apps[1])]
    qos_loss = [0, 0]

    # find initial prediction for app0
    # weighted geom mean collapse to a vector (wrt app0)
    vector_app0 = []
    for ind in range(norm_apps[1].size):
        vector_app0.append(weight_geom_mean(norm_apps[0], interference[0][ind]))
    # collapse the vector wrt app1
    qos_loss[0] = weight_geom_mean(norm_apps[1], vector_app0)

    # find initial prediction for app1
    # weighted geom mean collapse to a vector (wrt app0)
    vector_app1 = []
    for ind in range(norm_apps[1].size):
        vector_app1.append(weight_geom_mean(norm_apps[0], interference[1][ind]))
    # collapse the vector wrt app1
    qos_loss[1] = weight_geom_mean(norm_apps[1], vector_app1)

    # find out which app finished first
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0], qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    qos_loss[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * qos_loss[0]
    logger.debug("Predictions are: {}".format(qos_loss))
    final_pred = [1 / qos_loss[0], 1 / qos_loss[1]]
    return final_pred


def estimate_weigh_harm_mean(apps, interference, iter_lim) -> list:
    # normalized apps
    norm_apps = [apps[0] / np.sum(apps[0]), apps[1] / np.sum(apps[1])]
    qos_loss = [0, 0]

    # find initial prediction for app0
    # weighted harm mean collapse to a vector (wrt app0)
    vector_app0 = []
    for ind in range(norm_apps[1].size):
        vector_app0.append(weight_harm_mean(norm_apps[0], interference[0][ind]))
    # collapse the vector wrt app1
    qos_loss[0] = weight_harm_mean(norm_apps[1], vector_app0)

    # find initial prediction for app1
    # weighted harm mean collapse to a vector (wrt app0)
    vector_app1 = []
    for ind in range(norm_apps[1].size):
        vector_app1.append(weight_harm_mean(norm_apps[0], interference[1][ind]))
    # collapse the vector wrt app1
    qos_loss[1] = weight_harm_mean(norm_apps[1], vector_app1)

    # find out which app finished first
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0], qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    qos_loss[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * qos_loss[0]
    logger.debug("Predictions are: {}".format(qos_loss))
    final_pred = [1 / qos_loss[0], 1 / qos_loss[1]]
    return final_pred


def estimate_weigh_arith_mean(apps, interference, iter_lim) -> list:
    # two applications expressed as lengths of kernel runtimes
    norm_apps = [apps[0] / np.sum(apps[0]), apps[1] / np.sum(apps[1])]
    qos_loss = []

    weighted_0_inter = np.average(interference[0], 1, norm_apps[0])
    qos_loss.append(np.average(weighted_0_inter, 0, norm_apps[1]))
    logger.debug("QOS prediction for app0 is {}".format(qos_loss[0]))

    weighted_1_inter = np.average(interference[1], 1, norm_apps[0])
    qos_loss.append(np.average(weighted_1_inter, 0, norm_apps[1]))
    logger.debug("QOS prediction for app1 is {}".format(qos_loss[1]))

    # find out which app finished first
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0], qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    qos_loss[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * qos_loss[0]
    logger.debug("Predictions are: {}".format(qos_loss))
    final_pred = [1 / qos_loss[0], 1 / qos_loss[1]]
    return final_pred


def estimate_qos_steady(apps, qos_loss, iter_lim):
    # find out which app finished first
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0], qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    final_pred = [0, 0]
    final_pred[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * qos_loss[0]
    final_pred[abs(1 - longer_app)] = qos_loss[abs(1 - longer_app)]
    return final_pred


def estimate_steady(apps, qos_loss, iter_lim):
    # find out which app finished first
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0], qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    qos_loss[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * qos_loss[0]
    logger.debug("Predictions are: {}".format(qos_loss))
    final_pred = [1 / qos_loss[0], 1 / qos_loss[1]]
    return final_pred


def find_qos_loss(scaled_runtime, num_iter, isolated_total):
    return sum(scaled_runtime) / (isolated_total * num_iter)


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
    past_qos_loss = [0, 0]

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
                    curr_qos_loss = find_qos_loss(scaled_runtimes[0], iter_[0], sum(app_lengths[0]))
                    if abs(past_qos_loss[0] - curr_qos_loss) < qos_loss_error and steady_state_qos[0] == -1:
                        steady_state_qos[0] = curr_qos_loss
                        steady_state_iter[0] = iter_[0]
                    # update past qos loss to the current qos loss
                    past_qos_loss[0] = curr_qos_loss

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
                    curr_qos_loss = find_qos_loss(scaled_runtimes[1], iter_[1], sum(app_lengths[1]))
                    if abs(past_qos_loss[1] - curr_qos_loss) < qos_loss_error and steady_state_qos[1] == -1:
                        steady_state_qos[1] = curr_qos_loss
                        steady_state_iter[1] = iter_[1]
                    # update past qos loss to the current qos loss
                    past_qos_loss[1] = curr_qos_loss
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
                    curr_qos_loss = find_qos_loss(scaled_runtimes[0], iter_[0], sum(app_lengths[0]))
                    if abs(past_qos_loss[0] - curr_qos_loss) < qos_loss_error and steady_state_qos[0] == -1:
                        steady_state_qos[0] = curr_qos_loss
                        steady_state_iter[0] = iter_[0]
                    # update past qos loss to the current qos loss
                    past_qos_loss[0] = curr_qos_loss

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
                    curr_qos_loss = find_qos_loss(scaled_runtimes[1], iter_[1], sum(app_lengths[1]))
                    if abs(past_qos_loss[1] - curr_qos_loss) < qos_loss_error and steady_state_qos[1] == -1:
                        steady_state_qos[1] = curr_qos_loss
                        steady_state_iter[1] = iter_[1]
                    # update past qos loss to the current qos loss
                    past_qos_loss[1] = curr_qos_loss

            # get new remaining runtime for next kernel of app1
            rem_runtimes[1] = apps[1][ker[1]]
    # end of loop

    # finish off this iteration of remaining app in isolation
    scaled_runtimes[rem_app][rem_app_index] += rem_runtimes[rem_app]
    for ind in range(rem_app_index + 1, len(apps[rem_app])):
        scaled_runtimes[rem_app][ind] += apps[rem_app][ind]

    iter_[rem_app] += 1
    logger.debug("Full completed iterations are {}".format(iter_))
    logger.debug("========================================================================")
    logger.debug("Completing the remaining iterations of app {} in isolation".format(rem_app))
    logger.debug("Scaled_runtimes of app 0 are {}".format(scaled_runtimes[0]))
    logger.debug("Scaled_runtimes of app 1 are {}".format(scaled_runtimes[1]))

    rem_iter = iter_lim[rem_app] - iter_[rem_app]

    # complete the rest of the required iterations of remaining app in isolation
    rem_kern = [x * rem_iter for x in apps[rem_app]]
    iter_[rem_app] += rem_iter
    scaled_runtimes[rem_app] = [sum(x) for x in zip(rem_kern, scaled_runtimes[rem_app])]
    logger.info("=================================== Final Info =====================================")
    logger.debug("Total isolated runtimes of app 0 {}".format([x * iter_[0] for x in apps[0]]))
    logger.debug("Total isolated runtimes of app 1 {}".format([x * iter_[1] for x in apps[1]]))
    qos_loss = [sum(scaled_runtimes[0]) / (sum(apps[0]) * iter_[0]), sum(scaled_runtimes[1]) / (sum(apps[1]) * iter_[1])]
    logger.info("App0 has {} kernels, App1 has {} kernels".format(app0_size, app1_size))
    logger.info("Actual calculated QOS losses are {}".format(qos_loss))
    logger.info(("Steady predicted qos losses are {}".format(estimate_qos_steady(app_lengths, steady_state_qos, iter_lim))))
    logger.info("Steady state for app0 and app1 reached at iterations {} and {} respectively".format(steady_state_iter[0], steady_state_iter[1]))
    final_pred = [1 / qos_loss[0], 1 / qos_loss[1]]
    return final_pred


"""Initializing data structures and parameters"""
logger.setLevel(logging.DEBUG)
equality_error = 0.00001
steady_step = 20
qos_loss_error = 0.01


logger.info("Positive error = real QOS is better; Negative Error = real QOS is worse")
logger.info("========================================================================")

# perform multiple iterations with random data to get the errors
errors_arith_mean = np.empty([1, 0])
errors_geom_mean = np.empty([1, 0])
errors_harm_mean = np.empty([1, 0])
errors_steady = np.empty([1, 0])


for i in range(100):

    # list to keep track of estimated qos using steady state estimate
    steady_state_qos = [-1, -1]
    steady_state_iter = [0, 0]

    # testing random sizes:
    app0_size = random.randrange(1, 30)
    app1_size = random.randrange(1, 30)
    # logger.info("App0 has {} kernels, App1 has {} kernels".format(app0_size, app1_size))

    # generated random test data
    interference_0 = np.random.rand(app1_size, app0_size)
    interference_1 = np.random.rand(app1_size, app0_size)
    interference_matrix = [interference_0 + np.ones_like(interference_0), interference_1 + np.ones_like(interference_1)]

    # draw random data from normal distribution
    app_lengths = [np.absolute(np.random.uniform(8, 2000, size=app0_size)),
                   np.absolute(np.random.uniform(8, 2000, size=app1_size))]

    logger.debug("app_lengths are {}".format(app_lengths))

    # # hand-crafted data
    # interference_matrix = [np.array([[1.3, 2.1, 1.5], [1.8, 1.4, 1.7]]), np.array([[1.7, 1.7, 2.5], [1.3, 1.9, 1.1]])]
    # app_lengths = [np.array([4, 3, 8]), np.array([2, 8])]

    # interference matrix for kernels of two apps

    # counter for how many iterations each app has completed
    iter_limits = [10000, 10000]

    # # get an estimate of the QOS for both kernels
    # predictions_arith_mean = estimate_weigh_arith_mean(app_lengths, interference_matrix, iter_limits)
    # predictions_geom_mean = estimate_weigh_geom_mean(app_lengths, interference_matrix, iter_limits)
    # predictions_harm_mean = estimate_weigh_harm_mean(app_lengths, interference_matrix, iter_limits)

    # simulate and obtain actual QOS for both kernels
    qos = simulate(app_lengths, interference_matrix, iter_limits, 0, 0)
    predictions_steady = estimate_steady(app_lengths, steady_state_qos, iter_limits)

    # compute error for steady_state prediction
    perc_error_steady = [((qos[0] - predictions_steady[0]) / qos[0]), (qos[1] - predictions_steady[1]) / qos[1]]

    # # compute error for this prediction
    # perc_errors_arith = [((qos[0] - predictions_arith_mean[0]) / qos[0]), (qos[1] - predictions_arith_mean[1]) / qos[1]]
    # perc_errors_geom = [((qos[0] - predictions_geom_mean[0]) / qos[0]), (qos[1] - predictions_geom_mean[1]) / qos[1]]
    # perc_errors_harm = [((qos[0] - predictions_harm_mean[0]) / qos[0]), (qos[1] - predictions_harm_mean[1]) / qos[1]]
    #
    # # multiply error values by 100 to look nice
    errors_steady = np.append(errors_steady, [100 * x for x in perc_error_steady])
    # errors_arith_mean = np.append(errors_arith_mean, [100 * x for x in perc_errors_arith])
    # errors_geom_mean = np.append(errors_geom_mean, [100 * x for x in perc_errors_geom])
    # errors_harm_mean = np.append(errors_harm_mean, [100 * x for x in perc_errors_harm])
    # if min(errors_harm_mean) <= -10:
    #     logger.info("=======================================================================")
    #     logger.info("App0 has {} kernels, App1 has {} kernels".format(app0_size, app1_size))
    #     logger.info(interference_matrix)
    #     logger.info(app_lengths)

# # arithmetic mean errors_arith_mean reporting
# logger.info("=============== Arithmetic mean ===============")
# max_arith_error = max(errors_arith_mean)
# max_arith_index = np.where(errors_arith_mean == max_arith_error)
# min_arith_error = min(errors_arith_mean)
# min_arith_index = np.where(errors_arith_mean == min_arith_error)
# logger.info("Max is {} % at index {}".format(max_arith_error, max_arith_index))
# logger.info("Min is {} % at index {}".format(min_arith_error, min_arith_index))
# logger.info("Average estimate error is {} %".format(np.average(np.absolute(errors_arith_mean))))
# logger.info("Standard deviation is {} %".format(np.std(errors_arith_mean)))
#
# logger.info("=============== Geometric mean ===============")
# max_geom_error = max(errors_geom_mean)
# max_geom_index = np.where(errors_geom_mean == max_geom_error)
# min_geom_error = min(errors_geom_mean)
# min_geom_index = np.where(errors_geom_mean == min_geom_error)
# logger.info("Max is {} % at index {}".format(max_geom_error, max_geom_index))
# logger.info("Min is {} % at index {}".format(min_geom_error, min_geom_index))
# logger.info("Average estimate error is {} %".format(np.average(np.absolute(errors_geom_mean))))
# logger.info("Standard deviation is {} %".format(np.std(errors_geom_mean)))
#
# logger.info("=============== Harmonic mean ===============")
# max_harm_error = max(errors_harm_mean)
# max_harm_index = np.where(errors_harm_mean == max_harm_error)
# min_harm_error = min(errors_harm_mean)
# min_harm_index = np.where(errors_harm_mean == min_harm_error)
# logger.info("Max is {} % at index {}".format(max_harm_error, max_harm_index))
# logger.info("Min is {} % at index {}".format(min_harm_error, min_harm_index))
# logger.info("Average estimate error is {} %".format(np.average(np.absolute(errors_harm_mean))))
# logger.info("Standard deviation is {} %".format(np.std(errors_harm_mean)))

logger.info("=============== Steady Prediction ===============")
max_steady_error = max(errors_steady)
max_steady_index = np.where(errors_steady == max_steady_error)
min_steady_error = min(errors_steady)
min_steady_index = np.where(errors_steady == min_steady_error)
logger.info("Max is {} % at index {}".format(max_steady_error, max_steady_index))
logger.info("Min is {} % at index {}".format(min_steady_error, min_steady_index))
logger.info("Average estimate error is {} %".format(np.average(np.absolute(errors_steady))))
logger.info("Standard deviation is {} %".format(np.std(errors_steady)))

# TODO: set up to simulate on real data

# See if averaging predictions from all 3 predictors will produce lower errors (in case max errors are
# occuring on different samples).




















