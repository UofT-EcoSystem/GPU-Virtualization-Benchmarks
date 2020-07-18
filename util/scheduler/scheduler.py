import random
import numpy as np
import scipy
import logging
from scipy.stats.mstats import gmean

import data.scripts.common.help_iso as hi

"""Initializing the logger"""
logger = logging.getLogger("scheduler logger")
logger.propagate = False
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
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0],
                        qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    qos_loss[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * \
                           qos_loss[0]
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
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0],
                        qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    qos_loss[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * \
                           qos_loss[0]
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
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0],
                        qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    qos_loss[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * \
                           qos_loss[0]
    logger.debug("Predictions are: {}".format(qos_loss))
    final_pred = [1 / qos_loss[0], 1 / qos_loss[1]]
    return final_pred


def estimate_qos_steady(apps, qos_loss, iter_lim):
    # find out which app finished first
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0],
                        qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    final_pred = [0, 0]
    final_pred[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * \
                             qos_loss[0]
    final_pred[abs(1 - longer_app)] = qos_loss[abs(1 - longer_app)]
    return final_pred


def estimate_steady(apps, qos_loss, iter_lim):
    # find out which app finished first
    tot_est_runtimes = [qos_loss[0] * sum(apps[0]) * iter_lim[0],
                        qos_loss[1] * sum(apps[1]) * iter_lim[1]]
    longer_app = tot_est_runtimes.index(max(tot_est_runtimes))
    logger.debug("longer app is estimated as app{}".format(longer_app))
    # find how long rem app was executing in isolation
    rem_app_isol_runtime = max(tot_est_runtimes) - min(tot_est_runtimes)
    rem_app_isol_ratio = rem_app_isol_runtime / tot_est_runtimes[longer_app]
    # computed new QOS knowing the ratio of isolated runtime
    qos_loss[longer_app] = rem_app_isol_ratio * 1 + (1 - rem_app_isol_ratio) * \
                           qos_loss[0]
    logger.debug("Predictions are: {}".format(qos_loss))
    final_pred = [1 / qos_loss[0], 1 / qos_loss[1]]
    return final_pred


def find_qos_loss(scaled_runtime, num_iter, isolated_runtime):
    return sum(scaled_runtime) / (sum(isolated_runtime) * num_iter)


# FIXME: handle kernel serialization
def simulate(runtimes, interference, upper_lim, at_least_one=False,
             finish_remaining=True,
             offset=0, offset_app=0):
    # to keep track of iterations completed by apps
    iter_ = [0, 0]
    # scaled_runtimes array is in the form of
    # [[k1 time, k2 time...], [k1 time, k2 time, ...]]
    scaled_runtimes = [[0], [0]]
    # indeces of kernels for two apps - by default 0 and 0
    kidx = [0, 0]
    # by default the two kernels launch simultaneously
    remaining_runtimes = [runtimes[0][kidx[0]], runtimes[1][kidx[1]]]
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

    def handle_completed_kernel(app_idx):
        other_app_idx = (app_idx + 1) % NUM_APP

        kidx[app_idx] += 1

        # if app0 has finished an iteration
        if kidx[app_idx] == len(runtimes[app_idx]):
            # re-assignment of outer scope variables
            nonlocal remaining_app, remaining_app_kidx
            remaining_app = other_app_idx
            remaining_app_kidx = kidx[other_app_idx]

            kidx[app_idx] = 0

            # app0 has completed an iteration of all kernels
            iter_[app_idx] += 1

            # evaluate steady state
            if iter_[app_idx] % steady_step == 0:
                # compare qos to the past qos
                qos_loss = find_qos_loss(scaled_runtimes[app_idx],
                                         iter_[app_idx],
                                         runtimes[app_idx])

                if abs(past_qos_loss[app_idx] - qos_loss) < qos_loss_error \
                        and steady_state_qos[app_idx] == -1:
                    steady_state_qos[app_idx] = qos_loss
                    steady_state_iter[app_idx] = iter_[app_idx]

                # update past qos loss to the current qos loss
                past_qos_loss[app_idx] = qos_loss

        remaining_runtimes[app_idx] = runtimes[app_idx][kidx[app_idx]]

        if iter_[app_idx] < upper_lim[app_idx]:
            scaled_runtimes[app_idx].append(0)

    # main loop of the simulation
    while iter_[0] < upper_lim[0] and iter_[1] < upper_lim[1]:
        # figure out who finishes first by scaling the runtimes by the slowdowns
        kidx_pair = (kidx[1], kidx[0])
        app0_ker_scaled = remaining_runtimes[0] / interference[0][kidx_pair]
        # logger.debug("app0 kernel scaled runtime is: {}".format(
        # app0_ker_scaled))
        app1_ker_scaled = remaining_runtimes[1] / interference[1][kidx_pair]
        # logger.debug("app1 kernel scaled runtime is: {}".format(
        # app1_ker_scaled))

        # Advance scaled runtime by the shorter scaled kernel
        short_scaled = min(app0_ker_scaled, app1_ker_scaled)
        scaled_runtimes[0][-1] += short_scaled
        scaled_runtimes[1][-1] += short_scaled

        diff_scaled = abs(app0_ker_scaled - app1_ker_scaled)

        if diff_scaled <= equality_error:
            # both kernels finished at the same time update the total runtime
            # of both kernels to either kernel runtime since they have
            # finished together
            handle_completed_kernel(0)
            handle_completed_kernel(1)
        elif app0_ker_scaled < app1_ker_scaled:
            # app0 kernel will finish before app1 kernel update the total
            # runtime of both kernels to app0 kernel runtime since it
            # finished first

            # compute raw remaining runtime of app1
            remaining_runtimes[1] = diff_scaled * interference[1][kidx_pair]

            handle_completed_kernel(0)
        else:
            # app1 kernel will finish before app0 kernel update the total
            # runtime of both kernels to app1 kernel runtime since it
            # finished first

            # compute raw remaining runtime of app0
            remaining_runtimes[0] = diff_scaled * interference[0][kidx_pair]

            handle_completed_kernel(1)

        if at_least_one and iter_[0] >= 1 and iter_[1] >= 1:
            break
    # end of loop

    # finish off the last iteration of remaining app in isolation
    scaled_runtimes[remaining_app][-1] += \
        remaining_runtimes[remaining_app]
    for idx in range(remaining_app_kidx + 1, len(runtimes[remaining_app])):
        scaled_runtimes[remaining_app].append(runtimes[remaining_app][idx])

    iter_[remaining_app] += 1

    if finish_remaining:
        # complete the rest of the required iterations of
        # remaining app in isolation
        remaining_iter = upper_lim[remaining_app] - iter_[remaining_app]
        isolated_runtime = np.resize(runtimes[remaining_app], remaining_iter)
        scaled_runtimes[remaining_app] += list(isolated_runtime)

    # Get rid of tailing zero
    scaled_runtimes = [array[0:-1] if array[-1] == 0 else array
                       for array in scaled_runtimes]

    return scaled_runtimes, steady_state_iter, steady_state_qos


def calculate_qos(runtimes, end_stamps, short=False,
                  revert=True, print_info=print):
    # print_info("=" * 35 + " Full Simulation QoS Info " + "=" * 35)

    if short:
        # Only fetch the first iteration of the longer app
        qos_loss = hi.calculate_sld_short(end_stamps, runtimes)
    else:
        iter_0 = len(end_stamps[0]) / len(runtimes[0])
        iter_1 = len(end_stamps[1]) / len(runtimes[1])

        qos_loss = [end_stamps[0][-1] / (iter_0 * sum(runtimes[0])),
                    end_stamps[1][-1] / (iter_1 * sum(runtimes[1]))]

    # print_info(
    #     "App0 has {} kernels, App1 has {} kernels".format(len(runtimes[0]),
    #                                                       len(runtimes[1])))
    print_info("Actual calculated QOS losses are {}".format(qos_loss))

    if revert:
        final_pred = [1 / qos_loss[0], 1 / qos_loss[1]]
    else:
        final_pred = [qos_loss[0], qos_loss[1]]

    return final_pred


def calculate_norm_ipc(runtimes, scaled_runtimes):
    circular_runtimes = \
        [np.resize(runtimes[idx], len(app_time))
         for idx, app_time in enumerate(scaled_runtimes)
         ]

    norm_ipc = np.divide(circular_runtimes, scaled_runtimes)

    return norm_ipc


def print_steady_state(runtimes, iter_lim, steady_state_iter, steady_state_qos):
    if steady_state_iter[0] > 0 and steady_state_iter[1] > 0:
        logger.info(("Steady predicted qos losses are {}".format(
            estimate_qos_steady(runtimes, steady_state_qos, iter_lim))))
        logger.info(
            "Steady state for app0 and app1 reached at iterations "
            "{} and {} respectively".format(steady_state_iter[0],
                                            steady_state_iter[1]))


"""Initializing data structures and parameters"""
logger.setLevel(logging.INFO)
equality_error = 0.00001
steady_step = 20
qos_loss_error = 0.01
NUM_APP = 2


def main():
    logger.info("Positive error = real QOS is better; "
                "Negative Error = real QOS is worse")
    logger.info("=" * 72)

    # perform multiple iterations with random data to get the errors
    errors_arith_mean = np.empty([1, 0])
    errors_geom_mean = np.empty([1, 0])
    errors_harm_mean = np.empty([1, 0])
    errors_steady = np.empty([1, 0])

    for i in range(10):
        # testing random sizes:
        app0_size = random.randrange(1, 30)
        app1_size = random.randrange(1, 30)
        # logger.info("App0 has {} kernels, App1 has {} kernels".format(
        # app0_size, app1_size))

        # generated random test data
        interference_0 = np.random.rand(app1_size, app0_size)
        interference_1 = np.random.rand(app1_size, app0_size)
        interference_matrix = [interference_0 + np.ones_like(interference_0),
                               interference_1 + np.ones_like(interference_1)]

        # draw random data from normal distribution
        app_lengths = [np.absolute(np.random.uniform(8, 2000, size=app0_size)),
                       np.absolute(np.random.uniform(8, 2000, size=app1_size))]

        logger.debug("app_lengths are {}".format(app_lengths))

        # # hand-crafted data interference_matrix = [np.array([[1.3, 2.1,
        # 1.5], [1.8, 1.4, 1.7]]), np.array([[1.7, 1.7, 2.5], [1.3, 1.9,
        # 1.1]])] app_lengths = [np.array([4, 3, 8]), np.array([2, 8])]

        # interference matrix for kernels of two apps

        # counter for how many iterations each app has completed
        iter_limits = [10000, 10000]

        # # get an estimate of the QOS for both kernels
        # predictions_arith_mean = estimate_weigh_arith_mean(app_lengths,
        # interference_matrix, iter_limits) predictions_geom_mean =
        # estimate_weigh_geom_mean(app_lengths, interference_matrix,
        # iter_limits) predictions_harm_mean = estimate_weigh_harm_mean(
        # app_lengths, interference_matrix, iter_limits)

        # simulate and obtain actual QOS for both kernels
        scaled_runtimes, steady_state_iter, steady_state_qos = \
            simulate(app_lengths, interference_matrix, iter_limits)

        qos = calculate_qos(app_lengths, scaled_runtimes,
                            print_info=logger.info)

        predictions_steady = estimate_steady(app_lengths, steady_state_qos,
                                             iter_limits)

        # compute error for steady_state prediction
        perc_error_steady = [((qos[0] - predictions_steady[0]) / qos[0]),
                             (qos[1] - predictions_steady[1]) / qos[1]]

        # # compute error for this prediction perc_errors_arith = [((qos[0] -
        # predictions_arith_mean[0]) / qos[0]), (qos[1] -
        # predictions_arith_mean[1]) / qos[1]] perc_errors_geom = [((qos[0] -
        # predictions_geom_mean[0]) / qos[0]), (qos[1] -
        # predictions_geom_mean[1]) / qos[1]] perc_errors_harm = [((qos[0] -
        # predictions_harm_mean[0]) / qos[0]), (qos[1] -
        # predictions_harm_mean[1]) / qos[1]]
        #
        # # multiply error values by 100 to look nice
        errors_steady = np.append(errors_steady,
                                  [100 * x for x in perc_error_steady])
        # errors_arith_mean = np.append(errors_arith_mean, [100 * x for x in
        # perc_errors_arith]) errors_geom_mean = np.append(errors_geom_mean,
        # [100 * x for x in perc_errors_geom]) errors_harm_mean = np.append(
        # errors_harm_mean, [100 * x for x in perc_errors_harm]) if min(
        # errors_harm_mean) <= -10: logger.info(
        # "==============================================================
        # =========") logger.info("App0 has {} kernels, App1
        # has {} kernels".format(app0_size, app1_size))
        # logger.info(interference_matrix) logger.info(app_lengths)

    # # arithmetic mean errors_arith_mean reporting logger.info(
    # "=============== Arithmetic mean ===============") max_arith_error =
    # max(errors_arith_mean) max_arith_index = np.where(errors_arith_mean ==
    # max_arith_error) min_arith_error = min(errors_arith_mean)
    # min_arith_index = np.where(errors_arith_mean == min_arith_error)
    # logger.info("Max is {} % at index {}".format(max_arith_error,
    # max_arith_index)) logger.info("Min is {} % at index {}".format(
    # min_arith_error, min_arith_index)) logger.info("Average estimate error
    # is {} %".format(np.average(np.absolute(errors_arith_mean))))
    # logger.info("Standard deviation is {} %".format(np.std(
    # errors_arith_mean)))
    #
    # logger.info("=============== Geometric mean ===============")
    # max_geom_error = max(errors_geom_mean)
    # max_geom_index = np.where(errors_geom_mean == max_geom_error)
    # min_geom_error = min(errors_geom_mean)
    # min_geom_index = np.where(errors_geom_mean == min_geom_error)
    # logger.info("Max is {} % at index {}".format(max_geom_error, max_geom_index))
    # logger.info("Min is {} % at index {}".format(min_geom_error, min_geom_index))
    # logger.info("Average estimate error is {} %".format(np.average(
    # np.absolute(errors_geom_mean))))
    # logger.info("Standard deviation is {} %".format(np.std(errors_geom_mean)))
    #
    # logger.info("=============== Harmonic mean ===============")
    # max_harm_error = max(errors_harm_mean)
    # max_harm_index = np.where(errors_harm_mean == max_harm_error)
    # min_harm_error = min(errors_harm_mean)
    # min_harm_index = np.where(errors_harm_mean == min_harm_error)
    # logger.info("Max is {} % at index {}".format(max_harm_error, max_harm_index))
    # logger.info("Min is {} % at index {}".format(min_harm_error, min_harm_index))
    # logger.info("Average estimate error is {} %".format(np.average(
    # np.absolute(errors_harm_mean))))
    # logger.info("Standard deviation is {} %".format(np.std(errors_harm_mean)))

    logger.info("=============== Steady Prediction ===============")
    max_steady_error = max(errors_steady)
    max_steady_index = np.where(errors_steady == max_steady_error)
    min_steady_error = min(errors_steady)
    min_steady_index = np.where(errors_steady == min_steady_error)
    logger.info(
        "Max is {} % at index {}".format(max_steady_error, max_steady_index))
    logger.info(
        "Min is {} % at index {}".format(min_steady_error, min_steady_index))
    logger.info("Average estimate error is {} %".format(
        np.average(np.absolute(errors_steady))))
    logger.info("Standard deviation is {} %".format(np.std(errors_steady)))

    # TODO: set up to simulate on real data

    # See if averaging predictions from all 3 predictors will produce lower
    # errors (in case max errors are occuring on different samples).


if __name__ == '__main__':
    main()
