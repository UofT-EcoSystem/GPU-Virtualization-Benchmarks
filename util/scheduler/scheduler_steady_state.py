import random
import numpy as np
import scipy
import logging


"""Initializing the logger"""
logger = logging.getLogger("scheduler logger")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)




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
Helper function to find qos loss.
"""
def find_qos(scaled_runtime, num_iter, isolated_total):
    # logger.debug("{} {} {}".format(sum(scaled_runtime), num_iter, isolated_total))
    return (isolated_total * num_iter) / sum(scaled_runtime)


# """
# Estimate the QOS loss of apps when co-running using steady state estimate.
# """
# def estimate_steady_states(apps, interference):
#     # initialize values that will be returend
#     steady_state_qos = [-1, -1]
#     steady_state_iter = [0, 0]
#     # to keep track of iterations completed by apps
#     iter_ = [0, 0]
#     # scaled_runtimes array will be filled up as we compute
#     scaled_runtimes = [[0 for x in apps[0]], [0 for y in apps[1]]]
#     # indeces of kernels for two apps - by default 0 and 0
#     ker = [0, 0]
#     # by default the two kernels launch simultaneously
#     rem_runtimes = [apps[0][ker[0]], apps[1][ker[1]]]
#     # initializing variables
#     rem_app = 0  # app that has remaining kernels after the other app finished
#     rem_app_index = 0  # index of current kernel in remaining app
#     # variable to keep track of offsets - used to detect completed period
#     offsets = []
#     # past and total accumulated runtimes of apps
#     past_qos = [0, 0]
#
#     # initialize starting offset
#     # main loop of the estimator
#     while steady_state_qos[0] == -1 or steady_state_qos[1] == -1:
#         # figure out who finishes first by scaling the runtimes by the slowdowns
#         app0_ker_scaled = rem_runtimes[0] / interference[0][ker[0]][ker[1]]
#         # logger.debug("app0 kernel scaled runtime is: {}".format(app0_ker_scaled))
#         app1_ker_scaled = rem_runtimes[1] / interference[1][ker[0]][ker[1]]
#         # logger.debug("app1 kernel scaled runtime is: {}".format(app1_ker_scaled))
#         if abs(app0_ker_scaled - app1_ker_scaled) <= equality_error:
#             # both kernels finished at the same time
#             # update the total runtime of both kernels to either kernel runtime since they have finished together
#             scaled_runtimes[0][ker[0]] += app0_ker_scaled
#             scaled_runtimes[1][ker[1]] += app0_ker_scaled
#             # advance the index for both kernels
#             ker[0] += 1
#             # if app0 has finished an iteration
#             if ker[0] == len(apps[0]):
#                 rem_app = 1
#                 rem_app_index = ker[1]
#                 ker[0] = 0
#                 # app0 has completed an iteration of all kernels
#                 iter_[0] += 1
#                 # evaluate steady state for app 0
#                 if iter_[0] % steady_step == 0:
#                     # compare qos to the past qos
#                     curr_qos = find_qos(scaled_runtimes[0], iter_[0], sum(app_lengths[0]))
#                     if abs(past_qos[0] - curr_qos) < qos_error and steady_state_qos[0] == -1:
#                         steady_state_qos[0] = curr_qos
#                         steady_state_iter[0] = iter_[0]
#                     # update past qos to the current qos
#                     past_qos[0] = curr_qos
#
#             ker[1] += 1
#             # if app1 has finished an iteration
#             if ker[1] == len(apps[1]):
#                 rem_app = 0
#                 rem_app_index = ker[0]
#                 ker[1] = 0
#                 # app1 has completed an iteration of all kernels
#                 iter_[1] += 1
#                 # evaluate steady state for app 1
#                 if iter_[1] % steady_step == 0:
#                     # compare qos to the past qos
#                     curr_qos = find_qos(scaled_runtimes[1], iter_[1], sum(app_lengths[1]))
#                     if abs(past_qos[1] - curr_qos) < qos_error and steady_state_qos[1] == -1:
#                         steady_state_qos[1] = curr_qos
#                         steady_state_iter[1] = iter_[1]
#                     # update past qos to the current qos
#                     past_qos[1] = curr_qos
#             # get new remaining runtimes using new indeces for both kernels
#             rem_runtimes[0] = apps[0][ker[0]]
#             rem_runtimes[1] = apps[1][ker[1]]
#
#         elif app0_ker_scaled < app1_ker_scaled:
#             # app0 kernel will finish before app1 kernel
#             # update the total runtime of both kernels to app0 kernel runtime since it finished first
#             scaled_runtimes[0][ker[0]] += app0_ker_scaled
#             scaled_runtimes[1][ker[1]] += app0_ker_scaled
#             # remainder of app1 runtime scaled by slowdown
#             rem_runtimes[1] = app1_ker_scaled - app0_ker_scaled
#             # compute raw remaining runtime of app1
#             rem_runtimes[1] = rem_runtimes[1] * interference[1][ker[0]][ker[1]]
#             # move the index to the next kernel for app0
#             ker[0] += 1
#             if ker[0] == len(apps[0]):
#                 rem_app = 1
#                 rem_app_index = ker[1]
#                 ker[0] = 0
#                 # app0 has completed an iteration of all kernels
#                 iter_[0] += 1
#                 if iter_[0] % steady_step == 0:
#                     # compare qos to the past qos
#                     curr_qos = find_qos(scaled_runtimes[0], iter_[0], sum(app_lengths[0]))
#                     if abs(past_qos[0] - curr_qos) < qos_error and steady_state_qos[0] == -1:
#                         steady_state_qos[0] = curr_qos
#                         steady_state_iter[0] = iter_[0]
#                     # update past qos to the current qos
#                     past_qos[0] = curr_qos
#
#             # get new remaining runtime for next kernel of app0
#             rem_runtimes[0] = apps[0][ker[0]]
#         else:
#             # app1 kernel will finish before app0 kernel
#             # update the total runtime of both kernels to app1 kernel runtime since it finished first
#             scaled_runtimes[0][ker[0]] += app1_ker_scaled
#             scaled_runtimes[1][ker[1]] += app1_ker_scaled
#             # remainder of app0 runtime scaled by slowdown
#             rem_runtimes[0] = app0_ker_scaled - app1_ker_scaled
#             # compute raw remaining runtime of app0
#             rem_runtimes[0] = rem_runtimes[0] * interference[0][ker[0]][ker[1]]
#             # move the index to the next kernel for app1
#             ker[1] += 1
#             if ker[1] == len(apps[1]):
#                 rem_app = 0
#                 rem_app_index = ker[0]
#                 ker[1] = 0
#                 # app1 has completed an iteration of all kernels
#                 iter_[1] += 1
#                 if iter_[1] % steady_step == 0:
#                     # compare qos to the past qos
#                     curr_qos = find_qos(scaled_runtimes[1], iter_[1], sum(app_lengths[1]))
#                     if abs(past_qos[1] - curr_qos) < qos_error and steady_state_qos[1] == -1:
#                         steady_state_qos[1] = curr_qos
#                         steady_state_iter[1] = iter_[1]
#                     # update past qos to the current qos
#                     past_qos[1] = curr_qos
#
#             # get new remaining runtime for next kernel of app1
#             rem_runtimes[1] = apps[1][ker[1]]
#     # end of loop
#     return steady_state_qos, steady_state_iter

"""
Simulate and visuluze co-running of apps.
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
    logger.debug("app 0 is: {}".format(apps[0]))
    logger.debug("interference matrix of app 0 is:")
    logger.debug(interference_matrix[0])
    logger.debug("app 1 is: {}".format(apps[1]))
    logger.debug("interference matrix of app 1 is:")
    logger.debug(interference_matrix[1])

    # main loop of the simulation
    logger.debug("=============== Starting Simulation ================")
    while iter_[0] < iter_lim[0] and iter_[1] < iter_lim[1]:
        # figure out who finishes first by scaling the runtimes by the slowdowns
        app0_ker_scaled = round(rem_runtimes[0] / interference[0][ker[0]][ker[1]], 3)
        app1_ker_scaled = round(rem_runtimes[1] / interference[1][ker[0]][ker[1]], 3)
        logger.debug("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if abs(app0_ker_scaled - app1_ker_scaled) <= equality_error:
            # both kernels finished at the same time
            # logger.debug info about app 0
            logger.debug("iter {}  ker {}  |--- {} / {} = {} ---|".format(
                iter_[0], ker[0], rem_runtimes[0], interference[0][ker[0]][ker[1]], app0_ker_scaled,
            ))
            # logger.debug info about app 1
            logger.debug("iter {}  ker {}  |--- {} / {} = {} ---|".format(
                iter_[1], ker[1], rem_runtimes[1], interference[1][ker[0]][ker[1]], app1_ker_scaled,
            ))
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
            scaled_rem_time_longer_ker = round(app1_ker_scaled - app0_ker_scaled, 3)
            raw_rem_time_longer_ker = round(scaled_rem_time_longer_ker * interference[1][ker[0]][ker[1]], 3)
            # logger.debug info about app 0
            logger.debug("iter {}  ker {}  |--- {} / {} = {} ---|".format(
                iter_[0], ker[0], rem_runtimes[0], interference[0][ker[0]][ker[1]], app0_ker_scaled
            ))
            # logger.debug info about app 1
            logger.debug("iter {}  ker {}  |--- {} / {} = {} ---|  ->  |--- {} * {} = {} ---|".format(
                iter_[1], ker[1], rem_runtimes[1], interference[1][ker[0]][ker[1]], app1_ker_scaled,
                scaled_rem_time_longer_ker, interference[1][ker[0]][ker[1]], raw_rem_time_longer_ker
            ))
            # compute raw remaining runtime of app1
            rem_runtimes[1] = raw_rem_time_longer_ker
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
            # app1 kernel will finish before app0 kernel so we add shorter time to both
            scaled_runtimes[0][ker[0]] += app1_ker_scaled
            scaled_runtimes[1][ker[1]] += app1_ker_scaled
            # remainder of app0 runtime scaled by slowdown
            scaled_rem_time_longer_ker = round(app0_ker_scaled - app1_ker_scaled, 3)
            # compute raw remaining runtime of app0
            raw_rem_time_longer_ker = round(scaled_rem_time_longer_ker * interference[0][ker[0]][ker[1]], 3)
            # logger.debug info about app 0
            logger.debug("iter {}  ker {}  |--- {} / {} = {} ---|  ->  |--- {} * {} = {} ---|".format(
                iter_[0], ker[0], rem_runtimes[0], interference[0][ker[0]][ker[1]], app0_ker_scaled,
                scaled_rem_time_longer_ker, interference[0][ker[0]][ker[1]], raw_rem_time_longer_ker
            ))
            # logger.debug info about app 1
            logger.debug("iter {}  ker {}  |--- {} / {} = {} ---|".format(
                iter_[1], ker[1], rem_runtimes[1], interference[1][ker[0]][ker[1]], app1_ker_scaled,
            ))
            rem_runtimes[0] = raw_rem_time_longer_ker
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
        logger.debug("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # logger.debug("Scaled Runtimes: {}".format(scaled_runtimes))
    # end of loop
    logger.debug(" ")
    logger.debug("app 0 = {}".format(apps[0]))
    logger.debug("app 1 = {}".format(apps[1]))
    logger.debug("======================== finished co-run execution =========================")
    logger.debug("scaled runtimes are {}".format(scaled_runtimes))
    logger.debug(" ")

    # if one app finished before another
    if iter_[0] != iter_lim[0] or iter_[1] != iter_lim[1]:
        logger.info("================ completing isolated iteration {} of app {} ===================".format(
            iter_[rem_app], rem_app))
        logger.debug("REMAINING APP = app {} : ker index {} out of {} : on iter index {} out of {}".format(
             rem_app, ker[rem_app], len(apps[rem_app]) - 1, iter_[rem_app], iter_lim[rem_app] - 1
        ))
        # finish off this iteration of remaining app in isolation
        logger.debug("adding remaining isolated tail {} of mixed ker {}".format(rem_runtimes[rem_app], ker[rem_app]))
        scaled_runtimes[rem_app][ker[rem_app]] += rem_runtimes[rem_app]
        logger.debug("scaled runtimes are {}".format(scaled_runtimes))

        ker[rem_app] += 1
        if ker[rem_app] < len(apps[rem_app]):
            logger.debug(" ")
            logger.debug("============= app {} completing mixed iter {} in isolation from ker {} =============".format(
                rem_app, iter_[rem_app], ker[rem_app]))

            for ind in range(ker[rem_app], len(apps[rem_app])):
                logger.debug("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                logger.debug("adding isolated ker {} with runtime {}".format(ind, apps[rem_app][ind]))
                scaled_runtimes[rem_app][ind] += apps[rem_app][ind]
                logger.debug("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        logger.debug("============== remaining app {} completed mixed iteration {} ================".format(
            rem_app, iter_[rem_app]))
        iter_[rem_app] += 1

        # at this point we can calculate the co-run qos of the two apps
        co_run_qos_result = [0, 0]
        co_run_qos_result[0] = find_qos(scaled_runtimes[0], iter_[0], sum(apps[0]))
        co_run_qos_result[1] = find_qos(scaled_runtimes[1], iter_[1], sum(apps[1]))
        logger.debug("co-run QOS of apps is {}".format(co_run_qos_result))

        # if remaining app has any more iterations to complete, they will run in isolation
        if iter_[rem_app] != iter_lim[rem_app]:
            rem_iter = iter_lim[rem_app] - iter_[rem_app]
            logger.debug("============== running remaining {} isolated iterations of app {} ===============".format(
                rem_iter, rem_app
            ))
            # find the number of iterations left for the remaining app
            # complete the rest of the required iterations of remaining app in isolation
            logger.debug("scaled runtimes before isolated iterations are {}".format(scaled_runtimes))
            rem_kern = [x * rem_iter for x in apps[rem_app]]
            logger.debug("remaining isolated kernels are {}".format(rem_kern))
            iter_[rem_app] += rem_iter
            scaled_runtimes[rem_app] = [sum(x) for x in zip(rem_kern, scaled_runtimes[rem_app])]

        logger.debug("scaled runtimes after isolated iterations are {}".format(scaled_runtimes))

        # calculate resulting qos
        qos_result = [sum(scaled_runtimes[0]) / (sum(apps[0]) * iter_[0]), sum(scaled_runtimes[1]) / (sum(apps[1]) * iter_[1])]
    else:
        logger.debug("both apps completed their iterations simultaneously")
        # if both apps finished at the same time, then qos can be calculated immediately
        qos_result = [sum(scaled_runtimes[0]) / (sum(apps[0]) * iter_[0]), sum(scaled_runtimes[1]) / (sum(apps[1]) * iter_[1])]
        co_run_qos_result = qos_result

    logger.debug("==================== final QOS ========================")
    logger.debug("QOS of apps is {}".format(qos_result))
    return qos_result, co_run_qos_result


"""
Initializing data structures and parameters for predictions
"""
logger.setLevel(logging.DEBUG)
equality_error = 0.00001
steady_step = 100
qos_error = 0.001


logger.info("Positive error = real QOS is better; Negative Error = real QOS is worse")
logger.debug("Mixed kernel = kernel that executed both consecutively with other kernel and in isolation")
logger.debug("Mixed iteration = iteration that has mixed kernels")
logger.info("========================================================================")

perc_errors = np.empty([1, 0])

# main loop
for i in range(1):

    print("=========================================================")
    # testing random sizes:
    app0_size = random.randrange(1, 5)
    app1_size = random.randrange(1, 5)

    # generated random test data
    interference_0 = np.around(0.01 + np.random.rand(app0_size, app1_size), decimals=2)
    interference_1 = np.around(0.01 + np.random.rand(app0_size, app1_size), decimals=2)
    interference_matrix = [interference_0, interference_1]
    # print(interference_matrix)

    # draw random data from normal distribution
    app_lengths = [np.around(np.absolute(np.random.uniform(8, 2000, size=app0_size)), decimals=2),
                   np.around(np.absolute(np.random.uniform(8, 2000, size=app1_size)), decimals=2)]

    # logger.info("App0 has {} kernels, App1 has {} kernels".format(app0_size, app1_size))

    # # hand-crafted data
    # interference_matrix = [np.array([[0.3, 0.4, 0.2], [0.35556, 0.143, 0.59]]),
    #                        np.array([[0.652, 0.821, 0.469], [0.295, 0.5532, 0.972]])]
    # app_lengths = [np.array([40, 50, 8]), np.array([2, 3])]

    # print("matrix averages are: {} and {}".format(np.average(interference_matrix[0]), np.average(interference_matrix[1])))
    # print("app length sums are: {} and {}".format(sum(app_lengths[0]), sum(app_lengths[1])))
    # print("sanity check avg is: {} and {}".format((1 / np.average(interference_matrix[0])) * sum(app_lengths[0]),
    #                                               (1 / np.average(interference_matrix[1])) * sum(app_lengths[1])))

    # counter for how many iterations each app has completed
    iter_numbers = [10, 10]

    # compute the QOS losses for both apps
    qos, co_run_qos = simulate(app_lengths, interference_matrix, iter_numbers, 0, 0)
    # logger.info(("Actual times are {}".format(qos)))

#     # estimate using steady state prediction
#     co_run_qos_estimate, detection_iterations = estimate_steady_states(app_lengths, interference_matrix)
#     # logger.info("est. co-run qos of apps is: {}".format(co_run_qos_estimate))
#     # logger.info("real co-run qos of apps is: {}".format(co_run_qos))
#     # estimate the isolated tail of the longer app
#     qos_estimate = isolated_finish(app_lengths, co_run_qos_estimate, iter_limits)
#     # logger.info(("QOS predictions are {}".format(qos_estimate)))
#     # logger.info(("Iterations to reach steady state are {}".format(co_run_qos_estimate[1])))
#     if co_run_qos_estimate[0] == -1 or co_run_qos_estimate[1] == -1:
#         break
#     # logger.info("========================================================================")
#
#     # assign what we want to be displayed as errors
#     real_perc = [x * 100 for x in co_run_qos]
#     estimate_perc = [x * 100 for x in co_run_qos_estimate]
#
#     # compute errors from co-run qos
#     errors = [((real_perc[0] - estimate_perc[0]) / real_perc[0]), (real_perc[1] - estimate_perc[1]) / real_perc[1]]
#     # print(errors)
#     perc_errors = np.append(perc_errors, [100 * x for x in errors])
#     print("=========================================================")
#
# # print(perc_errors)
#
# logger.info("=============== Steady Prediction ===============")
# max_error = max(perc_errors)
# max_index = np.where(perc_errors == max_error)
# min_error = min(perc_errors)
# min_index = np.where(perc_errors == min_error)
# logger.info("Max is {} % at index {}".format(max_error, max_index))
# logger.info("Min is {} % at index {}".format(min_error, min_index))
# logger.info("Average estimate error is {} %".format(np.average(np.absolute(perc_errors))))
# logger.info("Standard deviation is {} %".format(np.std(perc_errors)))


# TODO:
#  1. check if the co-run qos estimate is done correctly - it is, we esentially control the error
#  2. expand the simulator to account for when apps can not be co-run with one another
#  you don't need to actually run all the SMs to saturate the input;
#  how to deal with the tail?





