import random
import numpy as np

from simulator import *


"""
Initializing data structures and parameters for predictions
"""
logger.setLevel(logging.INFO)
equality_error = 0.00001
steady_step = 100
qos_error = 0.001


logger.info("Positive error = real QOS is better; Negative Error = real QOS is worse")
logger.debug("Mixed kernel = kernel that executed both consecutively with other kernel and in isolation")
logger.debug("Mixed iteration = iteration that has mixed kernels")

perc_errors = np.empty([1, 0])

# main loop
for i in range(2):

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
    qos, co_run_qos = simulate(app_lengths, interference_matrix, iter_numbers, equality_error)
    print(qos)
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





