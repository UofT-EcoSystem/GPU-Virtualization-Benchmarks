import random
import numpy as np
import pickle
from multiprocessing import Process, Queue, current_process, freeze_support

from simulator import *
# from steady_state_predictor import *


"""
Initializing data structures and parameters for predictions
"""
logger.setLevel(logging.INFO)
equality_error = 0.00001
steady_step = 100
qos_error = 0.001
test_type = "pickled_multi"

# counter for how many iterations each app is to complete
iter_numbers = [10000, 10000]

logger.info("Positive error = real QOS is better; Negative Error = real QOS is worse")

perc_errors = np.empty([1, 0])

if test_type == "random_single":
    for i in range(10):
        # testing random sizes:
        app0_size = random.randrange(1, 30)
        app1_size = random.randrange(1, 30)

        # generated random test data
        interference_0 = np.around(0.01 + np.random.rand(app0_size, app1_size), decimals=2)
        interference_1 = np.around(0.01 + np.random.rand(app0_size, app1_size), decimals=2)
        interference_matrix = [interference_0, interference_1]
        # print(interference_matrix)

        # draw random data from normal distribution
        app_lengths = [np.around(np.absolute(np.random.uniform(8, 2000, size=app0_size)), decimals=2),
                       np.around(np.absolute(np.random.uniform(8, 2000, size=app1_size)), decimals=2)]


        # compute the QOS losses for both apps
        qos, co_run_qos = simulate(app_lengths, interference_matrix, iter_numbers, equality_error)
        print(qos)
elif test_type == "pickled":
    # load pickled data from file:
    apps = pickle.load(open("10_apps.bin", "rb"))

    for app_pair, interference_pair in apps:

        # compute the QOS losses for both apps
        qos, co_run_qos = simulate(app_pair, interference_pair, iter_numbers, equality_error)
        print(qos)

elif test_type == "pickled_multi":

    # load pickled data from file:
    apps = pickle.load(open("10_apps.bin", "rb"))

    # app_lengths = [x[0] for x in apps]
    # print(app_lengths)

    # for i in range(10):
    #     print(apps)

    # Function run by worker processes
    def worker(input_queue, output_queue):
        for func, args in iter(input_queue.get, 'STOP'):
            result = calculate(func, args)
            output_queue.put(result)

    # Function used to calculate result
    def calculate(func, args):
        result = func(*args)
        print("process {} analyzed app {}".format(current_process().name, result[0]))
        return result

    # Functions referenced by tasks
    def sim(ind, app_pair, interference_pair):
        return ind, simulate(app_pair, interference_pair, iter_numbers, equality_error)[0]

    total_list = []

    NUMBER_OF_PROCESSES = 10
    TASKS = [(sim, (i, apps[i][0], apps[i][1])) for i in range(10)]

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for task in TASKS:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    for i in range(len(TASKS)):
        idx, lst = done_queue.get()
        print(lst)

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')














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





#
# #
# # Function run by worker processes
# #
#
# def worker(input, output):
#     for func, args in iter(input.get, 'STOP'):
#         result = calculate(func, args)
#         output.put(result)
#
# #
# # Function used to calculate result
# #
#
# def calculate(func, args):
#     result = func(*args)
#     # print(result)
#     return result
#
# #
# # Functions referenced by tasks
# #
#
# def mul(lst, constant):
#     return [x * constant for x in lst]
#
# #
# #
# #
#
# def test():
#     total_list = []
#     NUMBER_OF_PROCESSES = 4
#     TASKS = [(mul, ([0, 1, 2, 3], 1)),
#              (mul, ([4, 5, 6, 7], 2)),
#              (mul, ([8, 9, 10, 11], 3)),
#              (mul, ([12, 13, 14, 15], 4))]
#
#     # Create queues
#     task_queue = Queue()
#     done_queue = Queue()
#
#     # Submit tasks
#     for task in TASKS:
#         task_queue.put(task)
#
#     # Start worker processes
#     for i in range(NUMBER_OF_PROCESSES):
#         Process(target=worker, args=(task_queue, done_queue)).start()
#
#     # Get and print results
#     for i in range(len(TASKS)):
#         total_list += done_queue.get()
#
#     # Tell child processes to stop
#     for i in range(NUMBER_OF_PROCESSES):
#         task_queue.put('STOP')
#
#     print(total_list)
#
#
# if __name__ == '__main__':
#     freeze_support()
#     test()
