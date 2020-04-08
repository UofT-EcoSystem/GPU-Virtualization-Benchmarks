import logging


"""Initializing the logger"""
logger = logging.getLogger("scheduler logger")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


"""
Helper function to find qos loss.
"""
def find_qos(scaled_runtime, num_iter, isolated_total):
    return (isolated_total * num_iter) / sum(scaled_runtime)


"""
Simulate and visuluze co-running of apps.
"""
def simulate(apps, interference, iter_lim, equality_error):
    # to keep track of iterations completed by apps
    iter_ = [0, 0]
    # scaled_runtimes array will be filled up as we compute
    scaled_runtimes = [[0*x for x in apps[0]], [0*y for y in apps[1]]]
    # indeces of kernels for two apps - by default 0 and 0
    ker = [0, 0]
    # by default the two kernels launch simultaneously
    rem_runtimes = [apps[0][ker[0]], apps[1][ker[1]]]
    # initializing variables
    rem_app = 0  # app that has remaining kernels after the other app finished
    logger.debug("app 0 size: {}".format(len(apps[0])))
    logger.debug("app 0 is: {}".format(apps[0]))
    logger.debug("interference matrix of app 0 is:")
    logger.debug(interference[0])
    logger.debug("app 1 size: {}".format(len(apps[1])))
    logger.debug("app 1 is: {}".format(apps[1]))
    logger.debug("interference matrix of app 1 is:")
    logger.debug(interference[1])

    # main loop of the simulation
    logger.debug("================================== SIMULTANEOUS EXECUTION ===================================")
    logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    while iter_[0] < iter_lim[0] and iter_[1] < iter_lim[1]:
        # figure out who finishes first by scaling the runtimes by the slowdowns
        app0_ker_scaled = round(rem_runtimes[0] / interference[0][ker[0]][ker[1]], 3)
        app1_ker_scaled = round(rem_runtimes[1] / interference[1][ker[0]][ker[1]], 3)
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
                ker[0] = 0
                # app0 has completed an iteration of all kernels
                iter_[0] += 1

            ker[1] += 1
            # if app1 has finished an iteration
            if ker[1] == len(apps[1]):
                rem_app = 0
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
                ker[1] = 0
                # app1 has completed an iteration of all kernels
                iter_[1] += 1
            # get new remaining runtime for next kernel of app1
            rem_runtimes[1] = apps[1][ker[1]]
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # logger.debug("Scaled Runtimes: {}".format(scaled_runtimes))
    # end of loop
    scaled_runtimes = [[round(x, 3) for x in scaled_runtimes[0]],
                       [round(x, 3) for x in scaled_runtimes[1]]]
    logger.debug("============================== FINISHED SIMULTANEOUS EXECUTION ==============================")
    logger.debug("")

    # if one app finished before another
    if iter_[0] != iter_lim[0] or iter_[1] != iter_lim[1]:
        # logger.debug("app 0 size: {}".format(len(apps[0])))
        logger.debug("app 0 = {}, size = {}".format(apps[0], len(apps[0])))
        # logger.debug("app 1 size: {}".format(len(apps[1])))
        logger.debug("app 1 = {}, size = {}".format(apps[1], len(apps[1])))
        logger.debug(("remaining app: {}".format(rem_app)))
        logger.debug(("remaining app on iteration: {}".format(iter_[rem_app])))
        logger.debug("remaining app on mixed kernel: {}".format(ker[rem_app]))
        logger.debug("scaled runtime app 0: {}".format(scaled_runtimes[0]))
        logger.debug("scaled runtime app 1: {}".format(scaled_runtimes[1]))
        logger.debug("")

        logger.debug("===================== EXECUTION OF ISOLATED PORTION OF MIXED KERNEL =========================")
        logger.debug("scaled runtime {} + tail {}".format(
            scaled_runtimes[rem_app][ker[rem_app]], rem_runtimes[rem_app]))
        scaled_runtimes[rem_app][ker[rem_app]] += rem_runtimes[rem_app]
        ker[rem_app] += 1
        logger.debug("============================ FINISHED EXECUTION OF MIXED KERNEL =============================")
        logger.debug("")
        logger.debug("scaled runtime app 0: {}".format(scaled_runtimes[0]))
        logger.debug("scaled runtime app 1: {}".format(scaled_runtimes[1]))
        logger.debug("")


        # remaining app completes isolated execution of mixed iteration
        if ker[rem_app] < len(apps[rem_app]):
            logger.debug(
                "==================== EXECUTION OF ISOLATED PORTION OF MIXED ITERATION =======================")
            logger.debug("remaining app on kernel: {}".format(ker[rem_app]))
            logger.debug(
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            for ind in range(ker[rem_app], len(apps[rem_app])):
                logger.debug("executing kernel {}: scaled runtime {} + {}".format(
                    ind, scaled_runtimes[rem_app][ind], apps[rem_app][ind]
                ))
                scaled_runtimes[rem_app][ind] += apps[rem_app][ind]
                logger.debug(
                    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            logger.debug(
                "================ FINISHED EXECUTION OF ISOLATED PORTION OF MIXED ITERATION ==================")
            logger.debug("")
        iter_[rem_app] += 1

        logger.debug("remaining app on iteration {}".format(iter_[rem_app]))
        logger.debug("scaled runtime app 0: {}".format(scaled_runtimes[0]))
        logger.debug("scaled runtime app 1: {}".format(scaled_runtimes[1]))

        # at this point we can calculate the co-run qos of the two apps
        co_run_qos_result = [0, 0]
        co_run_qos_result[0] = find_qos(scaled_runtimes[0], iter_[0], sum(apps[0]))
        co_run_qos_result[1] = find_qos(scaled_runtimes[1], iter_[1], sum(apps[1]))
        logger.debug("co-run QOS of apps is {}".format(co_run_qos_result))
        logger.debug("")

        # if remaining app has any more iterations to complete, they will run in isolation
        if iter_[rem_app] != iter_lim[rem_app]:
            rem_iter = iter_lim[rem_app] - iter_[rem_app]
            # find the number of iterations left for the remaining app
            logger.debug(
                "======================= EXECUTION OF REMAINING ISOLATED ITERATIONS =========================")
            # complete the rest of the required iterations of remaining app in isolation
            rem_kern = [x * rem_iter for x in apps[rem_app]]
            logger.debug("isolated kernel runtimes: {}".format(rem_kern))
            iter_[rem_app] += rem_iter
            scaled_runtimes[rem_app] = [sum(x) for x in zip(rem_kern, scaled_runtimes[rem_app])]
            logger.debug("scaled runtime app 0: {}".format(scaled_runtimes[0]))
            logger.debug("scaled runtime app 1: {}".format(scaled_runtimes[1]))
            logger.debug(
                "=================== FINISHED EXECUTION OF REMAINING ISOLATED ITERATIONS ====================")
            logger.debug("")

        # calculate resulting qos
        qos_result = [find_qos(scaled_runtimes[0], iter_[0], sum(apps[0])),
                      find_qos(scaled_runtimes[1], iter_[1], sum(apps[1]))]
    else:
        logger.debug("both apps completed their iterations simultaneously")
        # if both apps finished at the same time, then qos can be calculated immediately
        qos_result = [find_qos(scaled_runtimes[0], iter_[0], sum(apps[0])),
                      find_qos(scaled_runtimes[1], iter_[1], sum(apps[1]))]
        co_run_qos_result = qos_result

    logger.debug("======================================= FINAL QOS ==========================================")
    logger.debug("Simulated QOS of apps: {}".format(qos_result))
    return qos_result, co_run_qos_result
