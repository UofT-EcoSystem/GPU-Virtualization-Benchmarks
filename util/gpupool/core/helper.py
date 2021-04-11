from enum import Enum
import os
import pickle

from gpupool.core.configs import *

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


class QOS(Enum):
    PCT_50 = 0.5
    PCT_60 = 0.6
    PCT_70 = 0.7
    PCT_80 = 0.8
    PCT_90 = 0.9


class Allocation(Enum):
    One_D = 1
    Two_D = 2
    Three_D = 3


class StageOne(Enum):
    GPUSim = 1
    BoostTree = 2


class StageTwo(Enum):
    Full = 1
    Steady = 2
    Weighted = 3
    GPUSim = 4


class GpuPoolConfig:
    def __init__(self, alloc: Allocation, stage_1: StageOne, stage_2: StageTwo,
                 at_least_once, accuracy_mode, stage2_buffer):
        self.alloc = alloc
        self.stage_1 = stage_1
        self.stage_2 = stage_2

        # deprecated
        self.at_least_once = at_least_once

        self.model = None
        self.accuracy_mode = accuracy_mode
        self.stage2_buffer = stage2_buffer

        model_pkl_path = os.path.join(THIS_DIR, "model.pkl")

        from gpupool.core.predict import RunOption
        if self.stage_1 == StageOne.BoostTree:
            # Runtime mode, simply do inference on a single model
            if os.path.isfile(model_pkl_path):
                print("Load boosting tree from pickle {}"
                      .format(model_pkl_path))

                self.model = pickle.load(open(model_pkl_path, 'rb'))
            else:
                print("Training boosting tree for stage 1.")
                self.model = RunOption.train_boosting_tree(train_all=True)
                pickle.dump(self.model, open(model_pkl_path, 'wb'))

    def to_string(self):
        combo_name = "{}-{}-{}-{}".format(self.alloc.name,
                                          self.stage_1.name,
                                          self.stage_2.name,
                                          self.at_least_once)
        return combo_name

    def get_ws(self):
        return self.to_string() + '-ws'

    def get_perf(self):
        return self.to_string() + '-perf'

    def get_option(self):
        return self.to_string() + '-option'

    def get_time_prediction(self):
        return self.to_string() + '-time-prediction'

    def get_time_matching(self):
        return self.to_string() + '-time-matching'

    def get_model(self):
        return self.model


class Violation:
    def __init__(self, count=0, err_sum=0, err_max=0, gpu_increase=0,
                 actual_ws=0, actual_ws_list=None, job_sld=None,
                 ws_no_migrate=0):
        self.count = count
        self.sum = err_sum
        self.max = err_max
        self.gpu_increase = gpu_increase
        self.actual_ws = actual_ws
        self.ws_no_migrate = ws_no_migrate

        if job_sld is None:
            self.job_sld = {}
        else:
            self.job_sld = job_sld

        if actual_ws_list is None:
            self.actual_ws_list = []
        else:
            self.actual_ws_list = actual_ws_list

    def update(self, actual_qos, target_qos):
        from gpupool.core.predict import RunOption
        if actual_qos + QOS_LOSS_ERROR < target_qos:
            self.count += 1
            error = (target_qos - actual_qos) / target_qos
            self.sum += error

            if error > self.max:
                self.max = error

            # return did violate
            return True
        else:
            return False

    def mean_error_pct(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count * 100

    def max_error_pct(self):
        return self.max * 100

    def to_string(self, num_jobs):
        return "{} QoS violations ({:.2f}% jobs, {:.2f}% mean relative " \
               "error, {:.2f}% max error)".format(self.count,
                                                  self.count / num_jobs * 100,
                                                  self.mean_error_pct(),
                                                  self.max_error_pct())

    def __add__(self, other):
        new_count = self.count + other.count
        new_sum = self.sum + other.sum
        new_max = max(self.max, other.max)
        new_gpu_increase = self.gpu_increase + other.gpu_increase
        new_ws = self.actual_ws + other.actual_ws
        new_ws_list = self.actual_ws_list + other.actual_ws_list
        new_ws_no_migrate = self.ws_no_migrate + other.ws_no_migrate

        new_job_sld = {}
        new_job_sld.update(self.job_sld)
        new_job_sld.update(other.job_sld)

        return Violation(new_count, new_sum, new_max,
                         new_gpu_increase, new_ws, new_ws_list, new_job_sld,
                         new_ws_no_migrate)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)