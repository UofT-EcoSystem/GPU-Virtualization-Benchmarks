import data.scripts.predict.predict_slowdown as tree_model
import data.scripts.common.constants as const

# Synthetic workload configs #
REPEAT = 0
FRESH = 1
ITER_CHOICES = [400, 800, 1200, 1600, 2000, 2400, 2800]
NUM_JOB_CHOICES = [100, 200, 300, 400]
NUM_BENCH_CHOICES = [4, 8, 12, 16]

# MIG configs #
MAX_PARTITIONS = 8
MIG_PARTITIONS = [0, 1, 3, 7]

# Stage 1 predictor #
TRAINING_SET_RATIO = 0.8
KERNEL_COLS = list(set().union(list(tree_model.metric_dict.values()),
                               const.EXEC_CTX,
                               ['pair_str', 'intra']))

# Stage 2 predictor #
EQUALITY_ERROR = 0.0001
STEADY_STEP = 20
QOS_LOSS_ERROR = 0.001
