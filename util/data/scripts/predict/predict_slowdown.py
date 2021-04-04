from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.model_selection import KFold
from graphviz import Source
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import random

import data.scripts.common.constants as const

metric_dict = OrderedDict([
    ('norm_ipc', 'norm_ipc'),
    ('diff_mflat', 'avg_mem_lat'),
    ('ratio_mpc', 'mpc'),
    ('ratio_bw', 'avg_dram_bw'),
    ('sum_bw', 'avg_dram_bw'),
    ('ratio_thread', 'thread_count'),
    ('ratio_sp_busy', 'sp_busy'),
    ('sum_sp_busy', 'sp_busy'),
    ('ratio_int_busy', 'int_busy'),
    ('sum_int_busy', 'int_busy'),
    ('sum_not_selected_cycles', 'not_selected_cycles'),
    # ('ratio_dp_busy', 'dp_busy'),
    # ('sum_dp_busy', 'dp_busy'),
    ('l2_miss_rate', 'l2_miss_rate'),
    # ('ratio_rbh', 'avg_rbh'),
])

# deprecated
cols_ws = ['sum_norm_ipc',
           'sum_avg_rbh',
           'sum_mflat',
           'sum_mpc',
           'sum_bw',
           'sum_inst_empty',
           'sum_int_busy',
           'sum_sp_busy',
           ]


def export_tree(clf, path_png, tree_id=300):
    sub_tree_300 = clf.estimators_[tree_id, 0]

    dot_data = tree.export_graphviz(
        sub_tree_300,
        out_file=None, filled=True,
        rounded=True,
        special_characters=True,
        proportion=True,
    )

    graph = Source(dot_data)
    png_bytes = graph.pipe(format='pdf')
    with open(path_png, 'wb') as f:
        f.write(png_bytes)


def prepare_datasets(df_pair, training=True):
    df_features = pd.DataFrame()

    def feature_set(sfx_base, sfx_other):
        def calculate_diff(derived_metric):
            metric = metric_dict[derived_metric]
            this_metric = df_pair[metric + '_' + sfx_base]
            other_metric = df_pair[metric + '_' + sfx_other]
            # this_metric = this_metric.replace(0, 1e-10)

            df_features[derived_metric + '_' + sfx_base] = \
                (this_metric - other_metric) / this_metric

        def calculate_ratio(derived_metric):
            metric = metric_dict[derived_metric]
            this_metric = df_pair[metric + '_' + sfx_base]
            other_metric = df_pair[metric + '_' + sfx_other]
            # other_metric = other_metric.replace(0, 1e-10)

            df_features[derived_metric + '_' + sfx_base] = \
                this_metric / other_metric

        def calculate_sum(derived_metric):
            metric = metric_dict[derived_metric]
            df_features[derived_metric + '_' + sfx_base] = \
                df_pair[metric + '_' + sfx_base] + df_pair[
                    metric + '_' + sfx_other]

        def get_metric(derived_metric):
            metric = metric_dict[derived_metric]
            df_features[derived_metric + '_' + sfx_base] = \
                df_pair[metric + '_' + sfx_base]

        for derived in metric_dict:
            if derived.startswith('diff'):
                calculate_diff(derived)
            elif derived.startswith('ratio'):
                calculate_ratio(derived)
            elif derived.startswith('sum'):
                calculate_sum(derived)
            else:
                get_metric(derived)

        cols = [c + '_' + sfx_base for c in metric_dict]
        x = df_features[cols].values
        x = x.astype(np.double)
        # print('X invalid?', np.isnan(x).any())
        # replace inf and -inf
        x = np.nan_to_num(x, neginf=-1e30, posinf=1e30)

        return x

    def ground_truth(sld_idx):
        y = [sld_list[sld_idx] for sld_list in df_pair['sld']]
        # print('y invalid?', np.isnan(y).any())
        return y

    x1 = feature_set('x', 'y')
    x2 = feature_set('y', 'x')

    if training:
        aggregate_x = np.concatenate((x1, x2), axis=0)

        y1 = ground_truth(1)
        y2 = ground_truth(2)
        aggregate_y = np.concatenate((y1, y2), axis=0)

        aggregate_x, aggregate_y = shuffle(aggregate_x, aggregate_y)

        # print(aggregate_x.shape, aggregate_y.shape)

        return aggregate_x, aggregate_y
    else:
        return [x1, x2]


def prepare_ws_datasets(df_pair):
    def calculate_sum(derived_metric, metric):
        df_pair[derived_metric] = \
            df_pair[metric + '_x'] + df_pair[metric + '_y']

    calculate_sum('sum_norm_ipc', 'norm_ipc')
    calculate_sum('sum_avg_rbh', 'avg_rbh')
    calculate_sum('sum_mflat', 'avg_mem_lat')
    calculate_sum('sum_mpc', 'mpc')
    calculate_sum('sum_bw', 'avg_dram_bw')
    calculate_sum('sum_inst_empty', 'inst_empty_cycles')
    calculate_sum('sum_int_busy', 'int_busy')
    calculate_sum('sum_sp_busy', 'sp_busy')

    X = df_pair[cols_ws].values
    X = X.astype(np.double)

    y = df_pair['ws']

    print('X invalid?', np.isnan(X).any())
    print('y invalid?', np.isnan(y).any())

    # replace inf and -inf
    X = np.nan_to_num(X, neginf=-1e30, posinf=1e30)

    X, y = shuffle(X, y, random_state=5)

    return X, y


def train(X, y, cross_validation=True):
    params = {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 2,
              'learning_rate': 0.1, 'loss': 'huber'}

    if cross_validation:
        # Fit gradient boost tree regression model using K-fold cross validation
        kf = KFold(n_splits=3, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = ensemble.GradientBoostingRegressor(**params)

            clf.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, clf.predict(X_test))
            print('K-fold l1 error:', mae)

    # get the final model
    X_train = X
    y_train = y
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)

    return clf


def linear(X, y):
    reg = LinearRegression().fit(X, y)
    return reg


def mlp(X, y, cross_validation=True):
    # hyperparameter search
    # max_num_layers = 1
    # exp_per_layer_count = 50
    # max_num_neurons = 1000
    # min_num_neurons = 100

    n_iter = 500
    params = {'random_state': 1, 'max_iter': n_iter,
              'hidden_layer_sizes': 100,
              'activation': 'tanh', 'alpha': 0.001,
              'learning_rate_init': 0.008,
              'solver': 'adam'}

    # generate search space
    # hidden_sizes = []
    # for layer_count in range(1, max_num_layers+1):
    #     for exp in range(exp_per_layer_count):
    #         sizes = [random.randint(min_num_neurons, max_num_neurons)
    #                  for _ in range(layer_count)]
    #         hidden_sizes.append(tuple(sizes))
    hidden_sizes = [460]

    # find best arch config
    min_error = float("inf")
    min_config = hidden_sizes[0]
    for hidden_config in hidden_sizes:
        print("config:", hidden_config)
        params['hidden_layer_sizes'] = hidden_config

        if cross_validation:
            # Fit gradient boost tree regression model
            # using K-fold cross validation
            kf = KFold(n_splits=3, shuffle=True)
            errors = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                reg = MLPRegressor(**params)

                reg.fit(X_train, y_train)

                mae = mean_absolute_error(y_test, reg.predict(X_test))
                print('K-fold l1 error:', mae)
                errors.append(mae)

            current_error = np.average(errors)
            if current_error < min_error:
                min_error = current_error
                min_config = hidden_config

    print("min error:", min_error)
    print("min config:", min_config)
    params['hidden_layer_sizes'] = min_config
    # get the final model
    reg = MLPRegressor(**params)
    reg.fit(X, y)

    return reg


def plot_importance(clf, type='old'):
    fig = plt.figure()
    plt.style.use('seaborn-paper')
    # #########################################################################
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    feature_importance = feature_importance

    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    if type == 'ws':
        np_cols = np.array(cols_ws)
    else:
        np_cols = np.array(list(metric_dict.keys()))

    plt.yticks(pos, np_cols[sorted_idx])
    plt.xlabel('Importance Score')
    # plt.title('Variable Importance')

    # plt.show()
    fig.savefig(os.path.join(const.DATA_HOME, 'graphs/grad_boosting_imp.pdf'),
                bbox_inches='tight')


def predict_from_df(clf, df_pair):
    xs = prepare_datasets(df_pair, training=False)

    count = sum([x.shape[0] for x in xs])

    print(count, "predictions")

    start = time.perf_counter()
    ys = [clf.predict(x) for x in xs]
    end = time.perf_counter()

    print("Took", end - start, "seconds")

    return ys


def parse_args():
    parser = argparse.ArgumentParser('Train and test a model that predicts '
                                     'kernel slowdown in shared mode based on '
                                     'isolated mode metrics.')
    parser.add_argument('--pair_pkl',
                        required=True,
                        help='Pickle file that contains all the isolated mode '
                             'and shared mode performance counters.')

    parser.add_argument('--output_tree',
                        default=os.path.join(const.DATA_HOME,
                                             'plots/tree.png'),
                        help='Output path for visualizing the gradient boost '
                             'tree.')
    results = parser.parse_args()
    return results


def main():
    args = parse_args()
    X, y = prepare_datasets(args.pair_pkl)

    clf = train(X, y)
    export_tree(clf, args.output_tree)


if __name__ == '__main__':
    main()
