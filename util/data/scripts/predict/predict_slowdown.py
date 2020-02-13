from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.model_selection import KFold
from graphviz import Source
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

import data.scripts.common.constants as const

cols_prefix = ['norm_ipc',
               # 'avg_rbh',
               'diff_rbh',
               'diff_mflat',
               'diff_bw',
               'diff_mpc',
               'diff_thread',
               'sum_bw',
               # 'l2_miss_rate',
               # 'avg_dram_bw',
               # 'avg_mem_lat',
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
    png_bytes = graph.pipe(format='png')
    with open(path_png, 'wb') as f:
        f.write(png_bytes)


def prepare_datasets(df_pair, cc):

    def feature_set(sfx_base, sfx_other):
        def calculate_diff(derived_metric, metric):
            df_pair[derived_metric+'_'+sfx_base] = \
                (df_pair[metric+'_'+sfx_base] - df_pair[metric+'_'+sfx_other]) \
                / df_pair[metric+'_'+sfx_base]

        calculate_diff('diff_mflat', 'avg_mem_lat')
        calculate_diff('diff_bw', 'avg_dram_bw')
        calculate_diff('diff_mpc', 'mpc')
        calculate_diff('diff_thread', 'thread_count')
        calculate_diff('diff_rbh', 'avg_rbh')

        df_pair['sum_bw_'+sfx_base] = df_pair['avg_dram_bw_x'] + df_pair[
            'avg_dram_bw_y']

        cols = [c + '_' + sfx_base for c in cols_prefix]
        X = df_pair[cols].values
        X = X.astype(np.float32)

        sld_col = '1_sld' if sfx_base == 'x' else '2_sld'
        y = df_pair[sld_col]

        print('X invalid?', np.isnan(X).any())
        print('y invalid?', np.isnan(y).any())

        # replace inf and -inf
        X = np.nan_to_num(X, neginf=-1e30)

        return X, y

    X1, y1 = feature_set('x', 'y')
    X2, y2 = feature_set('y', 'x')

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    X, y = shuffle(X, y, random_state=7)

    print(X.shape, y.shape)

    return X, y


def train(X, y):
    # #########################################################################
    # Fit gradient boost tree regression model using K-fold cross validation
    kf = KFold(n_splits=5)
    params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'huber'}

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


def plot_importance(clf):
    # #########################################################################
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    np_cols = np.array(cols_prefix)
    plt.yticks(pos, np_cols[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')

    plt.show()
    plt.close()


def predict_from_df(clf, df_pair, suffix):
    cols = [c + '_' + suffix for c in cols_prefix]

    X = df_pair[cols].values.astype(float)
    X = np.nan_to_num(X, neginf=-1e30)

    return clf.predict(X)


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

