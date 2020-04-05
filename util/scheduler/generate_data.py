import random
import numpy as np
import pickle
import pandas as pd

# convert csv file to dataframe, round to
slowdowns = pd.read_csv('slowdown.csv', sep=',', header=None).round(3).replace(0.000, 0.500)
# get the kernels, divide by 2000 to convert cycles to microseconds at 2 GHz, cut first row and round to 3 decimals
kernels = pd.read_csv('kernel_idx.csv', sep=',', header=0, usecols=["idx", " runtime(cycles)"]).round(3)
kernels[" runtime(cycles)"] = kernels[" runtime(cycles)"] / 2000
# rename the column
kernels = kernels.rename(columns={"idx": "idx", " runtime(cycles)": "microseconds"})

# print(slowdowns)
# print(kernels)

apps = []

# build 10000 apps for testing
for i in range(10000):
    # randomly pick the sizes of two apps
    app0_size = random.randrange(1, 30)
    app1_size = random.randrange(1, 30)

    # shuffle the kernels
    kernels = kernels.sample(frac=1)
    # split dataframe into 2 dataframes down the middle
    app0_kernels = kernels[:8]
    app1_kernels = kernels[8:]

    # select kernels for app0
    app0_df = app0_kernels.sample(n=app0_size, replace=True)
    # print("app 0 is: \n", app0_df)
    app1_df = app1_kernels.sample(n=app1_size, replace=True)
    # print("app1 is : \n", app1_df)

    # build interference matrices
    interference_0 = np.zeros((app0_size, app1_size), dtype=float)
    for i in range(app0_size):
        for j in range(app1_size):
            # assign appropriate slowdown
            interference_0[i][j] = slowdowns[app1_df.iloc[j][0]][app0_df.iloc[i][0]]

    interference_1 = np.zeros((app0_size, app1_size), dtype=float)
    for i in range(app0_size):
        for j in range(app1_size):
            # assign appropriate slowdown
            interference_1[i][j] = slowdowns[app0_df.iloc[i][0]][app1_df.iloc[j][0]]

    # interference matrices ready to be pickled
    interference = [interference_0, interference_1]

    # convert app0 and app1 dataframes to numpy arrays
    app0 = (app0_df.loc[:, "microseconds"]).values
    app1 = (app1_df.loc[:, "microseconds"]).values

    # apps ready to be pickled
    app_lengths = [app0, app1]

    app = [app_lengths, interference]
    apps.append(app)

# pickle apps into a file
pickle.dump(apps, open("10000_apps.bin", "wb"))
print(apps[7])



