import random
import numpy as np
import scipy
import logging
import pandas as pd

# convert csv file to dataframe, round to
slowdowns = pd.read_csv('slowdown.csv', sep=',', header=None).round(3).replace(0.000, 0.500)
# get the kernels, divide by 2000 to convert cycles to microseconds at 2 GHz, cut first row and round to 3 decimals
kernels = (pd.read_csv('kernel_idx.csv', sep=',', skiprows=1, header=None).iloc[ : , 2] / 2000).round(3)

print(slowdowns)
print(kernels)

# build 10000 apps for testing
# for i in range(1000):
# randomly pick the sizes of two apps
app0_size = random.randrange(1, 30)
app1_size = random.randrange(1, 30)

# for each app, pick kernel indeces at random to fill the app's kernels
df_app0 = (kernels.sample(n=app0_size, replace=True))
# print(df_app0)
# app0 = df_app0.to_numpy()
# print(df_app0.loc[5])
# print(df_app0.iloc[5])

# for app 1, pick kernels at rando mas well, but be careful to not have 2 kernels that are in both apps



# if __name__ == "__main__":
