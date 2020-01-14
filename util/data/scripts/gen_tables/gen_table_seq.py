import common.constants as const
import common.help_iso as hi

import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser('Generate dataframe pickle for sequential run from csv.')
    parser.add_argument('--csv',
                        default=os.path.join(const.DATA_HOME, 'csv/seq.csv'),
                        help='CSV file to parse')
    parser.add_argument('--output',
                        default=os.path.join(const.DATA_HOME, 'pickles/seq.pkl'),
                        help='Output path for the dataframe pickle')

    results = parser.parse_args()
    return results


# Parse arguments
args = parse_args()

# Read CSV file
df = pd.read_csv(args.csv)

# drop any benchmarks that have zero runtime
df = df[df['runtime'] > 0]

# avg dram bandwidth
df['avg_dram_bw'] = df['dram_bw'].transform(hi.avg_array)
df['avg_dram_eff'] = df['dram_eff'].transform(hi.avg_array)

# standard deviation of dram bandwidth among channels
df['std_dram_bw'] = df['dram_bw'].transform(hi.std_array)
df['ratio_dram_bw'] = df['std_dram_bw'] / df['avg_dram_bw']

# MPKI
df['MPKI'] = df['l2_total_accesses'] * df['l2_miss_rate'] / (df['instructions'] / 1000)

# waves
df['waves'] = df.apply(lambda row:
                       row['grid_x'] * row['grid_y'] * row['grid_z']
                       / const.num_sm_volta / const.app_dict[row['pair_str']][0], axis=1)

# l2_accesses
df['l2_access_density'] = df['l2_total_accesses'] / (df['instructions'] / 1000)

# sort table based on benchmark name
df.sort_values('pair_str', inplace=True)

# Output pickle
df.to_pickle(args.output)
