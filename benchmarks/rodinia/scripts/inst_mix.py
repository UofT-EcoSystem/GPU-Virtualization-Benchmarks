import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os.path
import seaborn as sns


plt.rcParams["figure.figsize"] = (20, 15)


results = pd.DataFrame()

directory = "results-gpu/"

for filename in os.listdir(directory):
    full_filename = os.path.join(directory, filename)

    # find out number of lines to skip in file
    n_skip = 0
    n_lines = 0
    
    if not filename.endswith(".txt"):
        continue
        
    with open(full_filename) as search:
        for num, line in enumerate(search, 1):
            n_lines = n_lines + 1
            if '==PROF==' in line:
                n_skip = num
            

    # exit if ==prof== is on the last line
    if n_skip == n_lines or n_skip == 0:
        print(filename, "went wrong!")
        continue
    
    print("Parsing", filename)

    
    # read data using pandas
    data = pd.read_csv(full_filename, delimiter=',', skiprows=n_skip, thousands=',')
    # divide by 1 million
    data['Metric Value'] /= 1000000
    groups = data.groupby('Metric Name')[['Metric Value']].sum().T
    groups['Benchmark'] = filename.split('.')[0]
    groups['Control'] = groups['ADU'] + groups['CBU']
    results = results.append(groups)


cols = ['Benchmark', 'Control', 'FP16', 'FP32', 'FP64', 'INT', 
        'SPU', 'Tensor', 'UNIFORM', 'Load/Store', 'EXEC' ]
results = results[cols].values
# print(results)

# draw stack bars
idx = np.arange(results.shape[0])
width = 0.35

accum = np.zeros(results.shape[0], dtype=np.float32)

p = []
# colors = sns.color_palette("husl", len(cols)-1)
colors = sns.color_palette("Set2", len(cols))

for i, col in enumerate(results.T):
    if i == 0 or i == len(cols)-1:
        continue
    height = np.divide(col, results[:,-1])
    p.append(plt.bar(idx, height, width, bottom=accum, color=colors[i], edgecolor='black'))
    accum += height.astype(np.float32)
    
# add the other uncaptured types
p.append(plt.bar(idx, 1-accum, width, bottom=accum, edgecolor='black'))

plt.xticks(idx, results[:, 0], fontsize=16, rotation=30)
plt.yticks(fontsize=16)

legends = ['Control', 'FP16', 'FP32', 'FP64', 'INT', 
        'SPU', 'Tensor', 'Uniform', 'Ld/St', 'Other']
# flip legends
p.reverse()
legends.reverse()

plt.legend(p, legends, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('inst.pdf',  bbox_inches='tight')
    
