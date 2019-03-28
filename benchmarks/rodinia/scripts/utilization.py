import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os.path
import seaborn as sns


plt.rcParams["figure.figsize"] = (20, 15)


results = pd.DataFrame()

directory = "results-utilsust/"

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
    groups = data.groupby('Metric Name')[['Metric Value']].max().T
    groups['Benchmark'] = filename.split('.')[0]
    groups['Control'] = groups['ADU'] + groups['CBU']
    results = results.append(groups)


cols = ['Benchmark', 'Control', 'FP16', 'FP32', 'FP64', 'INT', 
        'SPU', 'Tensor', 'UNIFORM', 'Load/Store' ]
results = results[cols].values
# print(results)

# draw stack bars
idx = np.arange(results.shape[0])

p = []
# colors = sns.color_palette("husl", len(cols)-1)
colors = sns.color_palette("Set2", len(cols))

legends = ['Control', 'FP16', 'FP32', 'FP64', 'INT', 
        'SPU', 'Tensor', 'Uniform', 'Ld/St']

for i, col in enumerate(results.T):
    if i == 0:
        continue

    p.append(plt.plot(idx, results[:, i], color=colors[i], label=legends[i-1], marker='x'))
    

plt.xticks(idx, results[:, 0], fontsize=16, rotation=30)
plt.yticks(fontsize=16)
plt.ylabel("Utilization (%)", fontsize=18)
#plt.title("Utilization w.r.t Burst Mode During Active Cycles", fontsize=22)
plt.title("Utilization w.r.t Sustained Mode During Active Cycles", fontsize=22)


#plt.legend(p, legends, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend()

plt.savefig('util-sust.pdf',  bbox_inches='tight')
    
