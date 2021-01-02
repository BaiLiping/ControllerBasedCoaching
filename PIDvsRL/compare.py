from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

rl=[147,2251,2725]
pid=[82,1327,2143]

# data to plot
n_groups = 3

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, pid, bar_width,
alpha=opacity,
color='b',
label='With PID Coaching')

rects2 = plt.bar(index + bar_width, rl, bar_width,
alpha=opacity,
color='g',
label='Without Coaching')

plt.ylabel('episode number')
plt.xticks(index + bar_width, ('Inverted Pendulum', 'Double Inverted Pendulum', 'Hopper'))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',shadow=True, borderaxespad=0)
plt.tight_layout()
plt.savefig('compare.png')