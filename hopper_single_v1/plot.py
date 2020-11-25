from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

#setparameters
num_steps=100 #update exploration rate over n steps
initial_value=0.9 #initial exploartion rate
decay_rate=0.5 #exploration rate decay rate
set_type='exponential' #set the type of decay linear, exponential
exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)

episode_number=1500
evaluation_episode_number=10
average_over=50

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)
evaluation_reward_record_normal=pickle.load(open( "evaluation_normal_record.p", "rb"))
reward_record_normal_average=pickle.load(open( "normal_average_record.p", "rb"))

reward_record_single_average=pickle.load(open( "single_average_record.p", "rb"))
reward_record_single=pickle.load(open( "single_record.p", "rb"))
evaluation_reward_record_single=pickle.load(open( "evaluation_single_record.p", "rb"))

#plot
x=range(len(measure_length))
plt.figure(figsize=(20,10))
plt.plot(x,reward_record_normal_average,label='normal agent',color='black')
plt.plot(x,reward_record_single_average,label='single agent',color='red')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
plt.savefig('plot_compare.png')
print(evaluation_reward_record_normal)
print(evaluation_reward_record_single)
