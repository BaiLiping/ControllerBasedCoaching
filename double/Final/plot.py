from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


episode_number=2000
evaluation_episode_number=5
average_over=150
length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-10,-20,-30,-40,-50]
prohibition_position=[0.2,0.3,0.4,0.5]


reward_record_without=pickle.load(open( "without_record.p", "rb"))
evaluation_reward_record_without=pickle.load(open( "evaluation_without_record.p", "rb"))
reward_record_without_average=moving_average(reward_record_without,average_over)
print(evaluation_reward_record_without)


reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=pickle.load(open( "record.p", "rb"))
for k in range(len(prohibition_position)):
    for i in range(len(prohibition_parameter)):
        reward_record_average[k][i]=moving_average(reward_record[k][i],average_over)

evaluation_reward_record=pickle.load(open( "evaluation_record.p", "rb"))
print(evaluation_reward_record)

#plot training results
color_scheme=['yellowgreen','magenta','orange','blue','red','cyan','green']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure(figsize=(10,10))
    plt.plot(x,reward_record_without_average,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record_average[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
    plt.savefig('single_pendulum_with_boundary_at_%s_plot.png' %prohibition_position[i])
