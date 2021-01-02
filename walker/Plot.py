from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm

from Normal import episode_number
from Normal import average_over
from Normal import evaluation_episode_number
from Normal import exploration


#load data
walker_without=pickle.load(open( "walker_without_record.p", "rb"))
#walker_record=pickle.load(open( "walker_record.p", "rb"))[0][0][:5800]
walker_evaluation_record_without=pickle.load(open( "walker_evaluation_without_record.p", "rb"))
#walker_evaluation_record=pickle.load(open( "walker_evaluation_record.p", "rb"))[0][0][:5800]
evalu_without_ave=sum(walker_evaluation_record_without)/evaluation_episode_number
#evalu_ave=sum(walker_evaluation_record)/evaluation_episode_number


#plot training results
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
walker_without_average=moving_average(walker_without,average_over)
#walker_record_average=moving_average(walker_record,average_over)

fig=plt.figure(figsize=(13,7))
env_standard=800
x=range(len(walker_without_average))
plt.plot(x,walker_without_average,label='Normal Training\nEvaluation %s'%evalu_without_ave,color='black',linestyle='-.')
#plt.plot(x,walker_record_average,label='Coached by PID Controller\nEvaluation %s'%evalu_ave,color='magenta')
plt.xlabel('Episode Number', fontsize='large')
plt.ylabel('Episode Reward', fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 18})
plt.axhline(y=env_standard, color='black', linestyle='dotted')
plt.savefig('walker.png')