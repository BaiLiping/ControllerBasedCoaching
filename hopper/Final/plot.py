from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

episode_number=3000
evaluation_episode_number=50
average_over=100
length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-3,-5,-7,-10]
prohibition_position=[0.01,0.05,0.1,0.15]

reward_record_without=pickle.load(open( "without_record.p", "rb"))
evaluation_reward_record_without=pickle.load(open( "evaluation_without_record.p", "rb"))
reward_record_without_average=moving_average(reward_record_without,average_over)

reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=pickle.load(open( "record.p", "rb"))
for k in range(len(prohibition_position)):
    for i in range(len(prohibition_parameter)):
        reward_record_average[k][i]=moving_average(reward_record[k][i],average_over)

evaluation_reward_record=pickle.load(open( "evaluation_record.p", "rb"))
