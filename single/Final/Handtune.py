from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

kp=25
kd=2.27645649
# Pre-defined or custom environment
environment = gym.make('InvertedPendulum-v2')
print('testing PID controller')
episode_reward=0
states = environment.reset()
terminal=False
while not terminal:
	angle=states[1]
	angular_velocity=states[3]
	print('angle: %s velocity: %s' %(angle,angular_velocity))
	if abs(angle)<0.07:
		actions=kp*angle+kd*angular_velocity
		print('automatic action:', actions)
		states, reward, terminal,info = environment.step(actions)
	else:
		actions = input("Please enter the action:\n")
		states, reward, terminal,info = environment.step(actions)

	episode_reward+=reward
print(episode_reward)
