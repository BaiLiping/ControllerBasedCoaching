from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


environment = Environment.create(environment='gym', level='Walker2d-v3')
#polynomial regression"
RL= Agent.load(directory='Walker_RL', format='numpy')
internals = RL.initial_internals()
actions_record=[]
theta_states=[]
states = environment.reset()
terminal=False
episode_reward=0
while not terminal:
    theta_states.append([states[1],states[10]])
    actions, internals = RL.act(states=states, internals=internals, independent=True, deterministic=True)
    states, terminal, reward = environment.execute(actions=actions)
    actions_record.append(actions)
    episode_reward+=reward

y_record=[row[0] for row in theta_states]
y_velocity=[row[1] for row in theta_states]
x=range(len(y_record))
fig=plt.figure(figsize=(10,7))
plt.plot(x,y_record,label='Y Position',color='black')
plt.plot(x,y_velocity,label='Y Velocity',color='blue',alpha=0.5)
plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper right',ncol=1, borderaxespad=0,prop={'size': 16})
plt.ylim(-1,1)
#plt.savefig('RL.png')
#plt.show()
#plt.plot(x,actions_record)
plt.show()
#plt.plot(x,actions_record)
#plt.show()


RL.close()
environment.close()