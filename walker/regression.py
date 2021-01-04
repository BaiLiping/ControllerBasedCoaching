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
coach= Agent.load(directory='Walker_RL', format='numpy')
internals = coach.initial_internals()
actions_record=[]
theta_states=[]
for k in range(20):
    states = environment.reset()
    terminal=False
    episode_reward=0
    while not terminal:
        #theta_states.append([1.25-states[0],states[1],states[9],states[10]])
        #theta_states.append([1.25-states[0],states[1]])
        #theta_states.append([states[9],states[10]])
        theta_states.append([states[1],states[10]])
        actions, internals = coach.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        #print('z: %s y: %s z_velocity: %s y_velocity: %s'%(1.25-states[0],states[1],states[9],states[10]))
        #print('z: %s y: %s'%(0,states[10]))
        #print('actions: ',actions)
        print(states)
        actions_record.append(actions)
        episode_reward+=reward
#print(episode_reward)
x, y = np.array(theta_states), np.array(actions_record)

#y=[row[0] for row in theta_states]
#thigh=[row[1] for row in theta_states]
#left_thigh=[row[2] for row in theta_states]
#y_velocity=[row[1] for row in theta_states]
#thigh_velocity=[row[4] for row in theta_states]
#left_thigh_velocity=[row[5] for row in theta_states]

#length=len(y)
#plt.plot(range(length),y)
#plt.show()
#plt.plot(range(length),y_velocity)
#plt.show()
# plt.plot(range(length),thigh)
# plt.show()
# plt.plot(range(length),left_thigh)
# plt.show()
#x_ = PolynomialFeatures(degree=5, include_bias=True).fit_transform(x)
#model = LinearRegression().fit(x_, y)
#pickle.dump(model, open('regression_model.sav', 'wb'))

linear_model = LinearRegression().fit(x, y)
pickle.dump(linear_model, open('linear_model.sav', 'wb'))
print(linear_model.coef_)

