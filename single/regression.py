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


environment = Environment.create(environment='gym', level='InvertedPendulumBLP-v2')

#polynomial regression
coach= Agent.load(directory='model5', format='numpy')
internals = coach.initial_internals()
actions_record=[]
theta_states=[]
for k in range(1):
    states = environment.reset()
    terminal=False
    integrals=0
    while not terminal:
        integrals+=states[1]
        temp=[states[1],integrals,states[3]]
        theta_states.append(temp)
        actions, internals = coach.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        actions_record.append(actions)

x, y = np.array(theta_states), np.array(actions_record)
#x_ = PolynomialFeatures(degree=6, include_bias=True).fit_transform(x)
model = LinearRegression().fit(x, y)
pickle.dump(model, open('linear_model.sav', 'wb'))
print(model.coef_)

theta=[row[0] for row in theta_states]
theta_velocity=[row[2] for row in theta_states]
summation=[row[1] for row in theta_states]

length=len(theta)
x=range(len(theta))
plt.plot(x,theta,label='Angle',color='black')
plt.plot(x,theta_velocity,label='Angular Velocity',color='blue',alpha=0.5)
plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper right',ncol=1, borderaxespad=0,prop={'size': 12})
plt.ylim(-0.1,0.1)
plt.savefig('RL.png')

