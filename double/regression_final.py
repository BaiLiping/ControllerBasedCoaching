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


environment = Environment.create(environment='gym', level='InvertedDoublePendulum-v2')

#polynomial regression
coach= Agent.load(directory='Double_Model', format='numpy')
internals = coach.initial_internals()
actions_record=[]
theta_states=[]

theta1_record=[]
theta2_record=[]

theta1_integral_record=[]
theta2_integral_record=[]

for k in range(20):
    states = environment.reset()
    terminal=False
    theta1_integral=0
    theta2_integral=0

    while not terminal:
        sintheta1=states[1]
        theta1_record.append(sintheta1)
        theta1_integral+=sintheta1
        theta1_integral_record.append(theta1_integral)
        sintheta2=states[2]
        theta2_record.append(sintheta2)
        theta2_integral+=sintheta2
        theta2_integral_record.append(theta2_integral)
        velocity_theta1=states[6]
        velocity_theta2=states[7]
        theta_states.append([sintheta1,theta1_integral,velocity_theta1,sintheta2,theta2_integral,velocity_theta2])
        #theta_states.append([sintheta1,theta1_integral,velocity_theta1])        
        actions, internals = coach.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        actions_record.append(actions)
        #print('action:',actions)
x, y = np.array(theta_states), np.array(actions_record)
x_ = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
model = LinearRegression().fit(x, y)
pickle.dump(model, open('regression_model.sav', 'wb'))
print(model.coef_)




