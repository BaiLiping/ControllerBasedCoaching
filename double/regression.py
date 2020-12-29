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
coach= Agent.load(directory='model2', format='numpy')
internals = coach.initial_internals()
actions_record=[]
theta_states=[]
for k in range(3):
    states = environment.reset()
    terminal=False

    while not terminal:
        print('sintheta1 %s sintheta2 %s vs1 %s vs2 %s:'%(states[1],states[2],states[6],states[7]))
        theta_states.append([states[1],states[2],states[6],states[7]])
        actions, internals = coach.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        actions_record.append(actions)
        print('action:',actions)
x, y = np.array(theta_states), np.array(actions_record)
#x_ = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
#model = LinearRegression().fit(x_, y)
#pickle.dump(model, open('regression_model.sav', 'wb'))

linear_model = LinearRegression().fit(x, y)
pickle.dump(linear_model, open('linear_model.sav', 'wb'))

print(linear_model.coef_)

