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

#model=pickle.load(open('coaching_model.sav', 'rb'))
model=pickle.load(open('linear_model.sav', 'rb'))



# polynomial controller
environment_control = gym.make('Hopper-v3')
episode_reward=0
states = environment_control.reset()
terminal=False
while not terminal:
    #print('states:',states)
    x=np.array(states)
    #x_ = PolynomialFeatures(degree=6, include_bias=True).fit_transform([x])
    #actions_predict= model.predict(x_)
    actions_predict=model.predict([x])
    print('action: ',actions_predict)
    #actions_predict=np.clip(actions_predict.copy(), -10, 10)
    states, reward, terminal,info = environment_control.step(actions_predict)
    episode_reward+=reward
    print('reward: ',reward)

print(episode_reward)