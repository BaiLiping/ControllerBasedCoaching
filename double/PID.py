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


#kp=[-0.5,-2.9]
#kd=[-0.5,-0.6]
#kp=[-5.69332015e-01 ,-3.90236132e-02 ,-5.14855973e-01]
#kd=[-3.07251717e+00 ,-2.98465176e-03 ,-7.36120966e-01]
kp=[-0.5,0,-2.9]
kd=[-0.5,0,-0.6]


# polynomial controller
environment_control = gym.make('InvertedDoublePendulum-v2')
episode_reward=0
theta1_integral=0
theta2_integral=0
states = environment_control.reset()
terminal=False

theta1_record=[]
theta2_record=[]

theta1_velocity_record=[]
theta2_velocity_record=[]

theta1_integral_record=[]
theta2_integral_record=[]

actions_record=[]

while not terminal:
    sintheta1=states[1]
    theta1_record.append(sintheta1)
    theta1_integral+=sintheta1
    theta1_integral_record.append(theta1_integral)
    theta1_velocity_record.append(states[6])
    sintheta2=states[2]
    theta2_record.append(sintheta2)
    theta2_integral+=sintheta2
    theta2_integral_record.append(theta2_integral)
    theta2_velocity_record.append(states[7])
    #actions_predict=kp[0]*states[1]+kp[1]*states[2]+kd[0]*states[6]+kd[1]*states[7]
    actions_predict=kp[0]*states[1]+kp[1]*theta1_integral+kp[2]*states[2]+kd[0]*states[6]+kd[1]*theta2_integral+kd[2]*states[7]
    states, reward, terminal,info = environment_control.step(actions_predict)
    episode_reward+=reward
    actions_record.append(actions_predict)
print(episode_reward)

x=range(len(theta1_record))
fig=plt.figure(figsize=(10,7))
plt.plot(x,theta1_record,label='Lower Angle',color='black')
plt.plot(x,theta1_velocity_record,label='Lower Angular Velocity',color='blue',alpha=0.5)
plt.plot(x,theta1_integral_record,label='Integral',color='magenta',alpha=0.5)
plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper right',ncol=1, borderaxespad=0,prop={'size': 16})
#plt.ylim(-0.5,0.5)
#plt.savefig('double_PID.png')
plt.show()

fig=plt.figure(figsize=(10,7))
plt.plot(x,theta2_record,label='Upper Angle',color='black')
plt.plot(x,theta2_velocity_record,label='Upper Angular Velocity',color='blue',alpha=0.5)
plt.plot(x,theta2_integral_record,label='Integral',color='magenta',alpha=0.5)
plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper right',ncol=1, borderaxespad=0,prop={'size': 16})
#plt.ylim(-0.5,0.5)
plt.show()

fig=plt.figure(figsize=(10,7))
plt.plot(x,actions_record,label='actions',color='green')
plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper right',ncol=1, borderaxespad=0,prop={'size': 16})
#plt.ylim(-0.5,0.5)
plt.show()





environment_control.close()
