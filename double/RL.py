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

theta1_record=[]
theta2_record=[]

theta1_velocity_record=[]
theta2_velocity_record=[]

theta1_integral_record=[]
theta2_integral_record=[]

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
    theta1_velocity_record.append(states[6])
    theta2_velocity_record.append(states[7])
    actions, internals = coach.act(states=states, internals=internals, independent=True, deterministic=True)
    states, terminal, reward = environment.execute(actions=actions)
    actions_record.append(actions)

length=len(theta1_record)
x=range(length)
plt.plot(x,theta1_record,label='Lower Angle',color='black')
plt.plot(x,theta1_velocity_record,label='Lower Angular Velocity',color='blue',alpha=0.5)
plt.plot(x,theta1_integral_record,label='Integrals',color='magenta',alpha=0.5)

plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper right',ncol=1, borderaxespad=0,prop={'size': 12})
#plt.ylim(-0.1,0.1)
#plt.savefig('RL.png')
plt.show()

plt.plot(x,theta2_record,label='Upper Angle',color='black')
plt.plot(x,theta2_velocity_record,label='Upper Angular Velocity',color='blue',alpha=0.5)
plt.plot(x,theta2_integral_record,label='Integrals',color='magenta',alpha=0.5)

plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper right',ncol=1, borderaxespad=0,prop={'size': 12})
#plt.ylim(-0.1,0.1)
#plt.savefig('RL.png')
plt.show()


plt.plot(x,actions_record,label='actions',color='blue')
plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper right',ncol=1, borderaxespad=0,prop={'size': 12})
#plt.ylim(-0.1,0.1)
#plt.savefig('RL.png')
plt.show()





