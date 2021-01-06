from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym



kp=30
ki=0.001
kd=2.26


angle_record=[]
velocity_record=[]
actions_record=[]
environment = gym.make('InvertedPendulum-v2')

print('testing PID controller')
episode_reward=0
states = environment.reset()
terminal=False
integral=0
while not terminal:
    environment.render()
    integral+=states[1]
    actions = kp*states[1]+ki*integral+kd*states[3]
    states, reward, terminal,info = environment.step(actions)
    angle_record.append(states[1])
    velocity_record.append(states[3])
    actions_record.append(actions)
    episode_reward+=reward
print(episode_reward)


length=len(angle_record)
x=range(length)
plt.plot(x,angle_record,label='Angle',color='black')
plt.plot(x,velocity_record,label='Angular Velocity',color='blue',alpha=0.5)
plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 12})
plt.ylim(-0.1,0.1)
plt.savefig('PID.png')
#plt.close()
#plt.plot(x,actions_record,label='actions',color='magenta')
#plt.show()