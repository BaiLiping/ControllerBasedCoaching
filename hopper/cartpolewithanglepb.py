from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
episode_number=40000
average_over=50
# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='Hopper-v3')
'''
    Terminal State:
        healthy_z_range=(0.7, float('inf'))
        healthy_angle_range=(-0.2, 0.2)
'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-5,-10,-15,-20,-25,-30]
prohibition_position=[0.1,0.3,0.5,0.7,0.9]


reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
theta_threshold_radians=0.2

for k in range(len(prohibition_position)):
    for i in range(len(prohibition_parameter)):
        record=[]
        agent = Agent.create(agent='ppo', environment=environment, batch_size=64, learning_rate=1e-2)
        print('running experiment with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                angle=states[2]
                if angle>=prohibition_position[k]*theta_threshold_radians:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=[1,1,-1]
                elif angle<=-prohibition_position[k]*theta_threshold_radians:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=[1,1,-1]
                else:
                    actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
                episode_reward+=reward
            record.append(episode_reward)
        temp=np.array(record)
        reward_record[k][i]=moving_average(temp,average_over)

#compare to agent trained without prohibitive boundary
record=[]
agent = Agent.create(agent='ppo', environment=environment, batch_size=64, learning_rate=1e-2)
states=environment.reset()
terminal = False
print('running experiment without boundary')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal= False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward+=reward
        agent.observe(terminal=terminal, reward=reward)
    record.append(episode_reward)
temp=np.array(record)
reward_record_without=moving_average(temp,average_over)

#save data
pickle.dump( reward_record, open( "cartpole_angle_record.p", "wb"))


#plot results
color_scheme=['green','orange','red','blue','yellowgreen','magenta','cyan']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure()
    plt.plot(x,reward_record_without,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(loc="upper left")
    plt.savefig('hopper_with_angle_boundary_at_%s_plot.png' %prohibition_position[i])


agent.close()
environment.close()
