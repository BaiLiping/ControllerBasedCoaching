from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
episode_number=10000
average_over=100
# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='InvertedDoublePendulum-v2')
'''
    Observation:

        Num    Observation               Min            Max
        0      x_position
        1      sin(theta1)
        2      sin(theta2)
        3      cos(theta1)
        4      cos(theta2)
        5      velocity of x
        6      velocity of theta1
        7      velocity of theta2
        8      constraint force on x
        9      constraint force on theta1
        10     constraint force on theta2

    Action: (-1,1) actuation on the cart
    Terminal State:
        y<=1 can not be observed directly
    Boundary:
       the boundary should be set on theta1 since that is the more
       sensative part of the setup.
       state[1] abs(sin(theta1))<=0.5 +-30 degree
'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-1,-3,-5,-10,-15,-20]
prohibition_position=[0.7,0.9]


reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
theta_threshold=0.5

#compare to agent trained without prohibitive boundary
record=[]
agent = Agent.create(agent='a2c', environment=environment, batch_size=64, learning_rate=1e-2)
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



#with boundary
for k in range(len(prohibition_position)):
    for i in range(len(prohibition_parameter)):
        record=[]
        agent = Agent.create(agent='a2c', environment=environment, batch_size=64, learning_rate=1e-2)
        print('running experiment with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                sinangle=states[1]
                if sinangle>=prohibition_position[k]*theta_threshold:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=1
                elif sinangle<=-prohibition_position[k]*theta_threshold:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=-1
                else:
                    actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
                episode_reward+=reward
            record.append(episode_reward)
        temp=np.array(record)
        reward_record[k][i]=moving_average(temp,average_over)


#save data
pickle.dump( reward_record, open( "double_angle_record.p", "wb"))


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
    plt.savefig('double_with_angle_boundary_at_%s_plot.png' %prohibition_position[i])


agent.close()
environment.close()
