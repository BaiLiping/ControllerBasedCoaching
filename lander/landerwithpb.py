from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pickle
episode_number=1000
average_over=30
# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='LunarLander-v2')
'''
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Do Nothing
        1     Fire Left Engine
        2     Fire Main Engine
        3     Fire Right Engine

    Observation: Box(-np.inf, np.inf, shape=(8,))

        Type: Box(8)
        Num     Observation
        0       (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)
        1       (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2)
        2       vel.x*(VIEWPORT_W/SCALE/2)/FPS
        3       vel.y*(VIEWPORT_H/SCALE/2)/FPS
        4       Lander Angle
        5       20.0*self.lander.angularVelocity/FPS
        6       Legs[0] Contact with ground
        7       Legs[1] Contact with gound

    Terminal State:
        abs(state[0]) >= 1.0

    Prohibitive Boundary:
       the boundary set around abs(state[0])=0.95
           when x position is greater than 0.05, action 3
           when x position is less than -0.05, action 1
       the angle at abs(22) 0.4 radius
'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

reward_record=np.zeros((6,len(measure_length)))
prohibition_parameter=[-1,-1.5,-2,-2.5,-3,0]

for i in range(5):
    record=[]
    agent = Agent.create(
        agent='a2c', environment=environment, batch_size=64, learning_rate=1e-2
    )
    for _ in range(episode_number):
        episode_reward=0
        states = environment.reset()
        x_position=states[0]
        angle_position=states[4]
        terminal = False
        while not terminal:

            if angle_position>=.1:
                episode_reward+= prohibition_parameter[i]
                actions = agent.act(states=states)
                actions=3

            elif angle_position<=-.1:
                episode_reward+= prohibition_parameter[i]
                actions = agent.act(states=states)
                actions=1
            else:
                if x_position>=0.05:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=1
                elif x_position<=-0.05:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=3
                else:
                    actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            episode_reward+=reward
        record.append(episode_reward)

    temp=np.array(record)
    reward_record[i]=moving_average(temp,average_over)




#compare to agent trained without prohibitive boundary
agent = Agent.create(
    agent='a2c', environment=environment, batch_size=64, learning_rate=1e-2
)

states=environment.reset()
terminal=False
record=[]
for _ in range(episode_number):
    episode_reward=0
    states = environment.reset()
    terminal = False
    reward=0
    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward+=reward
        agent.observe(terminal=terminal, reward=reward)
    record.append(episode_reward)
    print(episode_reward)
temp=np.array(record)
reward_record[5]=moving_average(temp,average_over)
#plot results
color_scheme=['green','orange','red','yellow','yellowgreen','black']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure()
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(loc="upper left")
    plt.savefig('cartpole_with_angle_boundary_at_%s_plot.png' %prohibition_position[i])

agent.close()
environment.close()
