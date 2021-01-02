from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
from Normal import episode_number
from Normal import average_over
from Normal import evaluation_episode_number
from Normal import exploration
from Normal import measure_length
from Normal import moving_average
from Normal import prohibition_parameter
from Normal import prohibition_position
from Normal import environment

kp=25
kd=3
#training with coaching
reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),episode_number))
evaluation_reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),evaluation_episode_number))
for k in range(len(prohibition_position)):
    #training
    for i in range(len(prohibition_parameter)):
        record=[]
        angle_record=[]
        agent = Agent.create(agent='agent.json', environment=environment)
        print('training agent with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                theta=states[1]
                angular_velocity=states[3]
                theta_states=[theta,angular_velocity]
                actions = agent.act(states=states)
                if abs(theta)>=prohibition_position[k]:
                    #actions=kp*theta+kd*angular_velocity
                    states, terminal, reward = environment.execute(actions=actions)
                    reward=0
                    episode_reward+=reward
                    agent.observe(terminal=terminal, reward=reward)
                else:
                    states, terminal, reward = environment.execute(actions=actions)
                    agent.observe(terminal=terminal, reward=reward)
                    episode_reward+=reward
                #print('action ',actions)
            record.append(episode_reward)
            #print(episode_reward)
        reward_record[k][i]=record
        temp=np.array(record)
        reward_record_average[k][i]=moving_average(temp,average_over)
        x_angle=range(len(angle_record))
        plt.figure(figsize=(30,10))
        plt.plot(x_angle,angle_record)
        plt.savefig('angle%s%s.png' %(k,i))

        #evaluate
        episode_reward = 0.0
        eva_reward_record=[]
        print('evaluating agent with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for j in tqdm(range(evaluation_episode_number)):
            episode_reward=0
            states = environment.reset()
            internals = agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
                states, terminal, reward = environment.execute(actions=actions)
                episode_reward += reward
            eva_reward_record.append(episode_reward)
        evaluation_reward_record[k][i]=eva_reward_record
        agent.close()
#save data
pickle.dump(reward_record, open( "control_record.p", "wb"))
pickle.dump(reward_record_average, open( "control_average_record.p", "wb"))
pickle.dump(evaluation_reward_record, open( "control_evaluation_record.p", "wb"))
environment.close()