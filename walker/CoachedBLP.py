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
from Normal import environment

reward_record=[]
evaluation_reward_record=[]

thigh_actuator_kp=[-0.6]
thigh_actuator_kd=[0.03]
leg_actuator_kp=[-0.7]
leg_actuator_kd=[-0.21]
foot_actuator_kp=[-0.9]
foot_actuator_kd=[-0.13]
left_thigh_actuator_kp=[-0.73]
left_thigh_actuator_kd=[-0.18]
left_leg_actuator_kp=[-1.29]
left_leg_actuator_kd=[-0.25]
left_foot_actuator_kp=[-1.07]
left_foot_actuator_kd=[-0.15]

#training with coaching
agent = Agent.create(agent='agent.json', environment=environment)
print('Training Agent with PID Coaching')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal = False
    count=0
    positive=0
    consecutive=[0]
    y_velocity_batch=[0]
    cons_position=0

    while not terminal:
        count+=1
        print(count)
        rootz=states[0]
        velocity_rootz=states[9]

        rooty=states[1]
        velocity_rooty=states[10]

        thigh_angle=states[2]
        thigh_angular_velocity=states[11]

        leg_angle=states[3]
        leg_angular_velocity=states[12]

        foot_angle=states[4]
        foot_angular_velocity=states[13]

        left_thigh_angle=states[5]
        left_thigh_angular_velocity=states[14]

        left_leg_angle=states[6]
        left_leg_angular_velocity=states[15]

        left_foot_angle=states[7]
        left_foot_angular_velocity=states[16]

        if abs(velocity_rooty)<=1.3 and abs(velocity_rooty)>=1:
            print('before:', velocity_rooty)
            thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty
            leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty
            foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty
            left_thigh_actions = left_thigh_actuator_kp[0]*rooty+left_thigh_actuator_kd[0]*velocity_rooty
            left_leg_actions = left_leg_actuator_kp[0]*rooty+left_leg_actuator_kd[0]*velocity_rooty
            left_foot_actions = left_foot_actuator_kp[0]*rooty+left_foot_actuator_kd[0]*velocity_rooty
            intervention=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]                                   
            print('intervention:',intervention)
            states, terminal, reward = environment.execute(actions=intervention)
            print('after', states[10])
        else:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            episode_reward+=reward
    
    reward_record.append(episode_reward)

#evaluate
print('Evaluating Agent with PID Coaching')
episode_reward = 0.0
for j in tqdm(range(evaluation_episode_number)):
    episode_reward=0
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record.append(episode_reward)

agent.close()
#save data
pickle.dump(reward_record, open( "walker_record.p", "wb"))
pickle.dump(evaluation_reward_record, open( "walker_evaluation_record.p", "wb"))
environment.close()