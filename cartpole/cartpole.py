from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole-v1', max_episode_timesteps=500)
'''
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Terminal State:
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
'''
# Intialize reward record and set parameters
#define the length of the vector
episode_number=100
evaluation_episode_number=100
average_over=1
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[-15,-20,-25,-30]
prohibition_position=[0.5,0.7,0.9,0.95]


theta_threshold_radians=12*2*math.pi/360

#compare to agent trained without prohibitive boundary


#training of agent without prohibitive boundary
reward_record_without=[]
agent = Agent.create(agent='agent.json', environment=environment)
states=environment.reset()
terminal = False
print('training agent without boundary')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal= False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward+=reward
        agent.observe(terminal=terminal, reward=reward)
    reward_record_without.append(episode_reward)
temp=np.array(reward_record_without)
reward_record_without_average=moving_average(temp,average_over)
pickle.dump(reward_record_without_average, open( "cartpole_without_average_record.p", "wb"))
pickle.dump(reward_record_without, open( "cartpole_without_record.p", "wb"))

#evaluate the agent without Boundary
episode_reward = 0.0
evaluation_reward_record_without=[]
print('evaluating agent without boundary')
for _ in tqdm(range(evaluation_episode_number)):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(
            states=states, internals=internals, independent=True, deterministic=True
        )
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record_without.append(episode_reward)
pickle.dump(evaluation_reward_record_without, open( "evaluation_cartpole_without_record.p", "wb"))

#save agent and close agent
agent.save(directory='without', format='saved-model')
agent.close()

#training and evaluation with boundary
reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),episode_number))
evaluation_reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),evaluation_episode_number))

for k in range(len(prohibition_position)):
    #training
    for i in range(len(prohibition_parameter)):
        record=[]
        agent = Agent.create(agent='agent.json', environment=environment)
        print('training agent with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                angle=states[2]
                if angle>=prohibition_position[k]*theta_threshold_radians:
                    actions = agent.act(states=states)
                    actions = 0
                    states, terminal, reward = environment.execute(actions=actions)
                    reward+= prohibition_parameter[i]
                    agent.observe(terminal=terminal, reward=reward)
                elif angle<=-prohibition_position[k]*theta_threshold_radians:
                    actions = agent.act(states=states)
                    actions = 1
                    states, terminal, reward = environment.execute(actions=actions)
                    reward+= prohibition_parameter[i]
                    agent.observe(terminal=terminal, reward=reward)
                else:
                    actions = agent.act(states=states)
                    states, terminal, reward = environment.execute(actions=actions)
                    agent.observe(terminal=terminal, reward=reward)

                episode_reward+=reward
            record.append(episode_reward)
        reward_record[k][i]=record
        temp=np.array(record)
        reward_record_average[k][i]=moving_average(temp,average_over)

        #evaluate
        episode_reward = 0.0
        eva_reward_record=[]
        print('evaluating agent with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for j in tqdm(range(evaluation_episode_number)):
            states = environment.reset()
            internals = agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
                states, terminal, reward = environment.execute(actions=actions)
                episode_reward += reward
            eva_reward_record.append(episode_reward)
        evaluation_reward_record[k][i]=eva_reward_record

        agent.save(directory='%s %s' %(k,i) , format='saved-model')
        agent.close()

#save data
pickle.dump(reward_record, open( "cartpole_record.p", "wb"))
pickle.dump(reward_record_average, open( "cartpole_average_record.p", "wb"))
pickle.dump(evaluation_reward_record, open( "evaluation_cartpole_record.p", "wb"))


#plot training results
color_scheme=['yellowgreen','magenta','green','orange','red','blue','cyan']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure(figsize=(20,10))
    plt.plot(x,reward_record_without_average,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record_average[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
    plt.savefig('cartpole_with_boundary_at_%s_plot.png' %prohibition_position[i])

#plot evaluation results
x=range(len(evaluation_episode_number))
for i in range(len(prohibition_position)):
    plt.figure(figsize=(20,10))
    plt.plot(x,evaluation_reward_record_without,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,evaluation_reward_record[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
    plt.savefig('evaluate_cartpole_with_boundary_at_%s_plot.png' %prohibition_position[i])

environment.close()
