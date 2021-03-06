from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm


#setparameters
num_steps=10 #update exploration rate over n steps
initial_value=0.95 #initial exploartion rate
decay_rate=0.5 #exploration rate decay rate
set_type='exponential' #set the type of decay linear, exponential
exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)

episode_number=2000
evaluation_episode_number=5


# Pre-defined or custom environment
environment = Environment.create(environment='gym', level='InvertedDoublePendulum-v2')

length=np.zeros(episode_number)

reward_record_without=[]

agent_without = Agent.create(agent='agent.json', environment=environment,exploration=exploration)
states=environment.reset()
terminal = False
print('training agent without boundary')
angle_record=[]
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal= False
    while not terminal:

        actions = agent_without.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        sintheta1=states[1]
        sintheta2=states[2]
        theta1=math.asin(sintheta1)
        theta2=math.asin(sintheta2)
        angle=theta1-theta2

        if abs(angle)>0.5:
            reward=10
        else:
            reward=0.5

        #reward-=abs(actions)
        episode_reward+=reward
        angle_record.append(states[1])
        agent_without.observe(terminal=terminal, reward=reward)
    reward_record_without.append(episode_reward)
agent_without.save(directory='model3', format='numpy')
x=range(episode_number)
x_angle=range(len(angle_record))
plt.figure(figsize=(30,10))
plt.plot(x,reward_record_without,label='without prohibitive boundary',color='black')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
plt.savefig('plot1.png')

plt.figure(figsize=(30,10))
plt.plot(x_angle,angle_record)
plt.savefig('angle1.png')

#evaluate the agent without Boundary
episode_reward = 0.0
evaluation_reward_record_without=[]
print('evaluating agent without boundary')
for _ in tqdm(range(evaluation_episode_number)):
    episode_reward=0
    states = environment.reset()
    internals = agent_without.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent_without.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record_without.append(episode_reward)
pickle.dump(evaluation_reward_record_without, open( "evaluation_without_record.p", "wb"))
environment.close()
