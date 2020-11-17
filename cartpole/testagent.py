from tensorforce import Agent, Environment, Runner
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

agent = Agent.load(directory='without', format='saved-model', environment=environment)
runner = Runner(agent=agent, environment=environment)
runner.run(num_episodes=episode_number, evaluation=True)
'''
print('test agent without boundary')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal= False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward+=reward
    record.append(episode_reward)
reward_record=record
pickle.dump(reward_record, open( "cartpole_agent_record.p", "wb"))

#plot
x=range(episode_number)
plt.figure(figsize=(20,10))
plt.plot(x,reward_record,label='evaluate agent trained without boundary',color='black')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
plt.savefig('cartpole_agent_eval.png')

'''




runner.close()
agent.close()










# Close environment separately, since created separately
environment.close()
