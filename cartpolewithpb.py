from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
episode_number=400
average_over=20
# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole-v1', max_episode_timesteps=1000)
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

        self.theta_threshold_radians = 12 * 2 * math.pi / 360

'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

reward_record=np.zeros((6,len(measure_length)))
theta_threshold_radians = 12 * 2 * math.pi / 360
prohibition_parameter=[-10,-15,-20,-25,-30,0]


for i in range(5):
    record=[]
    agent = Agent.create(
        agent='ppo', environment=environment, batch_size=64, learning_rate=1e-2
    )
    for _ in range(episode_number):
        episode_reward=0
        states = environment.reset()
        terminal = False
        while not terminal:
            angle=states[2]
            if angle>=.95*theta_threshold_radians:
                episode_reward+= prohibition_parameter[i]
                actions = agent.act(states=states)
                actions=0
            elif angle<=-.95*theta_threshold_radians:
                episode_reward+= prohibition_parameter[i]
                actions = agent.act(states=states)
                actions=1
            else:
                actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            episode_reward+=reward
        record.append(episode_reward)
        print(episode_reward)
    temp=np.array(record)
    reward_record[i]=moving_average(temp,20)


#compare to agent trained without prohibitive boundary
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=64, learning_rate=1e-2
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
color_scheme=['green','orange','red','lime','yellowgreen','black']
x=range(len(measure_length))
for j in range(6):
    plt.plot(x,reward_record[j],label=str(prohibition_parameter[j]),color=color_scheme[j])
plt.show()
plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend(loc="upper left")
plt.savefig('plot.png')

pickle.dump( reward_record, open( "record.p", "wb"))
agent.close()
environment.close()
