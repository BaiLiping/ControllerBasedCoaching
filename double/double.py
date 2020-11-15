from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
episode_number=40000
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
       The most important element for keeping the balance of a double
       pendulum is the angle between the two poles. Observation has
       data on sin(theta1) and sin(theta2) and from there we can
       derive the actual angle between the poles (math.pi-theta1+theta2)
       then math.pi-(math.pi-theta1+theta2)=theta1-theta2 which is the
       information on how pointing the angle is. We want theta1-theta2
       to be as small as possible
'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-1,-3,-5]
prohibition_position=[0.1,0.2,0.3,0.4,0.5]


reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))

#compare to agent trained without prohibitive boundary
record=[]
agent = Agent.create(agent='agent.json', environment=environment)
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
pickle.dump(reward_record_without, open( "double_without_record.p", "wb"))



#with boundary
for k in range(len(prohibition_position)):
    for i in range(len(prohibition_parameter)):
        record=[]
        agent = Agent.create(agent='agent.json', environment=environment)
        print('running experiment with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                sintheta1=states[1]
                sintheta2=states[2]
                theta1=math.asin(sintheta1)
                theta2=math.asin(sintheta2)
                angle=theta1-theta2
                if angle>=prohibition_position[k]*math.pi:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=1
                elif angle<=-prohibition_position[k]*math.pi:
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
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('double_with_angle_boundary_at_%s_plot.png' %prohibition_position[i])


agent.close()
environment.close()
