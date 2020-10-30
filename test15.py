from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import pickle
episode_number=400
# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPoleBLP-v0', max_episode_timesteps=1000)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=64, learning_rate=1e-2
)


reward_record=[]
episode_reward=0
# Train for 300 episodes
for _ in range(episode_number):

    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward+=reward
        #print(episode_reward)
        agent.observe(terminal=terminal, reward=reward)

    reward_record.append(episode_reward)
    print(episode_reward)
    episode_reward=0


x=range(episode_number)
plt.plot(x,reward_record)
plt.show()
pickle.dump( reward_record, open( "blp15.p", "wb"))

agent.close()
environment.close()
