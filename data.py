from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np

episode_number=1000
y=np.random.randint(10,size=episode_number)
print(y)

x=range(episode_number)
plt.plot(x,y)
plt.show()
