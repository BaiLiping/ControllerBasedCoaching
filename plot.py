import matplotlib.pyplot as plt
import pickle
import numpy as np

episode_number=400

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

reward_record_blp=pickle.load(open( "blp.p", "rb"))
reward_record_blp=np.array(reward_record_blp)
reward_record_blp_av=moving_average(reward_record_blp,30)

reward_record_blp10=pickle.load(open("blp10.p","rb"))
reward_record_blp10=np.array(reward_record_blp10)
reward_record_blp10_av=moving_average(reward_record_blp10,30)
reward_record_blp15=pickle.load(open("blp15.p","rb"))
reward_record_blp15=np.array(reward_record_blp15)
reward_record_blp15_av=moving_average(reward_record_blp15,30)
reward_record_blp25=pickle.load(open("blp25.p","rb"))
reward_record_blp25=np.array(reward_record_blp25)
reward_record_blp25_av=moving_average(reward_record_blp25,30)
reward_record_blp30=pickle.load(open("blp30.p","rb"))
reward_record_blp30=np.array(reward_record_blp30)
reward_record_blp30_av=moving_average(reward_record_blp30,30)
reward_record_original=pickle.load(open('original.p','rb'))
reward_record_original=np.array(reward_record_original)
reward_record_original_av=moving_average(reward_record_original,30)

x=range(371)
plt.plot(x,reward_record_blp10_av,label='with training wheel-10',color='green')
plt.plot(x,reward_record_blp15_av,label='with training wheel-15',color='orange')
plt.plot(x,reward_record_blp_av,label='with training wheel-20',color='red')
plt.plot(x,reward_record_blp25_av,label='with training wheel-25',color='lime')
plt.plot(x,reward_record_blp30_av,label='with training wheel-30',color='yellowgreen')
plt.plot(x,reward_record_original_av,label='without training wheel',color='black')
plt.legend(loc="upper left")
plt.savefig('plot.png')
