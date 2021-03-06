from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

# thigh_actuator_kp=[-2,-2,-0.5,-1]
# thigh_actuator_kd=[-1.7725,1, 0.2,-0.4]
# leg_actuator_kp=[-0.4,-0.5,-0.1,-0.2]
# leg_actuator_kd=[-1,0.2,-1,-0.1]
# foot_actuator_kp=[-2, 1, 0.5, -1]
# foot_actuator_kd=[-0.4,-0.1,-0.1,-0.5]

thigh_actuator_kp=[-2,-2,-0.5,-1]
thigh_actuator_kd=[-2,1, 0.2,-0.4]
leg_actuator_kp=[-0.4,-0.5,-0.1,-0.2]
leg_actuator_kd=[-1,0.2,-1,-0.1]
foot_actuator_kp=[-2, 1, 0.5, -1]
foot_actuator_kd=[-0.4,-0.1,-0.1,-0.5]

environment = gym.make('Hopper-v3')
print('testing PID controller')
episode_reward=0
states = environment.reset()
terminal=False

y_record=[]
y_velocity=[]
actions_record=[]

x_record=[]
while not terminal:
	environment.render()

	rooty=states[1]
	velocity_rooty=states[7]

	thigh_angle=states[2]
	thigh_angular_velocity=states[8]

	leg_angle=states[3]
	leg_angular_velocity=states[9]

	foot_angle=states[4]
	foot_angular_velocity=states[10]

	thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty+thigh_actuator_kp[1]*thigh_angle+thigh_actuator_kd[1]*thigh_angular_velocity+thigh_actuator_kp[2]*leg_angle+thigh_actuator_kd[2]*leg_angular_velocity+thigh_actuator_kp[3]*foot_angle+thigh_actuator_kd[3]*foot_angular_velocity
	leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty+leg_actuator_kp[1]*thigh_angle+leg_actuator_kd[1]*thigh_angular_velocity+leg_actuator_kp[2]*leg_angle+leg_actuator_kd[2]*leg_angular_velocity+leg_actuator_kp[3]*foot_angle+leg_actuator_kd[3]*foot_angular_velocity
	foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty+foot_actuator_kp[1]*thigh_angle+foot_actuator_kd[1]*thigh_angular_velocity+foot_actuator_kp[2]*leg_angle+foot_actuator_kd[2]*leg_angular_velocity+foot_actuator_kp[3]*foot_angle+foot_actuator_kd[3]*foot_angular_velocity
	actions=[thigh_actions,leg_actions,foot_actions]

	states, reward, terminal,info = environment.step(actions)
	actions_record.append(actions)
	y_record.append(states[1])
	y_velocity.append(states[7])
	x_record.append(states[5])
	episode_reward+=reward

x=range(len(y_record))
fig=plt.figure(figsize=(10,7))
plt.plot(x,y_record,label='Y Position',color='black')
plt.plot(x,y_velocity,label='Y Velocity',color='blue',alpha=0.5)
plt.xlabel('Steps', fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 16})
plt.ylim(-1,1)
#plt.savefig('PID.png')
plt.show()
#plt.plot(x,x_record)
#plt.show()


environment.close()