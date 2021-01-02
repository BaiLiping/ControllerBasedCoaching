from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym


test_episodes=20

record=[]
environment = gym.make('Walker2d-v3')


# thigh_actuator_kp=[-1,-1.5,3,0.013,-0.5,3,-0.5]
# thigh_actuator_kd=[-0.02,-0.2,-0.2,0.1,0.2,-0.1,0.2]
# leg_actuator_kp=[-0.3, 1,3,0.3,-0.2, 3,-0.1]
# leg_actuator_kd=[0,-0.1,-0.2,0,0,0,0]
# foot_actuator_kp=[-0.2,  1.5,  3, 0,  0,  3,  0.2]
# foot_actuator_kd=[0.1, -0.2, 0,0,  0.2,  0.2,-0.2]

# left_thigh_actuator_kp=[1,-0.5,2.5,-0.2,-0.2, 2.8,-0.1]
# left_thigh_actuator_kd=[1,0,-0.1,-0.1,-0.1,-0.3, 0]
# left_leg_actuator_kp=[1.39339357e-02,6.85548175e-01,  1.11287125e+01,
#    1.71331481e-01, -2.07591073e-01 , 1.69707380e+01 ,-2.32905986e-01]
# left_leg_actuator_kd=[2.51231437e-03, -9.84687891e-02,
#   -1.28668081e-01,  1.15375851e-02,  9.08837213e-02, -7.99763370e-02,
#    7.43069003e-02]
# left_foot_actuator_kp=[-4.03056328e-01,  3.47425549e-01, -3.18646118e+00,
#    1.29329722e-01,  2.66864632e-01, -3.08932691e+00,  4.45821520e-01]
# left_foot_actuator_kd=[-2.85770739e-02, -1.11864397e-02,
#    5.39661583e-02,  8.23052299e-03,  2.04883711e-02,  2.06209261e-01,
#   -1.54434984e-01]


# thigh_actuator_kp=[1.73397369,-2,-0.5,-1,2,0.5,1]
# thigh_actuator_kd=[3.28081217,1, 0.2,-0.4,1, 0.2,-0.4]
# leg_actuator_kp=[1.38968547,-0.5,-0.1,-0.2,-0.5,-0.1,-0.2]
# leg_actuator_kd=[2.39102281,0.2,-1,-0.1,0.2,-1,-0.1]
# foot_actuator_kp=[1.7187277, 1, 0.5, -1,1, 0.5, -1]
# foot_actuator_kd=[2.55930771,-0.1,-0.1,-0.5,-0.1,-0.1,-0.5]
# left_thigh_actuator_kp=[-0.39722752,-2,-0.5,-1,2,0.5,1]
# left_thigh_actuator_kd=[0.10559088,1, 0.2,-0.4,1, 0.2,-0.4]
# left_leg_actuator_kp=[1.20503534,-0.5,-0.1,-0.2,-0.5,-0.1,-0.2]
# left_leg_actuator_kd=[ 2.26925593,0.2,-1,-0.1,0.2,-1,-0.1]
# left_foot_actuator_kp=[0.10300404, 1, 0.5, -1,1, 0.5, -1]
# left_foot_actuator_kd=[-0.30718452,-0.1,-0.1,-0.5,-0.1,-0.1,-0.5]

# thigh_actuator_kp=[-3,-2]
# thigh_actuator_kd=[-2,1]
# leg_actuator_kp=[-.5,-0.5]
# leg_actuator_kd=[-.5,0.2]
# foot_actuator_kp=[-.5, 1]
# foot_actuator_kd=[-.5,-0.1]
# left_thigh_actuator_kp=[-3,-2]
# left_thigh_actuator_kd=[-2,1]
# left_leg_actuator_kp=[-.5,-0.5]
# left_leg_actuator_kd=[-.5,0.2]
# left_foot_actuator_kp=[-0.5, 1]
# left_foot_actuator_kd=[-0.5,-0.1]

# thigh_actuator_kp=[-1.05999489,2.51772413]
# thigh_actuator_kd=[0.96885614,-0.83131062]
# leg_actuator_kp=[-0.52827962,1.86832832]
# leg_actuator_kd=[0.75822465,-0.58596585]
# foot_actuator_kp=[2.54100475, 2.12025646]
# foot_actuator_kd=[3.01929226,0.62454677]
# left_thigh_actuator_kp=[-1.38440953,-0.10243073]
# left_thigh_actuator_kd=[-0.57606987,-0.36981063]
# left_leg_actuator_kp=[-0.55181508,1.7567481]
# left_leg_actuator_kd=[0.78574238,-0.51248579]
# left_foot_actuator_kp=[1.72123472, -0.02385711]
# left_foot_actuator_kd=[0.93901675,0.66120021]


# thigh_actuator_kp=[-1.8244548]
# leg_actuator_kp= [-1.19699651]
# foot_actuator_kp=[-1.05320778]
# left_thigh_actuator_kp=[-0.50129912]
# left_leg_actuator_kp=[-1.24195404]
# left_foot_actuator_kp=[ 0.40596119]


'''
thigh_actuator_kp=[-7]
leg_actuator_kp= [-3]
foot_actuator_kp=[-2]
left_thigh_actuator_kp=[-7]
left_leg_actuator_kp=[-3]
left_foot_actuator_kp=[-2]
#rooty average:  365
'''

'''
thigh_actuator_kp=[-2.36784261]
leg_actuator_kp= [-1.62446179]
foot_actuator_kp=[-1.5601286 ]
left_thigh_actuator_kp=[-0.51640433]
left_leg_actuator_kp=[-1.62384398]
left_foot_actuator_kp=[ 0.3881404 ]
# rooty average:  476
'''
'''
# thigh_actuator_kp=[7,-2.36]
# leg_actuator_kp=  [5  , -1.6]
# foot_actuator_kp=[4.5 , -1.5 ]
# left_thigh_actuator_kp=[ 1.7 , -0.5]
# left_leg_actuator_kp=[5,  -1.6]
# left_foot_actuator_kp=[-1.48, 0.4]
# #1.25-rootz, rooty average:  353
'''
'''
thigh_actuator_kp=[ 9.09254081  ,0.80444947]
leg_actuator_kp=   [ 6.6320304 ,  0.72336937]
foot_actuator_kp= [ 6.89925402 , 0.93421442]
left_thigh_actuator_kp= [ 0.72111691 ,-0.30132071]
left_leg_actuator_kp= [ 6.82259363 , 0.72653893]
left_foot_actuator_kp= [-1.95860697 ,-0.14842815]
#1.25-rootz, rooty average:  97
'''
'''
thigh_actuator_kp=[7 ,0 ]
leg_actuator_kp= [5, 0]
foot_actuator_kp=[4.5 , 0]
left_thigh_actuator_kp=[1.7,0]
left_leg_actuator_kp=[5 ,0]
left_foot_actuator_kp=[ -1.48 ,0]
#1.25-rootz average:  388
'''
thigh_actuator_kp=[7 ,0.001]
leg_actuator_kp= [5, -0.002]
foot_actuator_kp=[4.5 , 0.005]
left_thigh_actuator_kp=[1.7,0.003]
left_leg_actuator_kp=[5 ,-0.01]
left_foot_actuator_kp=[ -1.48 ,0.001]


for i in range(test_episodes):
	episode_reward=0
	states = environment.reset()
	terminal=False
	while not terminal:

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

		#thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty+thigh_actuator_kp[1]*thigh_angle+thigh_actuator_kd[1]*thigh_angular_velocity+thigh_actuator_kp[2]*leg_angle+thigh_actuator_kd[2]*leg_angular_velocity+thigh_actuator_kp[3]*foot_angle+thigh_actuator_kd[3]*foot_angular_velocity+thigh_actuator_kp[4]*left_thigh_angle+thigh_actuator_kd[4]*left_thigh_angular_velocity+thigh_actuator_kp[5]*left_leg_angle+thigh_actuator_kd[5]*left_leg_angular_velocity+thigh_actuator_kp[6]*left_foot_angle+thigh_actuator_kd[6]*left_foot_angular_velocity
		#leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty+leg_actuator_kp[1]*thigh_angle+leg_actuator_kd[1]*thigh_angular_velocity+leg_actuator_kp[2]*leg_angle+leg_actuator_kd[2]*leg_angular_velocity+leg_actuator_kp[3]*foot_angle+leg_actuator_kd[3]*foot_angular_velocity+leg_actuator_kp[4]*left_thigh_angle+leg_actuator_kd[4]*left_thigh_angular_velocity+leg_actuator_kp[5]*left_leg_angle+leg_actuator_kd[5]*left_leg_angular_velocity+leg_actuator_kp[6]*left_foot_angle+leg_actuator_kd[6]*left_foot_angular_velocity
		#foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty+foot_actuator_kp[1]*thigh_angle+foot_actuator_kd[1]*thigh_angular_velocity+foot_actuator_kp[2]*leg_angle+foot_actuator_kd[2]*leg_angular_velocity+foot_actuator_kp[3]*foot_angle+foot_actuator_kd[3]*foot_angular_velocity+foot_actuator_kp[4]*left_thigh_angle+foot_actuator_kd[4]*left_thigh_angular_velocity+foot_actuator_kp[5]*left_leg_angle+foot_actuator_kd[5]*left_leg_angular_velocity+foot_actuator_kp[6]*left_foot_angle+foot_actuator_kd[6]*left_foot_angular_velocity
		#left_thigh_actions = left_thigh_actuator_kp[0]*rooty+left_thigh_actuator_kd[0]*velocity_rooty+left_thigh_actuator_kp[1]*thigh_angle+left_thigh_actuator_kd[1]*thigh_angular_velocity+left_thigh_actuator_kp[2]*leg_angle+left_thigh_actuator_kd[2]*leg_angular_velocity+left_thigh_actuator_kp[3]*foot_angle+left_thigh_actuator_kd[3]*foot_angular_velocity+left_thigh_actuator_kp[4]*left_thigh_angle+left_thigh_actuator_kd[4]*left_thigh_angular_velocity+left_thigh_actuator_kp[5]*left_leg_angle+left_thigh_actuator_kd[5]*left_leg_angular_velocity+left_thigh_actuator_kp[6]*left_foot_angle+left_thigh_actuator_kd[6]*left_foot_angular_velocity
		#left_leg_actions = left_leg_actuator_kp[0]*rooty+left_leg_actuator_kd[0]*velocity_rooty+left_leg_actuator_kp[1]*thigh_angle+left_leg_actuator_kd[1]*thigh_angular_velocity+left_leg_actuator_kp[2]*leg_angle+left_leg_actuator_kd[2]*leg_angular_velocity+left_leg_actuator_kp[3]*foot_angle+left_leg_actuator_kd[3]*foot_angular_velocity+left_leg_actuator_kp[4]*left_thigh_angle+left_leg_actuator_kd[4]*left_thigh_angular_velocity+left_leg_actuator_kp[5]*left_leg_angle+left_leg_actuator_kd[5]*left_leg_angular_velocity+left_leg_actuator_kp[6]*left_foot_angle+left_leg_actuator_kd[6]*left_foot_angular_velocity
		#left_foot_actions = left_foot_actuator_kp[0]*rooty+left_foot_actuator_kd[0]*velocity_rooty+left_foot_actuator_kp[1]*thigh_angle+left_foot_actuator_kd[1]*thigh_angular_velocity+left_foot_actuator_kp[2]*leg_angle+left_foot_actuator_kd[2]*leg_angular_velocity+left_foot_actuator_kp[3]*foot_angle+left_foot_actuator_kd[3]*foot_angular_velocity+left_foot_actuator_kp[4]*left_thigh_angle+left_foot_actuator_kd[4]*left_thigh_angular_velocity+left_foot_actuator_kp[5]*left_leg_angle+left_foot_actuator_kd[5]*left_leg_angular_velocity+left_foot_actuator_kp[6]*left_foot_angle+left_foot_actuator_kd[6]*left_foot_angular_velocity
		#actions=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]
		#thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty+thigh_actuator_kp[1]*thigh_angle+thigh_actuator_kd[1]*thigh_angular_velocity
		#leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty+leg_actuator_kp[1]*thigh_angle+leg_actuator_kd[1]*thigh_angular_velocity
		#foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty+foot_actuator_kp[1]*thigh_angle+foot_actuator_kd[1]*thigh_angular_velocity
		#left_thigh_actions = left_thigh_actuator_kp[0]*rooty+left_thigh_actuator_kd[0]*velocity_rooty+left_thigh_actuator_kp[1]*thigh_angle+left_thigh_actuator_kd[1]*thigh_angular_velocity
		#left_leg_actions = left_leg_actuator_kp[0]*rooty+left_leg_actuator_kd[0]*velocity_rooty+left_leg_actuator_kp[1]*thigh_angle+left_leg_actuator_kd[1]*thigh_angular_velocity
		#left_foot_actions = left_foot_actuator_kp[0]*rooty+left_foot_actuator_kd[0]*velocity_rooty+left_foot_actuator_kp[1]*thigh_angle+left_foot_actuator_kd[1]*thigh_angular_velocity
		#actions=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]

		# thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty
		# leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty
		# foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty
		# left_thigh_actions = left_thigh_actuator_kp[0]*rooty+left_thigh_actuator_kd[0]*velocity_rooty
		# left_leg_actions = left_leg_actuator_kp[0]*rooty+left_leg_actuator_kd[0]*velocity_rooty
		# left_foot_actions = left_foot_actuator_kp[0]*rooty+left_foot_actuator_kd[0]*velocity_rooty
		# actions=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]                                   

		# thigh_actions = thigh_actuator_kp[0]*rooty
		# leg_actions = leg_actuator_kp[0]*rooty
		# foot_actions = foot_actuator_kp[0]*rooty
		# left_thigh_actions = left_thigh_actuator_kp[0]*rooty
		# left_leg_actions = left_leg_actuator_kp[0]*rooty
		# left_foot_actions = left_foot_actuator_kp[0]*rooty
		# actions=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]                                   


		# thigh_actions = thigh_actuator_kp[0]*(1.25-rootz)+thigh_actuator_kp[1]*rooty
		# leg_actions = leg_actuator_kp[0]*(1.25-rootz)+leg_actuator_kp[1]*rooty
		# foot_actions = foot_actuator_kp[0]*(1.25-rootz)+foot_actuator_kp[1]*rooty
		# left_thigh_actions = left_thigh_actuator_kp[0]*(1.25-rootz)+left_thigh_actuator_kp[1]*rooty
		# left_leg_actions = left_leg_actuator_kp[0]*(1.25-rootz)+left_leg_actuator_kp[1]*rooty
		# left_foot_actions = left_foot_actuator_kp[0]*(1.25-rootz)+left_foot_actuator_kp[1]*rooty
		# actions=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]                                   

		thigh_actions = thigh_actuator_kp[0]*(1.25-rootz)+thigh_actuator_kp[1]*velocity_rootz
		leg_actions = leg_actuator_kp[0]*(1.25-rootz)+leg_actuator_kp[1]*velocity_rootz
		foot_actions = foot_actuator_kp[0]*(1.25-rootz)+foot_actuator_kp[1]*velocity_rootz
		left_thigh_actions = left_thigh_actuator_kp[0]*(1.25-rootz)+left_thigh_actuator_kp[1]*velocity_rootz
		left_leg_actions = left_leg_actuator_kp[0]*(1.25-rootz)+left_leg_actuator_kp[1]*velocity_rootz
		left_foot_actions = left_foot_actuator_kp[0]*(1.25-rootz)+left_foot_actuator_kp[1]*velocity_rootz
		actions=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]                                   



		#print('actions',actions)
		states, reward, terminal,info = environment.step(actions)
		#print('states: ',states)
		print('thigh_angle: %s velocity: %s'%(states[2],states[11]))
		
		episode_reward+=reward
	record.append(episode_reward)
	print(episode_reward)
environment.close()
ave=int(np.sum(record)/test_episodes)
print('average: ',ave)