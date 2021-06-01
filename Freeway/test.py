""" Test file for nudging and deep R-learning for the Freeway environment

The final models to evaluate muest be in the folder ./models/

This script requires the following packages be installed within the Python 
environment you are running this script in.
    * numpy
    * matplotlib
    * opencv
    * gym

"""

import os
# comment if not using GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
import numpy as np
sys.path.append('./deepq')
sys.path.append('./utils')
from utils.atari_wrappers import make_atari
from deepq.dqn import DQN
from tqdm import tqdm

# --------- remove extra verbosity ---------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['KMP_WARNINGS'] = 'off'
# ------------------------------------------

# create enviroment
env = make_atari('FreewayNoFrameskip-v0')
# resent environment
obs = env.reset()

# path to the model to evaluate for optimal nudging
model_name = './models/freeway_opt_nudging'

# path to the model to evaluate for deep r-learning
# model_name = './models/freeway_deep_rlearning'

# load model
model = DQN.load(load_path=f'{model_name}', env=env)

# number of tests to run
num_test = 100


# placeholders
episode_lenght = []
wins = []
episode_reward = []
cause = []

reward = 0.0
steps = 0
episode = 0
done = False


# TESTING
# while episode<num_test:
for episode in tqdm(range(num_test)):
    # get the recurren state sI
    for _ in range(500):
        obs, _, _, _ = env.step(0)
    while(not done):            
        # apply policy
        action, _states = model.predict(obs)
        obs_new, rew, dones, info = env.step(action)    
        # compute custom reward for this problem
        rew_, done, cause_end = model.compute_reward_test(obs, rew, steps)
        reward = reward + rew_
        steps = steps + 1
        obs = obs_new
    if done:
        if cause_end == 'cross':            
            wins.append(1)
            cause.append(1)
        elif cause_end == 'timeout':
            wins.append(0)
            cause.append(2)
        else:
            wins.append(0)
            cause.append(3)
        episode_lenght.append(steps-1)
        episode_reward.append(reward)
        # episode += 1
        steps = 0
        reward = 0.0
        done = False
        obs = env.reset() 

# Report
d = np.mean(episode_lenght)
Pw = np.sum(wins)/num_test
print('')
print('REPORT OVER 1000 EPISODES:')
print(f'Average duration, d: {d}')
print(f'Probability of crossing, Pw: {Pw}')
print(f'Episode reward: {np.mean(episode_reward)}')
print(f'Pw/d:{Pw/d}')




