""" Deep R Learning to the Freeway environment

In this script, we apply the deep R Learning algorithm to the freeway environment of OpenAI gym, 
available at: https://gym.openai.com/envs/Freeway-v0/

For implementing Deep R Learning, we followed the algorithm proposed by Shan, Jiang, Hart and Stone
in their paper: Deep R-Learning for Continual Area Sweeping, presented at IROS 2020
https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/IROS20-shah.pdf

For solving an average reward problem, we modified the enviroment in the following way:

State: RGB image of shape (210, 160, 3) 

Recurrent state: the initial state is always the state after the first 500 steps

Actions:    0 - no move
            1 - one step forward
            2 - one step backwards
            The action selected by the agent is executed with a 98% probability

Reward: +1.0 - if the agent crosses the street
        -1.0 - if the agent is hit by a car
        -1.0 - the agent has not crossed the street after 2000 steps
         0.0 - in other case

The episode ends if it meets any of the following conditions:
    * The agent has crossed the street
    * The agent was hit by a car
    * afeter 2000 steps, none of the two previous situations have occurred

We modified the DQN implementation from Stable Baselines, available at
https://github.com/hill-a/stable-baselines, to to follow the Deep R Learning 
algorithm, and incorporate the new reward definition and the recurrent state.


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
sys.path.append('./deepq')
sys.path.append('./utils')

from utils.atari_wrappers import make_atari
from deepq.policies import MlpPolicy, CnnPolicy
from deepq.dqn_rlearning import DQN


# --------- remove extra verbosity ---------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['KMP_WARNINGS'] = 'off'
# ------------------------------------------


# create enviroment
env = make_atari('FreewayNoFrameskip-v0')

# path to save data
folder = f'./results_drl/'
try:
	os.mkdir(folder)
except:
	pass

# maximum number of steps for deep R Learning
maxsteps = 1200000
# initial value for rho
rho = 0.0

model = DQN(policy=CnnPolicy, 
            env=env, 
            verbose=1, 
            gamma=1.0, 
            learning_rate=1e-4,
            beta_rate = 1e-5,
            buffer_size=10000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            double_q=True,
            prioritized_replay=True, 
            tensorboard_log=folder, 
            full_tensorboard_log=True)

model.learn(total_timesteps=maxsteps, directory=folder, iteration=0, rho=rho, log_interval=100)
model.save(f'{folder}/freeway_drl')

