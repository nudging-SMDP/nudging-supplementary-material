""" Nudging to the Freeway environment

In this script, we apply optimal nudging to the freeway environment of OpenAI gym, available at: 
https://gym.openai.com/envs/Freeway-v0/

For solving an average reward problem, having a recurrent state and to generate the Bertsekas Split,
we modified the enviroment in the following way:

State: RGB image of shape (210, 160, 3) 

Recurrent state: the initial state is always the state after the first 500 steps

Actions:    0 - no move
            1 - one step forward
            2 - one step backwards
            The action selected by the agent is executed with a 98% probability

Reward: +1.0 - if the agent crosses the street
        -1.0 - if the agent is hit by a car
        -1.0 - the agent has not crossed the street after 2000 steps
         0.0 - otherwise

The episode ends if it meets any of the following conditions:
    * The agent has crossed the street
    * The agent was hit by a car
    * After 2000 steps, none of the two previous situations have occurred

In this script, nudging uses, in every iteration, DQN as a black box to approximate the 
value of the recurrent state. This value is used to update (reduce) the enclosing triangle 
and to update the gain value.

We modified the DQN implementation from Stable Baselines, available at
https://github.com/hill-a/stable-baselines, to incorporate the new reward definition and 
the recurrent state.


This script requires the following packages be installed within the Python 
environment you are running this script in.
    * numpy
    * matplotlib
    * opencv
    * gym

"""

import os
# comment if not using GPU
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.append('./deepq')
sys.path.append('./utils')

from utils.atari_wrappers import make_atari
from deepq.policies import MlpPolicy, CnnPolicy
from deepq.dqn import DQN
from nudge.nudge_functions import *

# --------- remove extra verbosity ---------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['KMP_WARNINGS'] = 'off'
# ------------------------------------------


def plot_rho_value(rhos, values, directory):
    """ Plots the evolution of the gain (rho) and the value of the recurrent state
        
    Args:
        rho - array with all rho values given by nudging
        values - array of all values for the recurrent state, during all interations of nudging
        directory - path where to save the .png image
    """
    steps = np.arange(len(rhos))
    plt.subplot(1,2,1)
    plt.plot(rhos)
    plt.grid(True)
    plt.title('Gain (ρ)', color='black')
    plt.xlabel('Steps')
    plt.subplot(1,2,2)
    plt.plot(values)
    plt.grid(True)
    plt.title('Value recurrent state (sI)', color='black')
    plt.xlabel('Steps')
    plt.savefig(directory+'evol_rho_value_sI.png')
    plt.close()

# path to save data
folder = f'./results_nudging/'
try:
    os.mkdir(folder)
except:
    pass

# create enviroment
env = make_atari('FreewayNoFrameskip-v0')

# a bound on unsigned, unnudged reward. This bound can be obtained running q-learning in an
# undiscounted Freeway problem and getting the value of the initial state for the optimal policy
D = 1.25
# Initialize enclosing triangle
set_initial_enclosing_triangle(D)
# alpha for getting rho via alpha-nudging
alpha = 0.3
# max number of iterations for nudging
maxItersNudging = 10


# load undiscounted model (the same used to get D)
values_s0 = []
rhos = []
model = DQN.load(load_path=f'./models/freeway_dqn_base', env=env)
model.gamma = 1.0
model.learning_rate = 0.5e-4
model.full_tensorboard_log = True
model.tensorboard_log = folder

# maximum number of steps for the black-box solver (DQN)
maxsteps = 800000

# Get recurrent state (state_sI)
env.reset()
for _ in range(500):
    obs, _, _, _ = env.step(0)
    state_sI = obs



###################### Begins Nudged Learning Algorithm ######################
for i in range(maxItersNudging):
    env.reset()
    print(f'************* Iteration {i} ************* ')

    # approximate rho evaluating over several points in the left and right uncertainty 
    rho = get_r(points=1000000)
    
    # get rho as in the intersection point of the conic section of the left and right uncertainty
    # rho = get_optimal_rho_value()    

    # uncomment the following line for doing alpha-nudging
    # rho = get_alpha_rho_value(alpha=alpha)

    print(f'rho = {rho}')

    # train DQN with new rho value
    model.learn(total_timesteps=maxsteps, directory=folder, iteration=i, rho=rho, log_interval=100)
    # save model
    model.save(f'{folder}/freeway_opt_nudge_{i}')

    # get q_values for recurrent state
    _, q_values = model.predict(state_sI)    
    # get value of recurrent state
    v_k = max(q_values[0])
    print(f'Value s0 = {v_k}')

    values_s0.append(v_k)
    rhos.append(rho)   

    # update the enclosing triangle vertices, given the new value of the recurrent state
    exit_code, m = update_enclosing_triangle(rho, v_k, i, folder)
    print(f'exit code {exit_code}, m={m}')
    if exit_code==4 or exit_code==-1:
        break    

# plot the evolution of rho and the valur of sI, during nudging 
plot_rho_value(rhos, values_s0, folder)

# save records for rho and value of sI
np.save(f'{folder}summary_rhos.npy',rhos)
np.save(f'{folder}summary_values_sI.npy',values_s0)
    


