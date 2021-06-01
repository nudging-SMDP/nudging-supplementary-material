""" Nudging to the Microgrid environment

In this script, we apply optimal nudging to the microgrid, a designed general semi-Markov problem
with non-unitary costs

In this script, nudging uses, in every iteration, PPO as a black box to approximate the 
value of the recurrent state. This value is used to update (reduce) the enclosing triangle 
and to update the gain value. We use the PPO2 implementation from [Stable Baselines.](https://github.com/hill-a/stable-baselines)

This script requires the following packages be installed within the Python 
environment you are running this script in.
    * python = 3.6
    * numpy
    * matplotlib
    * ternsorflow = 1.15
    * pandas
    * tqdm
    * gym
    * statsmodels
    * stable_baselines
    * xlrd

"""

import os
# comment if not using GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from microgrid_env import Microgrid
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from nudge.nudge_functions import *

# --------- remove extra verbosity ---------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['KMP_WARNINGS'] = 'off'
# -----------------------------------------

# parameters for the recurrent state
day = 2
month = 1
gas_enable = 1 # 0 disable - 1 enable
battery_charge = 50 # initial battery charge in percentage

# maximum number of steps for the black-box solver (PPO)
maxsteps = 700000

# max number of iterations for nudging
maxItersNudging = 10

# path to save data
folder = f'./results_nudging/'
try:
    os.mkdir(folder)
except:
    pass

def plot_rho_value(rhos, values, directory):    
    """ Plots the evolution of the gain (rho) and the value of the recurrent state
        
    Args:
        rho - array with all rho values given by nudging
        values - array of all values for the recurrent state, during all interations of nudging
        directory - path where to save the .png image
    """
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

# create environment
ugrid = Microgrid(day, month, gas_enable, battery_charge)
# resent environment and get initial recurrent state
state_s0 = ugrid.reset()

# configure the black box RL solver
model = PPO2(
    policy=MlpPolicy,
    env=ugrid,
    verbose=1,
    # normalize = True,
    n_steps = 1024,
    nminibatches = 32,
    lam = 0.95,
    gamma = 1.0,
    noptepochs = 10,
    ent_coef = 0.0,
    learning_rate = 2.5e-4,
    cliprange = 0.2,
    full_tensorboard_log=True,
    tensorboard_log=f'{folder}/logs',
)

# A bound on unsigned, unnudged reward.
D = 15.0
# Initialize enclosing triangle
set_initial_enclosing_triangle(D)

values_s0 = []
rhos = []

###################### Begins Nudged Learning Algorithm ######################

for iter_nudge in range(1,maxItersNudging): 
    state_s0 = ugrid.reset()
    print(f'************* Iteration {iter_nudge} ************* ')

    # approximate rho evaluating over several points in the left and right uncertainty 
    rho = get_r(points=1000000)
    
    # get rho as in the intersection point of the conic section of the left and right uncertainty
    # rho = get_optimal_rho_value()    

    # uncomment the following line for doing alpha-nudging
    # rho = get_alpha_rho_value(alpha=alpha)

    ugrid.rho = rho
    print(f'Gain (ρ) = {rho}')

    # train PPO2 with new rho value
    model.learn(total_timesteps=maxsteps, log_interval=100)
    model.save(f'{folder}ppo_opt_nudge_{iter_nudge}')

    # get value of recurrent state
    _, value, _ = model.predict(observation=state_s0, deterministic=True)    
    v_k = value[0]
    print(f'Value sI = {v_k}')
    values_s0.append(v_k)
    rhos.append(rho)   

    # update the enclosing triangle vertices, given the new value of the recurrent state
    exit_code, m = update_enclosing_triangle(rho, v_k, iter_nudge, folder)
    
    if exit_code==4 or exit_code==-1:
        break 


plot_rho_value(rhos, values_s0, folder)

# save records for rho and value of
np.save(f'{folder}summary_rhos.npy',rhos)
np.save(f'{folder}summary_values_sI.npy',values_s0)

# Run last iteration for nudging
iter_nudge = iter_nudge + 1
if(exit_code==-1):
    print('Last iteration of nudging')
    Q = getQ()
    rho = 2*Q
    ugrid.rho = rho
    print(f'rho = {rho}')
    model.learn(total_timesteps=maxsteps, log_interval=100)
    model.save(f'{folder}ppo_opt_nudge_final')
    _, value, _ = model.predict(observation=state_s0, deterministic=True)    
    v_k = value[0]  
    print(f'Value s0 = {v_k}')
    print(f'rho={rho}')

ugrid.saveRecords(path=folder)