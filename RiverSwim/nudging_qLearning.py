""" Nudging to the RiverSwim problem

In this script, we apply alpha-nudging or optimal-nudging to the riverswim problem, to 
approximate iteratively the gain, rho.

In this script, nudging uses, in every iteration, q-learning as a black box to approximate the 
value of the recurrent state. This value is used to update (reduce) the enclosing triangle 
and to update the gain value.

From the slides of Alessandro Lazaric, Exploration-Exploitation in Reinforcement Learning (Part1), 
and solving analytically the RiverSwim, we know that the gain value is 0.4286

This script requires that `numpy`, `matplotlib` and `pickle` be installed within the Python 
environment you are running this script in.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mdp_riverswim import RiverSwim_Split
from nudge.nudge_functions import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def plot_rho_value(rhos, values, directory):
    """ Plots the evolution of the gain (rho) and the value of the recurrent state
        The plot of rho is compared with the baseline 0.4286, which is the gain for the RiverSwim

    Args:
        rho - array with all rho values given by nudging
        values - array of all values for the recurrent state, during all interations of nudging
        directory - path where to save the .png image
    """
    rho_base = 0.4286224337994642
    steps = np.arange(len(rhos))
    plt.subplot(1,2,1)
    plt.plot(rhos, color='r')
    plt.plot(rho_base*np.ones(len(rhos)), linestyle = 'dashed', color='k')
    plt.grid(True)
    plt.title('Gain (ρ)', color='black')
    plt.xlabel('Steps (x10³)')
    plt.subplot(1,2,2)
    plt.plot(values, color='k')
    plt.grid(True)
    plt.title('Value recurrent state (sI)', color='black')
    plt.xlabel('Steps (x10³)')
    plt.show()
    plt.savefig(directory+'evol_rho_value_sI.png')
    plt.close()


# path to save data
path = './results_nudging/'
if not os.path.exists(path):
    os.makedirs(path)

# a bound on unsigned, unnudged reward. This bound can be obtained running value iteration in the
# original RiverSwim problem and getting the value of state 1 for the optimal policy, which is always
# take the action Right
D = 10000

# max number of iterations for nudging
maxItersNudging = 100

# minimum number of steps for the black-box solver, in this script, q-learning
min_steps = int(150e3)
# maximum number of steps for the black-box solver, in this script, q-learning
max_steps = int(1000e3) 
# learning rate for q-learning
alpha_0 =  0.1
# recurrent state for RiverSwim (state 1)
sI = 1


values_s0 = []
rhos = []
rhoes = []

lastPolicy = None
lastValue = 0.0

# initialize enclosing triangle
set_initial_enclosing_triangle(D)

###################### Begins Nudged Learning Algorithm ######################
for iter_nudge in range(maxItersNudging): 
    # create the environment
    env = RiverSwim_Split(sI)
    print('')
    print(f'************* Iteration {iter_nudge} ************* ')

    # approximate rho evaluating over several points in the left and right uncertainty 
    rho = get_r(points=1000000)

    # uncomment the following line for doing alpha-nudging
    # rho = get_alpha_rho_value(alpha=0.6)

    print(f'rho value = {rho}') 

    if(rho<0.4):
        # decrese learning rate when rho is less than 0.4 (this is particular for the problem)        
        min_steps = int(1500e3) 
        max_steps = min_steps
        p, v = env.qLearning(rho=rho, epsilon=0.3, minsteps=min_steps, maxsteps=max_steps)
    else:
        alpha_0 =  0.1
        min_steps = int(150e3)
        max_steps = min_steps
        p, v = env.qLearning(rho=rho, epsilon=0.3, alpha=alpha_0, minsteps=min_steps, maxsteps=max_steps)
    

    values_s0 = np.concatenate((values_s0,v),0)
    rhos = np.concatenate((rhos,p),0)
    rhoes.append(p[-1])
    v_k = env.state_values[sI]
    print(f'V(sI) = {v_k}, policy={env.policy}')    

    # if the last and the current policies are the same but the value of the recurrent state 
    # changes signs, we can do a termination by zero crossing
    if((lastPolicy==env.policy).all() and lastValue*v_k<0):
        break
    else:
        lastPolicy = env.policy
        lastValue = v_k

    # update the enclosing triangle vertices, given the new value of the recurrent state
    exit_code, m = update_enclosing_triangle(rho, v_k, iter_nudge, path)
    
    if(exit_code==4 or exit_code==-1):
        break
    del env

# plot the evolution of rho and the valur of sI, during nudging
plot_rho_value(rhos, values_s0, path)
# save records for rho and value of sI
np.save(path+'rhos.npy',rhos)
np.save(path+'values_sI.npy',values_s0)
