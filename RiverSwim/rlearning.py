""" R Learning application to the RiverSwim problem 

In this script, we apply R Learning (Schwartz - 1993) to solve the average reward for the 
RiverSwim MDP.

From the slides of Alessandro Lazaric, Exploration-Exploitation in Reinforcement Learning (Part1), 
and solving analytically the RiverSwim, we know that the gain value is 0.4286

This script requires that `numpy`, `matplotlib` and `pickle` be installed within the Python 
environment you are running this script in.

"""

import os
import numpy as np
import random
from tqdm import tqdm
from mdp_riverswim import RiverSwim_Split
import matplotlib.pyplot as plt

def r_learning(env, alpha=0.01, beta=0.01, max_steps=50):
    """ R Learning algorithm (Schwartz - 1993) to solve the average reward problem

    Args:
        env - the object of the environment. In this case, the RiverSwim object.
        alpha - learning rate to update the Q-values
        beta - learning rate to update the gain, rho
        max_steps = maximum number of steps for r-learning
    
    Returns:
        rho - array with all rho values given by r-learning
        values - array of all values for the recurrent state, during all interations of r-learning
    """

    env.state_values, env.state_q_values = env.init_values()
    rho = 0.0
    epsilon=0.3
    
    rhos = []
    ves = []

    for t in tqdm(range(max_steps)):

        # every 20 steps we reset the initial state
        if(t%20==0):
            state = random.choice(env.states)

        # for 1000 steps, save rho and v(sI)
        if(t%1000==0):
            rhos.append(rho)
            ves.append(env.state_values[sI])
            # print(f't={t}, p={rho}, v_s0={env.state_values[sI]}')
        
        # get action for current state
        action = env.getAction(state, epsilon, t)

        # apply action, gte new state and reward
        new_state, reward = env.step(state, action)

        # update Q-values and the policy
        sample = (reward - rho) + np.max(env.state_q_values[new_state]) 
        env.state_q_values[state][action] = (1-alpha)*env.state_q_values[state][action] + alpha*sample
        env.state_q_values[0,:] = 0.0
        env.policy[state] = np.argmax(env.state_q_values[state])
        env.state_values[state] = np.max(env.state_q_values[state])
        
        # update rho
        rho = rho + beta*(reward - rho + np.max(env.state_q_values[new_state]) -
              np.max(env.state_q_values[state]))
       
        if(new_state==0):
            state = random.choice(env.states[1:])                
        else:
            state = new_state

    return rhos, ves

# ------------------------------------------------------------------------------

# recurrent state for RiverSwim (state 1)
sI = 1
# max number of steps for R Learning
steps = 4000e3 
# learning rate to update the Q-values
alpha = 0.0001
# learning rate to update the gain, rho
beta = 0.000001
# path to save data
path = './results_rlearning/'
if not os.path.exists(path):
    os.makedirs(path)

# Create the environment
env = RiverSwim_Split(sI)
# Solve by R Learning
rhos, ves = r_learning(env, alpha=alpha, beta=beta, max_steps=int(steps))

# save records for rho and value of sI
np.save(path+'rhos.npy',rhos)
np.save(path+'values_sI.npy',ves)
