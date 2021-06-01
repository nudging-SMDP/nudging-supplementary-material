""" Nudging to the Access Control Queuing problem

In the access control queuing task, at the head of a single queue that manages access to
n = 10 servers, customers of priorities {8, 4, 2, 1} arrive with probabilities {0.4, 0.2, 0.2, 0.2},
respectively. At each decision epoch, the customer at the head of the queue is either
assigned to a free server (if any are available), with a pay-off equal to the customer’s
priority; or rejected, with zero pay-off. Between decision epochs, servers free independently
with probability p = 0.06. The goal is to maximize the expected average reward.

For any state with m free servers, and any policy, there is a nonzero probability that all of
the next m customers will have priority 8 and no servers will free in the current and the next
m decision epochs. Since the only available action for customers with priority 8 is to accept 
them, all servers would fill, then, so for any state and policy there is a nonzero probability
of reaching the state with no free servers, making it recurrent.

In this script, we apply alpha-nudging or optimal-nudging to the Access Control Queuing problem, 
to approximate iteratively the gain, rho. Nudging uses, in every iteration, q-learning as a 
black box to approximate the value of the recurrent state. This value is used to update (reduce)
the enclosing triangle and to update the gain value.

This script requires that `numpy`, `matplotlib` and `pickle` be installed within the Python 
environment you are running this script in.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_qleaning import *
from nudge.nudge_functions import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def black_box(maxsteps, Qvalue_function):
    """
    q-learning algorithm to apprixamate Q-values of pair (s,a) based on exploration/explotation

    Args:
        maxsteps - maximum number of steps for running q-learning
        Qvalue_function - object with q-values matrix and information such as gain, learning rate,
                          states and actions

    Returns:
        values - array with the value of the recuurrent state during all iterations of q-learning
        rhos - array with the rho value
    """

    catc = int(10*1000000/1000/2)
    current_free_servers, current_priority = gets0unif()
    values = (int(maxsteps/catc)-1)*[0.0]
    rhos = (int(maxsteps/catc)-1)*[0.0]
    idx = 0
    for t in tqdm(range(1,maxsteps)):
        if(t%10==0):
            current_free_servers, current_priority = gets0()
        if(t%catc==0):
            vs = Qvalue_function.vs[0,3]
            values[idx] = vs
            rhos[idx] = Qvalue_function.rho
            # print(str(t)+" "+str(vs))
            Qvalue_function.update_vs()
            idx = idx + 1
        
        # choose action for current state
        current_action = get_action(current_free_servers, current_priority, 
                        Qvalue_function, greedy=False)
        # observe reward and next state
        new_free_servers, new_priority, reward = step(current_free_servers, 
                                                current_priority, current_action)
        #update q values
        Qvalue_function.learn(current_free_servers, current_priority, current_action,
                                    new_free_servers, new_priority, reward)
        
        if(reward==8 and new_free_servers==0):
            new_free_servers, new_priority = gets0unif()
        current_free_servers = new_free_servers
        current_priority = new_priority
    return values, rhos

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

###############################################

# path to save data
directory = './results_nudging/'
if not os.path.exists(directory):
    os.makedirs(directory)

# Run vanilla q-learning (undiscounted) to get a bound for D
Qvalue_function = QValueFunction()
# max steps for the black box Q-learning
maxsteps = 1000000
# learning rate use in the Q-values update rule
Qvalue_function.alpha = 0.05
# estimate D value with zero-gain
Qvalue_function.rho = 0.0

print(f'Computing the bound for the enclosing triangle, D')
print(f'Call to blackbox with rho = 0.0')
# approximate Q-values via q-learning
values_sI, rhos = black_box(maxsteps, Qvalue_function)
# get the bound for D as the value of the recurrent state sI
D =  Qvalue_function.vs[0,3]
print(f'D = {D} \n')


###################### Begins Nudged Learning Algorithm ######################

# max number of iterations for nudging
maxItersNudging = 10
# maximum number of steps for the black-box solver, in this script, q-learning
maxsteps = 1500000
# initialize enclosing triangle
set_initial_enclosing_triangle(D)

# placeholder for the gain that nudging approximates
rho = 0.0

for i in range(maxItersNudging):

    print(f'************* Iteration {i} ************* ')

    # approximate rho evaluating over several points in the left and right uncertainty 
    rho = get_r(points=1000000)

    # NOTE: uncomment the following line for doing alpha-nudging
    # rho = get_alpha_rho_value(alpha=0.6)

    print(f'rho = {rho}')

    # approximate Q-values via black-box q-learning with reward r_{t} − ρ*k_{t+1} and no discount
    Qvalue_function.rho = rho
    Qvalue_function.alpha = 0.01
    v, r = black_box(maxsteps, Qvalue_function)
    values_sI = np.concatenate((values_sI,v),0)
    rhos = np.concatenate((rhos,r),0)

    # get new value for recurrent state SI
    v_k = Qvalue_function.vs[0,3]
    print(f'V(sI) = {v_k}')
    # update the enclosing triangle vertices, given the new value of the recurrent state
    exit_code, m = update_enclosing_triangle(rho, v_k, i, directory)
    print(f'')
    if exit_code==4 or exit_code==-1:
        break
    

# plot the evolution of rho and the valur of sI, during nudging
plot_rho_value(rhos, values_sI, directory)
# save records for rho and value of sI
np.save(directory+'rhos.npy',rhos)
np.save(directory+'values_sI.npy',values_sI)
