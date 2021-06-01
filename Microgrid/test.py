import os
# comment if not using GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from microgrid_env import Microgrid
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from nudge.nudge_functions import *
import numpy as np


# path to final model
path = f'./results_nudging/ppo_opt_nudge_final'

# parameters for the recurrent state
day = 2
month = 1
gas_enable = True
battery_charge = 50

# number of episodes to test
test_episodes = 100


# get estimated gain
rhos = np.load(f'{path}/summary_rhos.npy')

# create environment
ugrid = Microgrid(day, month, gas_enable, battery_charge)
ugrid.rho = rhos[-1]

# reset environment and get recurrent state
state_s0 = ugrid.reset()

# load policy model
model = PPO2.load(f'{path}')

test_rewards = []
test_cost = []
test_rc = []

# ------------ TESTING ------------
for i in range(test_episodes):
    # reset environment
    state = ugrid.reset()
    done = False
    rewards = []
    costs = []
    while(not done):  
        # apply policy      
        action, value, _ = model.predict(observation=state_s0, deterministic=True) 
        # get next state, nudged reward and done flag
        state, r, done, _ = ugrid.step(action)     
        # save rewards and cost of the microgrid followinf the learned policy   
        rewards.append(ugrid.record_rewards[-1])
        costs.append(ugrid.record_cost[-1])

        if done:            
            test_rewards.append(np.mean(rewards))
            test_cost.append(np.mean(costs))
            rc = test_rewards[-1]/test_cost[-1]
            test_rc.append(rc)
            break
        
# Report
print(f'Rewards, r = {np.mean(test_rewards)}')
print(f'Cost, c = {np.mean(test_cost)}')
print(f'r/c = {np.mean(test_rc)}')