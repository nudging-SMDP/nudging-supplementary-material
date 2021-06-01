""" Helper functions for the Access Control Queuing problem

This script contains object definitions and herlper functions for the Access Control 
Queuing problem, which are useful for the q-learning implementation.

This file should be imported as a module and contains the following functions:

    * get_action - returns a greedy or random action, according to the exploration rate
    * step - get information about doing action in current state, like reward and nex state
    * priority_client - get a random priority for a nwe client
    * gets0unif - get a uniformly-selected random initial state

""" 

import numpy as np
import itertools

# possible priorities
PRIORITIES = np.arange(0, 4)
# reward for each priority
REWARDS = np.array([1,2,4,8])

# possible actions
REJECT = 0
ACCEPT = 1
ACTIONS = [REJECT, ACCEPT]

# total number of servers
NUM_OF_SERVERS = 10

# at each time step, a busy server will be free w.p. 0.06
PROBABILITY_FREE = 0.06

# learning rate
ALPHA = 0.05

# probability for exploration
EPSILON = 0.1

class QValueFunction:
    """ Class to represent the values of the agent and to apply its learning process following
    the q-learning's update rule

    Attributes
    ----------
        servers - array with servers (NUM_OF_SERVERS=10)
        priorities - customer's priorities
        actions - two posible actions: reject (0) or accept (1)
        alpha - learning rate used in the update rule of Q-values
        rho - gain 
        q_table - array for q-values of pairs (s,a)
        vs - array for values of each states

    """

    def __init__(self):       
        self.servers = np.arange(NUM_OF_SERVERS+1)
        self.priorities = PRIORITIES
        self.actions = ACTIONS
        self.alpha = ALPHA
        self.rho = 0.0
        self.q_table = np.zeros([NUM_OF_SERVERS+1, len(PRIORITIES), len(ACTIONS)])
        self.q_table[0,:,1] = np.nan
        self.q_table[3,1:,0] = np.nan
        self.vs = np.zeros([NUM_OF_SERVERS+1,len(PRIORITIES)])

    def value(self, num_free_servers, priority, action):
        """ Returns the value of a specific pair (s,a) """
        return self.q_table[num_free_servers, priority, action]


    def learn(self, free_servers, priority, action, new_free_servers, new_priority, reward):
        """ Apply update rule for q-learning """
        current_state = (free_servers, priority, action)        
        R = reward - self.rho
        if not(reward==8 and new_free_servers==0):
            self.q_table[current_state] = (1.0-self.alpha)*self.q_table[current_state] + \
                                        self.alpha*(R+np.nanmax(self.q_table[new_free_servers,new_priority]))
        else:
            self.q_table[current_state] = (1.0-self.alpha)*self.q_table[current_state] + self.alpha*R


    def update_vs(self): 
        """ Get the value of each state, given their Q values """ 
        x=map(lambda x:np.apply_along_axis(max, 1, self.q_table[x,:]), np.arange(NUM_OF_SERVERS+1))
        self.vs = np.array(list(x))
     



def get_action(free_servers, priority, Qvalue_function, greedy=False):
    """ Returns a greedy or random action, according to the exploration rate and greedy argument
           
    Args:
        free_servers & priority - current state of the environment
        Qvalue_function - object with Q-values of the agent
        greedy - if true, only explote; if false, balance exploration/explotation
    
    Returns:
        action - 0 (reject) or 1 (accept)
    """
    
    if free_servers == 0:
        return REJECT
    if priority == PRIORITIES[-1]:
        return ACCEPT
    if (np.random.uniform(0,1) > EPSILON) or greedy:
        return np.argmax(Qvalue_function.q_table[free_servers,priority])
    else:
        return np.random.choice(ACTIONS)


def step(free_servers, priority, action):
    """ Get information about doing action in state
        
    Returns:
        fnew - new number of free servers
        cnew - new customer priority        
        r - reward for performing action in state
    """

    f = free_servers
    n = NUM_OF_SERVERS
    p = priority
    r = 0.0
    a = action
    if(f==0):
        a = REJECT
        r = 0.0
    fnew = f + len(np.argwhere(np.random.rand(n-f)<=PROBABILITY_FREE))
    if(a==ACCEPT):
        fnew = fnew - 1
        r = REWARDS[p]
    cnew = priority_client()
    return fnew, cnew, r


def priority_client():
    """ Get priority of next customer

    Customers of priorities {8, 4, 2, 1} arrive with probabilities {0.4, 0.2, 0.2, 0.2}, 
    respectively    
    """
    return np.random.choice(PRIORITIES,size=1, replace=False, p=[0.2,0.2,0.2,0.4])[0]


def gets0():
    """ Get initial state """
    if(np.random.uniform(0,1) < 0.5):
        return 0, priority_client()
    else:
        return np.random.choice(NUM_OF_SERVERS+1,size=1, replace=False)[0], priority_client()


def gets0unif():
    return np.random.choice(NUM_OF_SERVERS+1,size=1, replace=False)[0], priority_client()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w