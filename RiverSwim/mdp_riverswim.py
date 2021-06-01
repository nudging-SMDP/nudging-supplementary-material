""" MDP for River Swim problem

This MDP follows the parameters exposed by Alessandro Lazaric in his slides about
Exploration-Exploitation in Reinforcement Learning (Part1)

Assuming the task is unichain and state 1 as the recurrent state, we generate the Bertsekas Split, 
where we generate an artificial terminal state, that once reached transitions to itself with 
probability one, reward zero, and, for numerical stability, cost zero. This is our state 0.

This script requires that `numpy` and `itertools` be installed within the Python environment you
are running this script in.

This file can also be imported as a module and contains the class RiverSwim_Split for solving the 
average reward problem with nudging.

"""

import numpy as np
import copy
import random
from tqdm import tqdm
from itertools import product

# transition matrix with probabilities for the right action
tranMatrix_right = np.array([[0.40, 0.60, 0.00, 0.00, 0.00, 0.00],
                             [0.05, 0.60, 0.35, 0.00, 0.00, 0.00],
                             [0.00, 0.05, 0.60, 0.35, 0.00, 0.00],
                             [0.00, 0.00, 0.05, 0.60, 0.35, 0.00],
                             [0.00, 0.00, 0.00, 0.05, 0.60, 0.35],
                             [0.00, 0.00, 0.00, 0.00, 0.40, 0.60]])

 # transition matrix with probabilities for the left action       
tranMatrix_left = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

class RiverSwim_Split():
    """ Class to create the object RiverSwim_Split with sI as recurrent state
        This object acts as the environment for the RL algorithm

    Attributes
    ----------
        states - state space 
        num_states -  cardinality of state space
        visit_states - array for counting the times a state was visited
        actions - action space. 0 means <- and 1 means ->
        state_values - array for storing the value of each state
        state_q_values - array for storing the q-values for each (s,a) pair
        policy - array with the greedy action for each state
        P - transition matrix
        sI - recurrent initial state

    """

    def __init__(self, sI):
        self.states = [0,1,2,3,4,5,6] #s0 is the artificial terminal state
        self.num_states = len(self.states)
        self.visits_states = np.ones(self.num_states)
        self.actions = [0,1]
        self.state_values, self.state_q_values = self.init_values()
        self.policy = np.zeros(len(self.states))
        self.P = np.zeros([self.num_states, self.num_states, 2])
        self.sI = sI
        self.updateP(sI)


    def updateP(self, sI):
        """ updates de transition matriz according with the selected recurrent state
            uses the tranMatrix_left and tranMatrix_right, but takes into account the 
            added artificial terminal state
        
        Args:
            sI: recurrent state
        """

        states = [1,2,3,4,5,6]
        states.remove(sI)
        for s in states:
            self.P[1:,s,0] = tranMatrix_left[:,s-1]
            self.P[1:,s,1] = tranMatrix_right[:,s-1]
        self.P[1:,0,0] = tranMatrix_left[:,sI-1]
        self.P[1:,0,1] = tranMatrix_right[:,sI-1]

    def getPoliciesValues(self, rho=0.0, gamma=1.0):  
        """ Generates all posible policies and computed for all their value.
            It is used for getting an analytic solution
        
        Args:
            rho - gain. Default value is 0.0
            gamma - discount factor. For average reward problems it's default value is 1.0
        
        Returns:
            PI_values: value of each state for each policy
        """

        R = np.zeros([self.num_states, self.num_states, 2]) - rho
        if(self.sI==1):
            R[1,0,0] = 0.01 - rho
        else:
            R[1,1,0] = 0.01 - rho
        if(self.sI==6):
            R[6,5,1] = 1.0 - rho
            R[6,0,1] = 1.0 - rho
        else:
            R[6,5:7,1] = 1.0 - rho
        P = self.P
        PI = list(product(range(2), repeat=self.num_states))
        PI_values = np.zeros((len(PI), self.num_states))
        for i in range(len(PI)):            
            pi = list(PI[i])            
            p = self.get_m(P,pi)
            r = self.get_m(R,pi)
            pr = np.sum(p*r,axis=1)
            LD = np.diag(np.ones(self.num_states)) - gamma*p
            if(np.linalg.det(LD)):
                v = np.linalg.solve(LD,pr)
            else:
                v = np.nan
            PI_values[i] = v 
        return PI_values

    def getPoliciesCost(self, gamma=1.0): 
        """ Generates all posible policies and computed for all their cost.
            It is used for getting an analytic solution.
            All transition cost are assumed to be 1.0
        
        Args:
            gamma - discount factor. For average reward problems it's default value is 1.0
        
        Returns:
            PI_cost: cost of each state for each policy
        """
      
        P = self.P
        PI = list(product(range(2), repeat=self.num_states))
        PI_cost = np.zeros((len(PI), self.num_states))
        for i in range(len(PI)):            
            pi = list(PI[i])            
            p = self.get_m(P,pi)
            if(i==0):
                r = 0.0
            else:
                r = 1.0
            pr = np.sum(p*r,axis=1)
            LD = np.diag(np.ones(self.num_states)) - gamma*p
            if(np.linalg.det(LD)):
                v = np.linalg.solve(LD,pr)
            else:
                v = np.nan
            PI_cost[i] = v 
        return PI_cost

    def solve_bellman(self, rho=0.0, gamma=1.0, minsteps=500, maxsteps=15000, show=False ):
        """ Solves the MDP with value iteration

        Args:
            rho - gain
            gamma - discount factor. For average reward problems it's default value is 1.0
            minsteps - minimum number of steps for running value iteration
            maxsteps - maximum number of steps for running value iteration
            show - boolean for debugging. Default value is False.

        Returns:
            values_sI: list with the records of the value for the recurrent state
        """
        
        self.state_values, self.state_q_values = self.init_values()
        values_sI = []
        lastValue = 10**6
           
        for i in range(maxsteps):            
            state_values_1, state_q_values_1 = self.init_values()
            for state in self.states[1:]:
                for action in self.actions:
                    q = 0.0
                    for state_ in self.states: 
                        rew = self.getR(state, action)   
                        prob_trans = self.P[state][state_][action]
                        q += prob_trans*( (rew-rho) + gamma*self.state_values[state_] ) 
                    state_q_values_1[state][action] = q
                self.policy[state] = random.choice( np.where(state_q_values_1[state] == state_q_values_1[state].max() )[0])
                state_values_1[state] = np.max(state_q_values_1[state])
            self.state_values = copy.deepcopy(state_values_1)
            values_sI.append(self.state_values[self.sI])
            if show:
                print(f'steps={i} - rho={rho} - v(sI)={self.state_values[self.sI]}')
            self.state_q_values = copy.deepcopy(state_q_values_1)
            if(i>minsteps and np.abs(lastValue-values_sI[-1])<10**-9):
                break
            else:
                lastValue = copy.deepcopy(values_sI[-1])
            i += 1
        return values_sI

    def get_m(self, M, v):
        l = len(v)
        m = np.nan * np.ones([l,M.shape[1]])
        for i in range(l):
            m[i,:] = M[i,:,v[i]]
        return m
        

    def init_values(self):
        """ Initialize arrays for state_values and state_q_values """
        state_values = np.zeros(self.num_states)
        state_q_values = np.zeros((self.num_states, 2))
        return state_values, state_q_values

    def getR(self, state, action):
        """ Returns reward for performing action in state """
        reward = 0.0
        if(state==1 and action==0):
            reward = 0.01
        elif(state==6 and action==1):
            reward = 1.0
        return reward
    
    def getS(self, state, action):
        """ Returns next state, according to performed action and the transition probabilities """
        p = self.P[state,:,action]
        newS = random.choices(self.states, weights=p, k=1)[0]
        return newS

    def step(self, state, action):
        """ Get information about doing action in state
        
        Returns:
            next_state - new state of the environment after doing action in state
            reward - reward for performing action in state
        """

        next_state = self.getS(state, action)
        reward = self.getR(state, action)
        self.visits_states[next_state] += 1
        return next_state, reward

    
    def getAction(self, state, eps_0, t):
        """ Returns a greedy or random action, according to the exploration rate

            During 1M timesteps, the exploration rate is 1.0. After that, the exploration rate decays
            with the number of visits for each state

        Args:
            state - current state of the environment
            eps_0 - base exploration rate
            t - timestep
        
        Returns:
            action - 0 (left) or 1 (right)
        """

        if(t<int(1000e3)):
            epsilon = 1.0
        else:
            epsilon = eps_0 / (self.visits_states[state]**(1/2))

        if( np.random.uniform(0,1) < epsilon ):
            action = random.choice(self.actions)
        else:
            action = np.argmax(self.state_q_values[state])
        return action



    def reset(self):
        """ resets the environment """
        self.state_values, self.state_q_values = self.init_values()
        self.visits_states = np.ones(self.num_states)
        self.policy = np.zeros(len(self.states))

    
    def qLearning(self, rho=0.0, epsilon=0.3, alpha=0.01, minsteps=500, maxsteps=int(300e3)):
        """ Solves the MDP with Q-leanring

        Args:
            rho - gain
            epsilon - base explorarion rate
            alpha - learning rate
            minsteps - minimum number of steps for running q-learning
            maxsteps - maximum number of steps for running q-learning
            
        Returns:
            rhos: list with the record of rho
            ves: list with the records of the value for the recurrent state
        """
        
        self.reset()
        rhos = []
        ves = []
        lastValue = 10**6
        last_t = 0
        

        for t in tqdm(range(0,maxsteps)):

            if(t%20==0):
                state = random.choice(self.states[1:])                
            if(t%1000==0):
                rhos.append(rho)
                ves.append(self.state_values[self.sI])
                

            action = self.getAction(state, epsilon, t)
            new_state, reward = self.step(state, action)            
            sample = (reward - rho) + np.max(self.state_q_values[new_state])            
            self.state_q_values[state][action] = (1-alpha)*self.state_q_values[state][action] + alpha*sample
            self.state_q_values[0,:] = 0.0
            self.policy[state] = np.argmax(self.state_q_values[state])
            self.state_values[state] = np.max(self.state_q_values[state])

            if(new_state==0):
                state = random.choice(self.states[1:])                
            else:
                state = new_state

            if(t>minsteps and np.abs(lastValue-self.state_values[self.sI])<10**-9):                
                break
            else:
                lastValue = copy.deepcopy(self.state_values[self.sI])

        return rhos, ves
    
    