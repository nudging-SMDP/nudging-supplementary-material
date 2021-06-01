""" Microgrid Environment

This designed microgrid has distributed generation (DG), such as gas-fired power plant, 
solar and wind renewable energy sources, and energy storage device. It is connected to 
the  electric power system.

State:
    * Hour
    * Day
    * Month
    * Gas plant enabled
    * Gas plant generation cost
    * Battery charge

Actions:
    * % battery charge/discharge
    * Enable gas plant
    * Gas plant generation (MWh)

Reward: based on the amount of energy from DGs used to supply the demand and if it was 
        able to sell energy to the main grid.

Cost: the operating cost includes the main grid backup and the generation cost of the 
      gas-fired power plant (all in USD)

----
Note: the provided data for load and renewable energy is private

"""


import numpy as np
import gym
from gym import spaces



from gas import Gas
from renewable import Renewable
from load import Load
from battery import EnergyStorage
from grid import MainGrid

COL2US = 0.00028
MAX_REWARD = np.inf

class Microgrid(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, day, month, gas_enable, battery_charge):
        super(Microgrid, self).__init__()

        # definition of the range for the reward
        self.reward_range = (0, MAX_REWARD) 
        self.num_envs = 1

        # Action space: 
        #   Battery       [-100%, 100]
        #   GasEnable     {0, 1}
        #   GasGeneration [5, 8] MWH
        self.action_space = spaces.Box(
            low=np.array([-100, 0, 5]), high=np.array([100, 1, 8]), dtype=np.float16)

        # State space:
        #    Hour                      {0,1,2,...,22,23 }
        #    Day                       {0,1,2,...,30,31 }
        #    Month                     {0,1,2,...,11,12 }
        #    Gas plant enabled         {0,1}
        #    Gas plant generation cost [0, 1000] USD/1000
        #    Battery charge            [0, 100]  %
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([23, 31, 12, 1, 1000, 100]), dtype=np.float16)

        # Average reward for the microgrid
        self.rho = 0.0

        # temporal variables to describe the recurrent state
        self.ini_month = month
        self.ini_day = day
        self.ini_gas_enable = gas_enable
        self.ini_bat_charge = battery_charge

        # temporal variables to describe the state
        self.year = 2000
        self.hour = -1
        self.month = month
        self.day = day
        self.sim_days = 0
        self.steps = 0

        # if True, one day of simulation completed
        self.done = False

        # Main grid
        self.grid = MainGrid()

        # Gas-fired power plant (Distributed Generation, DG)
        # by default, initial generation is set to 5MWh if the plant is enabled
        self.gasDG = Gas()
        self.gasDG.enable = gas_enable
        self.gasDG.setGen(gen=5)

        # Renewable sources
        self.renewableDG = Renewable()

        # Energy Storage
        self.battery = EnergyStorage()

        # Load
        self.load = Load(num_loads=1)

        # simulation dates
        self.simDates = self.load.getSimDates()

        # placeholders to append rewards and costs
        self.record_rewards = []
        self.record_cost = []

        # placeholder for the state of the microgrid
        self.state = []


       

    def reset(self):
        """ Reset the microgrid environment

            Returns:
                self.state: the initial state which corresponds to the recurrent state
        """

        self.hour = 0
        self.month = self.ini_month
        self.day = self.ini_day
        self.year += 1
        self.sim_days = 0
        self.steps = 0
        self.gasDG.enable = self.ini_gas_enable
        self.gasDG.setGen(gen=5)
        self.battery.charge = (self.ini_bat_charge/100.0)*self.battery.cap_max_h
        self.updateStateReset()
        return self.state

    def getReward(self, currGen, currLoad, genBat):
        """ Get the reward for the microgrid environment

            Args:
                currGen: current generation of the DG plant, in MWh, when enabled
                currLoad: current load of the microgrid, in MWh
                genBat: amount of power delivered by the energy storage system to the microgrid

            Returns:
                reward: computed based on the amount of energy from DGs used to supply the demand 
                        and if it was able to sell energy to the main grid
        """

        pGenRen = (self.renewableDG.getGen(self.month, self.hour) + genBat)/ currLoad
        if pGenRen>1:
            pGenRen = 1
        if(currGen>currLoad):
            sale = 1.0
        else:
            sale = 0.0
        reward = pGenRen + sale
        return reward

    def getCost(self, usageGrid):
        """ Get the cost associated with the generation to supply the demand

            Args:
                usageGrid: amount of power provided by the main grid                

            Returns:
                cost: computed including the main grid backup and the generation 
                      cost of the gas-fired power plant (all in USD)
        """
        # cost of energy provided by the main grid
        cost = self.grid.getCost(self.month, usageGrid)
        # cost of generation by DGs
        cost += self.gasDG.getCost(self.month)
        return cost
        

    def step(self, a):  
        """ Apply action to the microgrid

            Args:
                a: action, where:
                    a[0]: percentage of battery charge/discharge
                    a[1]: enable/disable of DGs
                    a[2]: amount of generation for DGs, in MWh

            Returns:
                state: next state of the environment
                reward: nudged reward in the form: r-ρc
                done: True if completed one day of simulation
        """

        # battery charge/discharge
        if(a[0]<0):
            genBat = self.battery.discharged(p=-a[0]/100)
            loadBat = 0.0
        else:
            genBat = 0.0
            loadBat = self.battery.charged(p=a[0]/100)
        # DG enable/disable
        if(a[1]>=0.5):
            self.gasDG.setEnable(state=1)
        else:
            self.gasDG.setEnable(state=0)
        # amount of generation for DGs, in MWh
        self.gasDG.setGen(gen=a[2])

        # net generation of the microgrid: DGs + renewable + batteries discharge
        currGen = self.gasDG.getGen() + self.renewableDG.getGen(self.month, self.hour) + genBat
        # net demand of the microgrid: load + battery charge
        currLoad = self.load.getDemand(self.month, self.day, self.hour) + loadBat
        # energy that must be provided by the main grid
        usageGrid = currLoad - currGen        
        # ------------   
        # compute cost and reward
        r = self.getReward(currGen, currLoad, genBat)
        c = self.getCost(usageGrid)/1000.0
        # ------------
        # update state
        self.hour += 1
        self.steps += 1
        if(self.hour > 23):
            self.hour = 0
            self.sim_days += 1
        self.updateState()         
        # ------------
        # Compute nudged reward                
        reward = r - self.rho*c
        # ------------
        # Save reward and cost of the action  
        self.record_rewards.append(r)
        self.record_cost.append(c)
        return self.state, reward, self.done, {}


    def updateState(self):
        """ Update state of the microgrid """    
        self.state = []  
        self.state.append(self.hour)
        self.state.append(self.day)
        self.state.append(self.month)
        # gas generation
        self.state.append(int(self.gasDG.enable))
        self.state.append(self.gasDG.getFuelPrice(month=self.month))
        # energy storage
        self.state.append(self.battery.getCharge())
        if(self.month==self.month and self.day==self.day and self.hour==23):
            # if one day of simulation is completed, done=True
            self.done = True
            self.steps = 0
        else:
            self.done = False
        self.state = np.array(self.state)
    
    def updateStateReset(self):
        """ Reset state of the microgrid to its recurrent state """ 
        self.state = []
        self.state.append(self.hour)
        self.state.append(self.day)
        self.state.append(self.month)
        self.state.append(int(self.gasDG.enable))
        self.state.append(self.gasDG.getFuelPrice(month=self.month, reset=True))
        self.state.append(self.battery.getCharge())
        self.state = np.array(self.state)
    
    def saveRecords(self, path):
        """ Save records to .npy files """
        rewards = np.array(self.record_rewards)
        costs = np.array(self.record_cost)
        np.save(f'{path}/data_ugrid_rewards.npy', rewards)
        np.save(f'{path}/data_ugrid_costs.npy', costs)

    def render(self, mode='human'):
        pass


    def updateRecord(self, i, year, currGen, currLoad, usageGrid, reward, cost):
        """ Save data about the microgrid """
        self.simRecord[i*year][self.steps][0] = currGen
        self.simRecord[i*year][self.steps][1] = currLoad
        self.simRecord[i*year][self.steps][2] = reward
        self.simRecord[i*year][self.steps][3] = cost
        self.simRecord[i*year][self.steps][4] = usageGrid
        self.simRecord[i*year][self.steps][5] = self.grid.cost
        self.simRecord[i*year][self.steps][6] = self.gasDG.enable
        self.simRecord[i*year][self.steps][7] = self.gasDG.start
        self.simRecord[i*year][self.steps][8] = self.gasDG.generation
        self.simRecord[i*year][self.steps][9] = self.gasDG.cost
        self.simRecord[i*year][self.steps][10] = self.battery.charge/self.battery.cap_max_h