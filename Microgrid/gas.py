import numpy as np
import pandas as pd

COL2US = 0.00028

class Gas:

	def __init__(self):
		self.enable = False
		self.generation = 0.0
		self.priceMonthly = [6.35, 6.28, 6.23, 6.20, 6.17, 6.15, 6.15, 6.15, 6.16, 6.18, 6.21, 6.25] # ramal guajira
		self.minPriceMonthly = [3.31, 3.26, 3.21, 3.18, 3.14, 3.12, 3.09, 3.07, 3.06, 3.04, 3.03, 3.02]
		self.maxPriceMonthly = [9.20, 9.24, 9.18, 9.19, 9.21, 9.25, 9.48, 9.49, 9.52, 9.57, 9.62, 9.68]
		self.factorConversion = 8
		self.startCost = 6643382 * COL2US
		self.AOM = 8
		self.AOMfixed = (3*1e3)/(365*24)
		self.current_price = 0.0
		self.minGen = 5
		self.maxGen = 8
		self.start = False
		self.cost = 0.0

	def setEnable(self, state):
		""" Enable the gas-fired power plant """
		if(self.enable==False and state==1):
			self.start = True
		else:
			self.start = False
		self.enable = bool(state)
		if(not self.enable):
			self.current_price = 0


	def getFuelPrice(self, month, reset=False):
		""" Get generation cost based on the prices of the month """
		price = 0.0
		if reset:
			price = self.priceMonthly[month-1]
		else:
			while(price<self.minPriceMonthly[month-1] or price>self.maxPriceMonthly[month-1]):
				mu = self.priceMonthly[month-1]
				sigma = (mu - self.minPriceMonthly[month-1])/2
				price = np.random.normal(loc=mu, scale=sigma)			
		self.current_price = price*self.factorConversion*self.generation
		return self.current_price


	def setGen(self, gen):
		""" Set generation in MWh, between 5 and 8 MWh """
		if(self.enable):		
			if(gen<self.minGen):
				self.generation = self.minGen
			elif(gen>self.maxGen):
				self.generation = self.maxGen
			else:
				self.generation = gen
		else:
			self.generation = 0.0

	def getGen(self):
		""" Get current generation in MWh """
		return self.generation


	def getCost(self, month):
		""" Compute full cost of gas generation, based on the following costs:
			* Start the power plant on
			* Administration, Operation and Maintenance
			* Generation cost based on fuel price per month
		"""
		cost = 0.0
		if(self.start):
			cost += self.startCost
		cost += (self.AOM + self.AOMfixed) * self.generation
		cost += self.getFuelPrice(month)
		self.cost = cost
		return cost


