COL2US = 0.00028

class MainGrid:

	def __init__(self):		
		# prices in COL/kWh in 2019
		self.priceMonthly = [602.08, 614.92, 641.83, 627.95, 607.56, 613.90, 616.02, 630.60, 624.87, 634.29, 647.05, 623.12]
		self.backupCost = ((958e6*COL2US)/365)/24
		self.current_price = 0.0
		self.cost = 0.0
		self.energy = 0.0

	def getCost(self, month, energy):
		""" Returns cost for using the main grid"""
		self.current_price = self.priceMonthly[month-1]*COL2US*1000
		cost = self.backupCost
		if(energy>0):
			cost += self.current_price*energy
		self.energy = energy
		self.cost = cost
		return cost




