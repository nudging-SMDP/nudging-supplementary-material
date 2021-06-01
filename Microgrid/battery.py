class EnergyStorage():
	
	def __init__(self):
		self.efficiency = 0.9
		self.cap_max_h = 9.125 #MWh 
		self.charge = self.cap_max_h

	def discharged(self, p):
		""" Discharge battery  

			Args
				p: percentage to discharge from the battery
		
			Returns:
				eDischarged: amount of energy delivered to the load or main-grid
		"""

		eDischarge = p*self.cap_max_h
		if(self.charge >= eDischarge):
			self.charge -= eDischarge
		else:
			eDischarge = self.charge
			self.charge = 0.0
		return eDischarge

	def charged(self, p):
		""" Charge battery  

			Args
				p: percentage to charge the battery
		
			Returns:
				eDischarged: amount of energy delivered to the battery
		"""
		eCharge = p*self.cap_max_h
		if((self.cap_max_h - self.charge) >= eCharge):
			self.charge += eCharge			
		else:
			eCharge = self.cap_max_h - self.charge
			self.charge = self.cap_max_h
		return eCharge


	def getCharge(self):
		""" Get current percentage of charge of the battery """
		battery = 100*self.charge/self.cap_max_h	
		return battery

