import pandas as pd

class Renewable:
	def __init__(self):
		self.genSolar = None
		self.genWind = None
		self.generation = 0.0
		self.loadData()

	def loadData(self):
		dfile = pd.read_excel ('renewable.xlsx', sheet_name='solarMWh')	
		dfile2 = pd.read_excel ('renewable.xlsx', sheet_name='windMWh')	
		self.genSolar = {}
		self.genWind = {}
		for mes in range(1,13):
			self.genSolar[mes] = list(dfile.iloc[mes-1].values[1:])
			self.genWind[mes] = list(dfile2.iloc[mes-1].values[1:])


	def getGen(self, month, hour):
		self.generation = self.genSolar[month][hour] + self.genWind[month][hour]
		return self.generation




