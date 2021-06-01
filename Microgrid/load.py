import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMAResults
import scipy.stats 
import random

class Load:
    def __init__(self, loads_file='load.xlsx', num_loads=1):
        self.loads_file = loads_file
        self.num_loads = num_loads    
        self.lowerDemand = []
        self.upperDemand = []
        self.col_names = []
        self.demand = None
        self.models = {}
        self.noiseDist = {}
        self.errorPred = {}
        self.processFile()
        self.loadModels()

    def processFile(self):
        """ Read load or demand data from load.xlsx """
        dfile = pd.read_excel (self.loads_file, sheet_name='data_KWh')        
        for h in range(1,25):
            self.col_names.append(f'H {h}')
        carga_total = None
        for i in range(1, self.num_loads+1):
            if i==1:
                carga_total = dfile[dfile['Set']==i][self.col_names].values/1000
            else:
                carga_total += dfile[dfile['Set']==i][self.col_names].values/1000
        fechas = dfile[dfile['Set']==1]['D'].values
        index = pd.DatetimeIndex(fechas)
        self.demand = pd.DataFrame(carga_total, index=index, columns=self.col_names)

    def loadModels(self):
        """ Load de ARIMA models for the demand of the grid in an hourly basis 

            We use the load estimated by the ARIMA model to add stochasticity to the environment

        """

        for h in range(0,24):
            model = ARIMAResults.load(f'./models/ARIMA_h{h+1}.pkl')
            data = self.demand[f'H {h+1}'].values
            predictions = self.getPredictions(model, data)            
            error = data[0:len(predictions)]- predictions
            shape, loc, scale = scipy.stats.distributions.gamma.fit(error)
            x = np.linspace(np.min(error), np.max(error), len(model.fittedvalues))
            noise_dist = scipy.stats.distributions.gamma.pdf(x,shape,loc,scale)
            noise_dist[noise_dist<noise_dist.std()] = 0.0
            self.models[h] = model
            self.noiseDist[h] = noise_dist
            self.errorPred[h] = error

    def plotErrorDist(self, h):
        """ Plot, for hour h, the real demand and the one estimated with ARIMA """
        plt.figure()
        plt.subplot(211)
        data = self.demand[f'H {h+1}'].values
        plt.plot(data, label='Original Load')
        predictions = self.getPredictions(self.models[h], data)
        plt.plot(predictions, label='Loas estimation with ARIMA model')
        plt.legend()
        plt.subplot(212)
        error = self.errorPred[h]
        x = np.linspace(np.min(error), np.max(error), len(self.models[h].fittedvalues))
        plt.hist(error, density=True)
        plt.plot(x, self.noiseDist[h],'r-')
        plt.title(f'Error distribution - Hour {h}')        
        plt.show()

    def getPredictions(self, model, data):
        """ Get load prediction with the ARIMA model """ 
        predictions = []
        for i in range(len(model.fittedvalues)):
            pred = np.exp(model.fittedvalues[i] + np.log(data[i]))
            predictions.append(pred)
        return predictions


    def showStats(self):
        """ Returns some stats about the demand of the microgrid
            
            The data is in MWh in an hourly basis:
            * Min demand
            * Max demand
        
        """
        stats = self.demand.describe()
        self.lowerDemand = []
        self.upperDemand = []
        print('--- Minimum and maximum demand per hour [MWh] ---')
        for i in self.col_names:
            self.lowerDemand.append(stats[i]['min'])
            self.upperDemand.append(stats[i]['max'])
            print(f'{i} \t min:{self.lowerDemand[-1]:10.2f} \t max:{self.upperDemand[-1]:10.2f}')
        print('------------------------------------------------')

    def showDemand(self):
        """ Plots the load of the microgrid over the year for each hour"""
        self.showStats()
        ax = self.demand.plot(subplots=True, layout=(6,4), ylim=(min(self.lowerDemand), max(self.upperDemand)))
        for i in range(len(ax)):
            for j in range(1,len(ax[i])):
                ax[i][j].get_yaxis().set_visible(False)
        plt.suptitle('Load [MWh]')
        plt.show()
        # histogram
        self.demand.plot(kind='hist', alpha=0.2, bins=20)
        plt.title('Load [MWh]')
        plt.show()

    def getSimDates(self):
        """ Returns dates for each load sequence """
        return self.demand.index

    def getPredLoad(self, index, hour):
        """ Get load prediction with the ARIMA model """ 
        load = [-1]
        data = self.demand[f'H {hour+1}'][index]
        pred = np.exp(self.models[hour].fittedvalues[index] + np.log(data))
        while(load[0]<data*0.2):        
            noise = random.choices(self.errorPred[hour], self.noiseDist[hour])
            load = pred + noise
        return load[0]


    def getDemand(self, month, day, hour):
        """ Returns next load data """ 
        date = datetime.datetime(2001,month,day)
        index = np.where(self.demand.index.to_pydatetime()==date)[0][0] - 1
        load = self.getPredLoad(index, hour)
        return load


