
import numpy as np
from orbitDict import chosenOrbits
import pickle
import os

class designConfiguration: 

    def __init__(self, ID, orbits, numSats, totSats, solarPanelSize, batterySize, laserPower, apetRad, receiverRad_power, diameterTxM, diameterTxO, dataRate, dataRate_ED, FOM):
        self.ID = ID 
        self.orbits = orbits # All orbits (family, trajectory, velocity, period, percent eclipsed) in current design
        self.numSats = numSats # Number of satellites on each orbit
        self.totSats = totSats # Total number of satellites in constellation 
        self.solarPanelSize = solarPanelSize # Solar panel area [m^2]
        self.batterySize = batterySize # Battery mass [kg]
        self.laserPower = laserPower # Wattage required to power the laser [W]
        self.apetRad = apetRad # radius of output lens on SC [m]
        self.receiverRad_power = receiverRad_power # radius of ground receiver [m]
        self.diameterTxM = diameterTxM # antenna diameter of the receiver on the moon [m]
        self.diameterTxO = diameterTxO # antenna diameter of the receiver on the satellite [m]
        self.dataRate = dataRate # desired lunar data rate [bps]
        self.dataRate_ED = dataRate_ED # desired data rate for earth downlink [bps]
        self.FOM = FOM
        self.commsObj = []
        self.powerObj = float
        self.roiObj = float
        self.constraint = int
    
    # def add_orbit(self, obj):
    #     self.orbits.append(obj) 


    def add_commsObj(self, obj1, obj2):
        self.commsObj = [obj1, obj2]
    
    def add_powerObj(self, obj1):
        self.powerObj = obj1
    
    def add_roiObj(self, obj1): 
        self.roiObj = obj1 

    def add_constraint(self,commsConstr, powerConstr, roiConstr): 
        self.constraint = commsConstr*powerConstr*roiConstr
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName,"wb")
        pickle.dump(self,f)
        f.close()
    def load(self, fileName):
        """Return a thing loaded from a file."""
        this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
        data_path = os.path.join(this_dir, fileName)
        f = open(data_path, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj

# if __name__ == "__main__":
#     # code for standalone use
#     foo = designConfiguration("1", "2", "3", "4", "5", "6", "7", "8", "9", "10","11","12","13","14")
#     x = (1,2,3)
#     foo.save("allDesignsBackup/foo" + str(x) + ".dat")
#     x = foo.load("allDesignsBackup/foo" + str(x) + ".dat")
#     print(x.dataRate)


class orbitConfiguration:
    def __init__(self, orbitID, orbits, numSats, totSats):
        self.orbitID = orbitID 
        self.orbits = orbits # All orbits (family, trajectory, velocity, period, percent eclipsed) in current design
        self.numSats = numSats # Number of satellites on each orbit
        self.totSats = totSats # Total number of satellites in constellation 
        self.FOM = [] # FOM function output [TimeLOS, percentCoverage, meanCoverageGap, timeAvgGap, maxCoverageGap, meanResponseTime]

class powerConfiguration:
    def __init__(self, powerID, solarPanelSize, batterySize, laserPower, apetRad, receiverRad_power):
        self.powerID  = powerID 
        self.solarPanelSize = solarPanelSize # Solar panel area [m^2]
        self.batterySize = batterySize # Battery mass [kg]
        self.laserPower = laserPower # Wattage required to power the laser [W]
        self.apetRad = apetRad # radius of output lens on SC [m]
        self.receiverRad_power = receiverRad_power # radius of ground receiver [m]

class commsConfiguration:
    def __init__(self, commsID, diameterTxM, diameterTxO, dataRate, dataRate_ED):
        self.commsID = commsID 
        self.diameterTxM = diameterTxM # antenna diameter of the receiver on the moon [m]
        self.diameterTxO = diameterTxO # antenna diameter of the receiver on the satellite [m]
        self.dataRate = dataRate # desired lunar data rate [bps]
        self.dataRate_ED = dataRate_ED # desired data rate for earth downlink [bps]