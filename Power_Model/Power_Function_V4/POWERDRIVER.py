import orbitDict
chosenOrbits = orbitDict.chosenOrbits
from design import designConfiguration, orbitConfiguration, powerConfiguration, commsConfiguration

import math
import numpy
import matplotlib.pyplot as plt
from dataclasses import dataclass
#from csltk.jpl_query import JPLFamily
#from csltk import cr3bp
#from csltk.utilities import System
# import pickle
# import itertools
from collections import Counter
from os import listdir
from os.path import isfile, join

import multiprocessing
import time
import sys
import os

import FullPowerModelFunction

allDesigns=[]
designScores = []
# # Import orbitDict files
# files = [131073, 131074, 131075, 131076, 131077, 131078, 131079, 131080, 131081, 131082, 131083, 131084, 131085, 131086, 131087, 131088, 131089, 131090, 131091, 131092, 131093, 131094, 131095, 131096, 131097, 131098, 131099, 131100, 131101, 131102, 131103, 131104, 131105, 131106, 131107, 131108, 211201, 221201, 611201, 621201, 781201]
# lengthOfFiles = len(files)
# print(lengthOfFiles, 'designs being tested')
# for i in files:
#     filename = 'Bullshit/design' + str(i) + '.dat'
#     allDesigns.append(chosenOrbits.load(filename))

files = [f for f in listdir('Bullshit') if isfile(join('Bullshit', f))]
for i in range(len(files)):
    filename = 'Bullshit/' + files[i]
    if filename != 'Bullshit/.DS_Store':
        allDesigns.append(chosenOrbits.load(filename))

lengthOfFiles = len(files)-1

###
### CDR DESIGN ID: 42
###

design_ID = 42

#
# ## Spring semester i dont remember anything
# ############################################################
# print('R E M E M B E R : : ID =', allDesigns[design_ID].ID)
# print('R E M E M B E R : : orbits =', len(allDesigns[design_ID].orbits))
# print('R E M E M B E R : : numSats =', allDesigns[design_ID].numSats)
# print('R E M E M B E R : : totSats =', allDesigns[design_ID].totSats)
# print('R E M E M B E R : : solarPanelSize =', allDesigns[design_ID].solarPanelSize)
# print('R E M E M B E R : : batterySize =', allDesigns[design_ID].batterySize)
# print('R E M E M B E R : : laserPower =', allDesigns[design_ID].laserPower)
# print('R E M E M B E R : : apetRad =', allDesigns[design_ID].apetRad)
# print('R E M E M B E R : : receiverRad_power =', allDesigns[design_ID].receiverRad_power)
# print('R E M E M B E R : : diameterTxM =', allDesigns[design_ID].diameterTxM)
# print('R E M E M B E R : : diameterTxO =', allDesigns[design_ID].diameterTxO)
# print('R E M E M B E R : : dataRate =', allDesigns[design_ID].dataRate)
# print('R E M E M B E R : : dataRate_ED =', allDesigns[design_ID].dataRate_ED)
# print('R E M E M B E R : : commsObj =', allDesigns[design_ID].commsObj)
# print('R E M E M B E R : : powerObj =', allDesigns[design_ID].powerObj)
# print('R E M E M B E R : : roiObj =', allDesigns[design_ID].roiObj)
# print('R E M E M B E R : : constraint =', allDesigns[design_ID].constraint)
# ############################################################
  


 
################### now start the actual power driver ################

import numpy
E_tot = []
N_start = 0
N_end = lengthOfFiles-1
#N_end = 1
print(N_end-N_start, 'designs being tested')
for i in range(N_start,N_end):
    design_ID = i

    # design variable extraction
    ID = allDesigns[design_ID].ID
    orbits = allDesigns[design_ID].orbits # List of all orbits (family, trajectory, velocity, period, percent eclipsed) in current design
    numSats = allDesigns[design_ID].numSats # Number of satellites on each orbit as list
    totSats = allDesigns[design_ID].totSats # Total number of satellites in constellation
    solarPanelSize = allDesigns[design_ID].solarPanelSize # Solar panel area [m^2]
    batterySize = allDesigns[design_ID].batterySize # Battery mass [kg]
    laserPower = allDesigns[design_ID].laserPower # Wattage required to power the laser [W]
    r_aperture = allDesigns[design_ID].apetRad # radius of output lens on SC [m]
    r = allDesigns[design_ID].receiverRad_power # radius of ground receiver [m]
    panelSize = allDesigns[design_ID].solarPanelSize
    LI_battery_mass_total = allDesigns[design_ID].batterySize
    laser_intake_wattage = allDesigns[design_ID].laserPower

        # for i in range(0,len(orbits)):
        #     orbit_curr = orbits[i]
        #     numSC_curr = numSats[i]
        #     x = orbit_curr.x
        #     y = orbit_curr.y
        #     z = orbit_curr.z
        #     vx = orbit_curr.vx
        #     vy = orbit_curr.vy
        #     vz = orbit_curr.vz
        #     eclipse_percent = orbit_curr.eclipse
        #     Period = orbit_curr.T

        # t = numpy.linspace(0,Period,len(x))

        # PosnTime = [x,y,z,t]

        # determine total energy received for this orbit configuration
    print("______________________________________________________")
    print('Design',design_ID,':\n')
    E_tot.append(FullPowerModelFunction.FullPowerModelFunction(orbits, numSats, panelSize, LI_battery_mass_total, laser_intake_wattage, r_aperture, r))

max_E = max(E_tot)
max_E_ID = E_tot.index(max_E)

print("______________________________________________________")
print('\nFinal Results for All Designs:')
print('Maximum Energy achieved by a design:',max_E,'kWh/24h')
print('Maximum Energy Orbit ID:',max_E_ID)
