import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from csltk.jpl_query import JPLFamily
from csltk import cr3bp
from csltk.utilities import System
import pickle
import itertools
from collections import Counter
from os import listdir
from os.path import isfile, join
import multiprocessing
import time
import sys

import orbitDict
chosenOrbits = orbitDict.chosenOrbits
from design import designConfiguration, orbitConfiguration, powerConfiguration, commsConfiguration
from ROI_Model.ROI_Driver import roi
roi = roi()
from Communications_Model.Comms_Driver import comms
comms = comms()
# from Power_Function_V4.POWERDRIVER import power  #idk
from Power_Model.Power_Driver import power  # old
power = power()



# Functions to get combinations w/o repetitions
def repeat_chain(values, counts):
    return itertools.chain.from_iterable(map(itertools.repeat, values, counts))

def unique_combinations_from_value_counts(values, counts, r):
    n = len(counts)
    indices = list(itertools.islice(repeat_chain(itertools.count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), repeat_chain(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), repeat_chain(itertools.count(j), counts[j:])):
            indices[i] = j

def unique_combinations(iterable, r):
    values, counts = zip(*Counter(iterable).items())
    return unique_combinations_from_value_counts(values, counts, r)

# https://stackoverflow.com/questions/28965734/general-bars-and-stars
def partitions(n, k):
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]

# Functions to allow for multiprocessing
def orbitConfig(orbitDesign, orbit, orbitNumber, numSat_max):
    print("Getting Constellation Configurations")
    orbitID = 0
    ## total number of satellites at once
    for i in list(range(1, numSat_max+1)):
        ## total number of orbits at once
        for j in list(range(1, i+1)):
            # every combination of orbits given total orbits at once (j)
            for combination in unique_combinations(orbitNumber, j):
                currOrbits = []
                # extract all orbits in given combination of orbits
                for k in combination:
                    currOrbits.append(orbit[k])
                # iterate through all possible distributions of satellites into current orbit combo
                for satDistr in partitions(i, j):
                    orbitDesign.append( orbitConfiguration(orbitID,currOrbits,satDistr,i) )
                    orbitID += 1
    print("Got Constellation Configurations")

def powerConfig(powerDesign, solarPanelSize_Range, batterySize_Range, laserPower_Range, apetureRad_range, powerReceiverRad_Range, P_per_kg):
    print("Getting Power Configurations")
    powerID = 0
    for batterySize in batterySize_Range:
        for solarPanelSize in solarPanelSize_Range:
            laserPower_Range = np.linspace(1000, P_per_kg*batterySize, 20)
            for laserPower in laserPower_Range:
                for apetRad in apetureRad_range:
                    for powerReceiverRad in powerReceiverRad_Range:
                        powerDesign.append( powerConfiguration(powerID,batterySize,solarPanelSize,laserPower,apetRad,powerReceiverRad) )
                        powerID += 1
    print("Got Power Configurations")

def commsConfig(commsDesign, diameterTxM, diameterTxO_Range, dataRate_Range, dataRate_ED_Range):
    print("Getting Comms Configurations")
    commsID = 0
    for diameterTxO in diameterTxO_Range:
        for dataRate in dataRate_Range:
            for dataRate_ED in dataRate_ED_Range:
                commsDesign.append( commsConfiguration(commsID, diameterTxM, diameterTxO, dataRate, dataRate_ED) )
                commsID += 1
    print("Got comms Configurations")

# Function to call drivers, can function in parallel
def driverCall(design):
    # powerOutput = power.driver(design)
    # if powerOutput == 0:
    #     design.powerObj = 0
    #     design.commsObj = [[0,0,0,0,0], 0]
    #     design.roiObj = 0
    #     design.add_constraint(0, 0, 0)
    #     return design
    powerOutput = 100
    print("Passed Power Feasibility")
    commsOutput, commsConstraint = comms.driver(design)
    roiConstraint = roi.driver(design, powerOutput, commsOutput)
    design.add_constraint(1, commsConstraint, roiConstraint)
    return design


def main():
    # Import orbitDict files
    orbitFiles = [f for f in listdir('trajectories') if isfile(join('trajectories', f))]
    orbit = []
    orbitNumber = []
    numOrbits = 0
    for i in range(len(orbitFiles)):
        filename = 'trajectories/' + orbitFiles[i]
        orbit.append(chosenOrbits.load(filename))
        orbitNumber.append(numOrbits)
        numOrbits += 1
    print("loaded orbit files")

    num_to_plot = 5 # number of orbits in family to plot
    orbit = orbit[::int(len(orbit)/num_to_plot)]
    orbitNumber = range(len(orbit))

    # Intermediate Constants
    Receiver_Reflectivity_Const = 0.1 # equal to 1 pane of glass
    Receiver_Intensity_Cutoff = 1380*450 # [W/m^2]
    P_per_kg = 1500 # [W/kg] (for Lithium Ion Battery)
    E_per_kg = 200 # [Wh/kg] (for Lithium Ion Battery)


    # Design Variables
    # diameterTxM = 1 # [m] antenna diameter of the receiver on the moon
    # batterySize_Range = np.linspace(200,5000,10)
    # solarPanelSize_Range = np.linspace(50, 1000, 10) # [m^2] solar panel area
    # laserPower_Range = [] # 1000  to  P_per_kg*LI_battery_mass [W]
    # apetureRad_range = np.linspace(0.01, 0.05, 10) # [m] radius of output lens on SC
    # powerReceiverRad_Range = np.linspace(1, 20, 10) # [m] radius of ground power receiver
    # diameterTxO_Range = np.linspace(1.3, 5, 5)
    # dataRate_Range = list(np.arange(3e6, 40e6, 2.5e5)) # [bps]
    # dataRate_ED_Range = dataRate_Range # [bps] desired data rate for earth downlink


    diameterTxM = 1 # [m] antenna diameter of the receiver on the moon
    batterySize_Range = np.linspace(200,5000,2)
    solarPanelSize_Range = np.linspace(50, 1000, 2) # [m^2] solar panel area
    laserPower_Range = [] # 1000  to  P_per_kg*LI_battery_mass [W]
    apetureRad_range = np.linspace(0.01, 0.05, 2) # [m] radius of output lens on SC
    powerReceiverRad_Range = np.linspace(1, 20, 2) # [m] radius of ground power receiver
    diameterTxO_Range = np.linspace(1.3, 5, 2)
    dataRate_Range = np.linspace(3e6, 40e6, 2) #list(np.arange(3e6, 40e6, 2.5e5)) # [bps]
    dataRate_ED_Range = dataRate_Range # [bps] desired data rate for earth downlink

    numSat_max = 3
    currTime = str(time.time())
    print("Begin design variable array creation (multiprocessing)" + currTime)
    ## Multiprocessing Call to separate large for loops
    # Begin orbit design variable array creation
    orbitDesign = multiprocessing.Manager().list()
    pOrbit = multiprocessing.Process(target=orbitConfig, args=(orbitDesign, orbit, orbitNumber, numSat_max,))
    pOrbit.start()

    # Begin power design variable array creation
    powerDesign = multiprocessing.Manager().list()
    pPower = multiprocessing.Process(target=powerConfig, args=(powerDesign, solarPanelSize_Range, batterySize_Range, laserPower_Range, apetureRad_range, powerReceiverRad_Range, P_per_kg,))
    pPower.start()

    # Begin comms design variable array creation
    commsDesign = multiprocessing.Manager().list()
    pComms = multiprocessing.Process(target=commsConfig, args=(commsDesign, diameterTxM, diameterTxO_Range, dataRate_Range, dataRate_ED_Range,))
    pComms.start()

    #Finish all multiprocessing before moving forward in code
    pOrbit.join()
    pPower.join()
    pComms.join()


    # Combine all unique subsystem design variables to create every possible unique combination of design variables
    currTime = str(time.time())
    print("Combine all unique design variables, " + currTime)
    # allDesigns = multiprocessing.Manager().list()
    currDesigns = []
    processes = []

    # orbitDesign1 = [orbitDesign[0],orbitDesign[1],orbitDesign[2]]
    # powerDesign1 = [powerDesign[0],powerDesign[1],powerDesign[2],powerDesign[3],powerDesign[4]]
    # commsDesign1 = [commsDesign[0],commsDesign[1],commsDesign[2],commsDesign[3],commsDesign[4]]

    orbitDesign1 = [orbitDesign[0]]
    powerDesign1 = [powerDesign[0]]
    commsDesign1 = [commsDesign[0]]

    for i in orbitDesign1:
        for j in powerDesign1:
            for k in commsDesign1:
                design = designConfiguration(ID = (i.orbitID, j.powerID, k.commsID), orbits = i.orbits, numSats = i.numSats, totSats = i.totSats, solarPanelSize = j.solarPanelSize, batterySize = j.batterySize, laserPower = j.laserPower, apetRad = j.apetRad, receiverRad_power = j.receiverRad_power, diameterTxM = k.diameterTxM, diameterTxO = k.diameterTxO, dataRate = k.dataRate, dataRate_ED = k.dataRate_ED)
                currDesigns.append(design)
                # p = multiprocessing.Process(target=driverCall, args = (allDesigns, design,))
                # p.start()
                # processes.append(p)
    # num_p = len(orbitDesign)*len(powerDesign)*len(commsDesign)
    import os
    currTime = str(time.time())
    print("Input design variables into model functions (multiprocessing), " + currTime)
    print(os.cpu_count())
    p = multiprocessing.Pool(os.cpu_count())
    allDesigns = p.map(driverCall, currDesigns)
    p.close()
    p.join()

    # Multiprocessing for many parallel instances of driver calls with unique design variables for each
    # print("Input design variables into model functions")
    # processes = []
    # for solution in allDesigns:
    #     p = multiprocessing.Process(target=driverCall, args = (solution,))
    #     p.start()
    #     processes.append(p)

    # Do not let code continue until multiprocessing completes
    # for process in processes:
    #     process.join()

    # Scoring
    print("Scoring")
    print(allDesigns[0].commsObj[0])
    # FOM min-max
    minComms00 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][0]
    maxComms00 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][0]
    print(maxComms00)
    minComms01 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][1]
    maxComms01 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][1]
    minComms02 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][2]
    maxComms02 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][2]
    minComms03 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][3]
    maxComms03 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][3]
    minComms04 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][4]
    maxComms04 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][4]

    minComms1 = min(allDesigns, key=lambda x: x.commsObj[1]).commsObj[1]
    maxComms1 = max(allDesigns, key=lambda x: x.commsObj[1]).commsObj[1]

    minPower = min(allDesigns, key=lambda x: x.powerObj).powerObj
    maxPower = max(allDesigns, key=lambda x: x.powerObj).powerObj

    minROI = min(allDesigns, key=lambda x: x.roiObj).roiObj
    maxROI = max(allDesigns, key=lambda x: x.roiObj).roiObj

    designScores = []
    for currDesign in allDesigns:
        # FOM Scoring
        comms00 = currDesign.commsObj[0][0]
        comms01 = currDesign.commsObj[0][1]
        comms02 = currDesign.commsObj[0][2]
        comms03 = currDesign.commsObj[0][3]
        comms04 = currDesign.commsObj[0][4]

        fom00 = (comms00-minComms00)/(maxComms00-minComms00)
        fom01 = (comms01-minComms01)/(maxComms01-minComms01)
        fom02 = (comms02-minComms02)/(maxComms02-minComms02)
        fom03 = (comms03-minComms03)/(maxComms03-minComms03)
        fom04 = (comms04-minComms04)/(maxComms04-minComms04)

        weightPC = 0.16
        weightMaxCG = 0.11
        weightMeanCG = 0.07
        weightTAR = 0.22
        weightMRT = 0.44
        fomScore = fom00*weightPC + fom01*weightMaxCG + fom02*weightMeanCG + fom03*weightTAR + fom04*weightMRT
        currDesign.commsObj[0] = fomScore

        comms1 = currDesign.commsObj[1]
        currDesign.commsObj[1] = (comms1-minComms1)/(maxComms1-minComms1)
        currDesign.commsObj = (currDesign.commsObj[0] + currDesign.commsObj[1])/2

        powerObj = currDesign.powerObj
        currDesign.powerObj = (powerObj-minPower)/(maxPower-minPower)

        roiObj = currDesign.roiObj
        currDesign.roiObj = (roiObj-minROI)/(maxROI-minROI)

        designScores.append([currDesign.powerObj, currDesign.commsObj, currDesign.roiObj])

    print("Save finalized designs with scores")
    with open("savedDesigns.dat", "rb") as f:
        pickle.dump(allDesigns, f)


    # Pareto
    print("Pareto plot")
    ## https://stackoverflow.com/questions/37000488/how-to-plot-multi-objectives-pareto-frontier-with-deap-in-python
    def simple_cull(inputPoints, dominates):
        paretoPoints = set()
        candidateRowNr = 0
        dominatedPoints = set()
        while True:
            candidateRow = inputPoints[candidateRowNr]
            inputPoints.remove(candidateRow)
            rowNr = 0
            nonDominated = True
            while len(inputPoints) != 0 and rowNr < len(inputPoints):
                row = inputPoints[rowNr]
                if dominates(candidateRow, row):
                    # If it is worse on all features remove the row from the array
                    inputPoints.remove(row)
                    dominatedPoints.add(tuple(row))
                elif dominates(row, candidateRow):
                    nonDominated = False
                    dominatedPoints.add(tuple(candidateRow))
                    rowNr += 1
                else:
                    rowNr += 1

            if nonDominated:
                # add the non-dominated point to the Pareto frontier
                paretoPoints.add(tuple(candidateRow))

            if len(inputPoints) == 0:
                break
        return paretoPoints, dominatedPoints

    def dominates(row, candidateRow):
        return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

    paretoPoints, dominatedPoints = simple_cull(designScores, dominates)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))
    print(pp.shape,dp.shape)
    ax.scatter(dp[:,0],dp[:,1],dp[:,2])
    ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')

    if len(pp[:,0]) > 2:
        import matplotlib.tri as mtri
        triang = mtri.Triangulation(pp[:,0],pp[:,1])
        ax.plot_trisurf(triang,pp[:,2],color='red')
    plt.show()
# Main
if __name__ == "__main__":
    main()