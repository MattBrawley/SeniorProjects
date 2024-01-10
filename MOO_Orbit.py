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
import os

import orbitDict
chosenOrbits = orbitDict.chosenOrbits
from design import designConfiguration, orbitConfiguration, powerConfiguration, commsConfiguration
from ROI_Model.ROI_Driver import roi
roi = roi()
from Communications_Model.Comms_Driver import comms
comms = comms()
from Power_Model.Power_Driver import power
power = power()


def main():
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
    # print(lengthOfFiles, 'designs being tested')

    print(allDesigns[42].powerObj)
    print(allDesigns[42].commsObj)
    print(allDesigns[42].roiObj)



    # Scoring
    # print("Scoring")
    # print('First design has', allDesigns[0].totSats, 'satellites')
    # print('Last design has', allDesigns[lengthOfFiles-1].totSats, 'satellites')
    # FOM min-max
    minComms00 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][0]
    maxComms00 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][0]
    minComms01 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][1]
    maxComms01 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][1]
    minComms02 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][2]
    maxComms02 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][2]
    minComms03 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][3]
    maxComms03 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][3]
    minComms04 = max(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][4]
    maxComms04 = min(allDesigns, key=lambda x: x.commsObj[0]).commsObj[0][4]

    # print('minComms01:', minComms01)
    # print('maxComms01:', maxComms01)
    # print('minComms02:', minComms02)
    # print('maxComms02:', maxComms02)
    # print('minComms03:', minComms03)
    # print('maxComms03:', maxComms03)
    # print('minComms04:', minComms04)
    # print('maxComms04:', maxComms04)

    minComms1 = min(allDesigns, key=lambda x: x.commsObj[1]).commsObj[1]
    maxComms1 = max(allDesigns, key=lambda x: x.commsObj[1]).commsObj[1]
    # print('minComms1:', minComms1)
    # print('maxComms1:', maxComms1)

    minPower = min(allDesigns, key=lambda x: x.powerObj).powerObj
    maxPower = max(allDesigns, key=lambda x: x.powerObj).powerObj
    # print('minPower:', minPower)
    # print('maxPower:', maxPower)

    minROI = min(allDesigns, key=lambda x: x.roiObj).roiObj
    maxROI = max(allDesigns, key=lambda x: x.roiObj).roiObj
    # print('minROI:', minROI)
    # print('maxROI:', maxROI)

    for currDesign in allDesigns:
        # FOM Scoring
        comms00 = currDesign.commsObj[0][0]
        comms01 = currDesign.commsObj[0][1]
        comms02 = currDesign.commsObj[0][2]
        comms03 = currDesign.commsObj[0][3]
        comms04 = currDesign.commsObj[0][4]
        # # print(comms00)
        # # print(comms01)
        # # print(comms02)
        # # print(comms03)
        # # print(comms04)
        # #
        fom00 = (comms00-minComms00)/(maxComms00-minComms00)
        fom01 = (comms01-minComms01)/(maxComms01-minComms01)
        fom02 = (comms02-minComms02)/(maxComms02-minComms02)
        fom03 = (comms03-minComms03)/(maxComms03-minComms03)
        fom04 = (comms04-minComms04)/(maxComms04-minComms04)
        # # print(fom00)
        # # print(fom01)
        # # print(fom02)
        # # print(fom03)
        # # print(fom04)
        # #
        weightPC = 0.16
        weightMaxCG = 0.11
        weightMeanCG = 0.07
        weightTAR = 0.22
        weightMRT = 0.44
        fomScore = fom00*weightPC + fom01*weightMaxCG + fom02*weightMeanCG + fom03*weightTAR + fom04*weightMRT
        currDesign.commsObj[0] = fomScore
        #
        comms1 = currDesign.commsObj[1]
        currDesign.commsObj[1] = (comms1-minComms1)/(maxComms1-minComms1)
        # print('BRO:', currDesign.commsObj[1])
        currDesign.commsObj = (currDesign.commsObj[0] + currDesign.commsObj[1])/2
        # currDesign.commsObj = currDesign.commsObj[1]
        # print('comms1:', comms1)
        #
        #
        powerObj = currDesign.powerObj
        currDesign.powerObj = (powerObj-minPower)/(maxPower-minPower)
        # print('powerObj:', powerObj)
        #
        roiObj = currDesign.roiObj
        currDesign.roiObj = (roiObj-minROI)/(maxROI-minROI)
        # print('roiObj:', roiObj)

        designScores.append([round(currDesign.powerObj,7), round(currDesign.commsObj, 7), round(currDesign.roiObj, 7)])

    print(allDesigns[42].powerObj)

    def read_list():
        # for reading also binary mode is important
        with open('CRATER_MOO_DESIGN', 'rb') as fp:
            n_list = pickle.load(fp)
            return n_list

    # list of names

    bro = read_list()
    # print(bro[10000].powerObj)

    allDesigns = allDesigns+bro #######################################################################################################################################################################################################################




    for currDesign in bro:
        designScores.append([round(currDesign.powerObj,7), round(currDesign.commsObj, 7), round(currDesign.roiObj, 7)])


    def write_list(a_list):
        # store list in binary file so 'wb' mode
        with open('listfile', 'wb') as fp:
            pickle.dump(a_list, fp)
            # print('Done writing list into a binary file')

    # # Read list to memory
    # def read_list():
    #     # for reading also binary mode is important
    #     with open('listfile', 'rb') as fp:
    #         n_list = pickle.load(fp)
    #         return n_list

    # list of names
    # write_list(allDesigns)


    # Pareto
    # print('')
    # print("Pareto plot")
    ## https://stackoverflow.com/questions/37000488/how-to-plot-multi-objectives-pareto-frontier-with-deap-in-python
    def simple_cull(inputPoints, dominates):
        inputPoints_tmp = inputPoints[:]
        paretoIndex = []  # Actually the row number of the good designs in the complete list of designs
        paretoPoints = set()
        candidateRowNr = 0
        dominatedPoints = set()
        while True:
            candidateRow = inputPoints_tmp[candidateRowNr]
            inputPoints_tmp.remove(candidateRow)
            rowNr = 0
            nonDominated = True
            while len(inputPoints_tmp) != 0 and rowNr < len(inputPoints_tmp):
                row = inputPoints_tmp[rowNr]
                if dominates(candidateRow, row):
                    # If it is worse on all features remove the row from the array
                    inputPoints_tmp.remove(row)
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
                paretoIndex.append(candidateRow) # values in pp


            if len(inputPoints_tmp) == 0:
                break
        return paretoPoints, dominatedPoints, paretoIndex

    def dominates(row, candidateRow):
        return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

    paretoPoints, dominatedPoints, paretoIndex = simple_cull(designScores, dominates)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))  # points on pareto
    # print(pp.shape,dp.shape)
    # ax.scatter(dp[:,0],dp[:,1],dp[:,2])
    # ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
    # ax.set_xlabel('Power')
    # ax.set_ylabel('Comms')
    # ax.set_zlabel('ROI')
    #
    # if len(pp[:,0]) > 2:
    #     import matplotlib.tri as mtri
    #     triang = mtri.Triangulation(pp[:,0],pp[:,1])
    #     ax.plot_trisurf(triang,pp[:,2],color='red')

    # print('pp', pp)
    # print('paretoIndex:', paretoIndex)  # values in pp

    index = []
    GoodIDs = []

    for i in range(len(paretoIndex)):
        index.append(designScores.index(paretoIndex[i]))
    # print('index in design score:', index)  # index in designScores

    for i in index:
        GoodIDs.append(bro[i].ID)
    # print('GoodIDs:', GoodIDs)  # design IDs which are on pareto
    # print('')
    # print(len(GoodIDs),' designs on the pareto surface')

    GoodDesigns = []
    for i in index:
        GoodDesigns.append(allDesigns[i].roiObj)

    maxROIguy = max(GoodDesigns)
    # print('GoodDesigns', GoodDesigns)
    maxROIindex = GoodDesigns.index(maxROIguy)
    # print('maxROIindex', maxROIindex)

    # NEED:
    # from csltk.utilities import System
    # import matplotlib.pyplot as plt
    # import orbitDict
    idx = maxROIindex
    # sys = System(mu=0.01215058560962404, lstar=389703.2648292776, tstar=382981.2891290545)
    # ax = sys.plot_system()
    design = allDesigns[idx]
    print(design.powerObj)
    powerOutput = power.driver(currDesign=design)
    print("power output")
    print(powerOutput)
    commsOutput, commsConstraint = comms.driver(currDesign)
    roiConstraint = roi.driver(currDesign, powerOutput, commsOutput)

    # print('Printing Design', design.ID)
    # orbits = design.orbits
    # print('Constellation Parameters:')
    # print('Number of Orbits:', len(orbits), 'Number of Satellites', design.totSats)
    # counter = 0
    # for orb in orbits:
    #     ax.plot(orb.x, orb.y, orb.z)
    #     print('Family:', orb.family, ', Sats on Orbit:', design.numSats[counter], ', Stability:', orb.stability,', Period:', orb.T)
    #     counter += 1
    #
    # print('Power Parameters:')
    # print('Solar Panel Size:', design.solarPanelSize, ', Battery Size:', design.batterySize, ', Laser Power:', design.laserPower)
    # print('Output Lense:', design.apetRad, ', Ground Reciever:', design.receiverRad_power)
    # print('Comms Parameters:')
    # print('Antenna Diameter of the Receiver on the Moon:', design.diameterTxM,', antenna diameter of the receiver on the satellite:', design.diameterTxO)
    # print('lunar data rate:', design.dataRate, ', data rate for earth downlink', design.dataRate_ED)
    # ax.set_aspect('auto')
    # plt.show()



    # plt.show()

# Main
if __name__ == "__main__":
    main()