# FILE DESCRIPTION
# Used to test the manually obtained designs and toss them on the pareto

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
import pickle

import orbitDict
chosenOrbits = orbitDict.chosenOrbits
from design import designConfiguration, orbitConfiguration, powerConfiguration, commsConfiguration
# from csltk.utilities import System
# import matplotlib.pyplot as plt
# from ROI_Model.ROI_Driver import roi
# roi = roi()
# from Communications_Model.Comms_Driver import comms
# comms = comms()
# from Power_Model.Power_Driver import power
# power = power()




def main():
    allDesigns=[]
    designScores = []
    # # Import orbitDict files
    # files = [131073, 131074, 131075, 131076, 131077, 131078, 131079, 131080, 131081, 131082, 131083, 131084, 131085, 131086, 131087, 131088, 131089, 131090, 131091, 131092, 131093, 131094, 131095, 131096, 131097, 131098, 131099, 131100, 131101, 131102, 131103, 131104, 131105, 131106, 131107, 131108, 211201, 221201, 611201, 621201, 781201]
    # lengthOfFiles = len(files)
    # print(lengthOfFiles, 'designs being tested')
    # for i in files:
    #     filename = 'LunarDesigns/design' + str(i) + '.dat'
    #     allDesigns.append(chosenOrbits.load(filename))

    files = [f for f in listdir('LunarDesigns') if isfile(join('LunarDesigns', f))]
    for i in range(len(files)):
        filename = 'LunarDesigns/' + files[i]
        if filename != 'LunarDesigns/.DS_Store':
            allDesigns.append(chosenOrbits.load(filename))

    lengthOfFiles = len(files)-1
    print('|||', lengthOfFiles, 'Designs being tested |||')
    print('')

    ####################################################################################################################

    print('Length of allDesigns: ', len(allDesigns))
    # Highest ROI rn: ID = 1900003, idx = 1798, 1 orbit
    # Highest Power rn: ID = 7853070, idx = 683, 3 orbits
    # Highest Comms rn: ID = 5583062, idx = 665, 1 orbit
    # Highest Overall rn: ID = 17032260, idx = 1081, 2 orbits

    #indexes: [, , , , , , , , , , , , , , , , , , , ,
              # , , , , , , , , , , , , , , , , ,
              # , ]

    designWanted = 42

    print('Index =', designWanted)
    print('ID =', allDesigns[designWanted].ID)

    print('------------------First orbit------------------')
    print('orbits.family =', allDesigns[designWanted].orbits[0].family)
    # print('orbits.x =', allDesigns[designWanted].orbits[0].x)
    # print('orbits.y =', allDesigns[designWanted].orbits[0].y)
    # print('orbits.z =', allDesigns[designWanted].orbits[0].z)
    # print('orbits.vx =', allDesigns[designWanted].orbits[0].vx)
    # print('orbits.vy =', allDesigns[designWanted].orbits[0].vy)
    # print('orbits.vz =', allDesigns[designWanted].orbits[0].vz)
    print('orbits.T =', allDesigns[designWanted].orbits[0].T)
    print('orbits.eclipse =', allDesigns[designWanted].orbits[0].eclipse)
    print('orbits.stability =', allDesigns[designWanted].orbits[0].stability)
    print('------------------------------------')
    print('')

    # print('------------------Second orbit------------------')
    # print('orbits.family =', allDesigns[designWanted].orbits[1].family)
    # # print('orbits.x =', allDesigns[designWanted].orbits[1].x)
    # # print('orbits.y =', allDesigns[designWanted].orbits[1].y)
    # # print('orbits.z =', allDesigns[designWanted].orbits[1].z)
    # # print('orbits.vx =', allDesigns[designWanted].orbits[1].vx)
    # # print('orbits.vy =', allDesigns[designWanted].orbits[1].vy)
    # # print('orbits.vz =', allDesigns[designWanted].orbits[1].vz)
    # print('orbits.T =', allDesigns[designWanted].orbits[1].T)
    # print('orbits.eclipse =', allDesigns[designWanted].orbits[1].eclipse)
    # print('orbits.stability =', allDesigns[designWanted].orbits[1].stability)
    # print('------------------------------------')
    # print('')
    #
    # print('------------------Third orbit------------------')
    # print('orbits.family =', allDesigns[designWanted].orbits[2].family)
    # # print('orbits.x =', allDesigns[designWanted].orbits[2].x)
    # # print('orbits.y =', allDesigns[designWanted].orbits[2].y)
    # # print('orbits.z =', allDesigns[designWanted].orbits[2].z)
    # # print('orbits.vx =', allDesigns[designWanted].orbits[2].vx)
    # # print('orbits.vy =', allDesigns[designWanted].orbits[2].vy)
    # # print('orbits.vz =', allDesigns[designWanted].orbits[2].vz)
    # print('orbits.T =', allDesigns[designWanted].orbits[2].T)
    # print('orbits.eclipse =', allDesigns[designWanted].orbits[2].eclipse)
    # print('orbits.stability =', allDesigns[designWanted].orbits[2].stability)
    # print('------------------------------------')
    # print('')

    print('------------------Constellation Information------------------')
    print('numSats =', allDesigns[designWanted].numSats)
    print('totSats =', allDesigns[designWanted].totSats)
    print('solarPanelSize =', allDesigns[designWanted].solarPanelSize)
    print('batterySize =', allDesigns[designWanted].batterySize)
    print('laserPower =', allDesigns[designWanted].laserPower)
    print('apetRad =', allDesigns[designWanted].apetRad)
    print('receiverRad_power =', allDesigns[designWanted].receiverRad_power)
    print('diameterTxM =', allDesigns[designWanted].diameterTxM)
    print('diameterTxO =', allDesigns[designWanted].diameterTxO)
    print('dataRate =', allDesigns[designWanted].dataRate)
    print('dataRate_ED =', allDesigns[designWanted].dataRate_ED)
    print('commsObj =', allDesigns[designWanted].commsObj)  # [[weighted average of ] , constraints]
    print('powerObj =', allDesigns[designWanted].powerObj)
    print('roiObj =', allDesigns[designWanted].roiObj)
    print('constraint =', allDesigns[designWanted].constraint)
    print('------------------------------------')
    print('')
    ####################################################################################################################

    # ####################################################################################################################
    # print('Return on Investment over Time')
    years = np.linspace(0, 50, num=51)
    a1 = 17  # max number of customer
    b1 = 15  # delay start in curvature beginning
    c1 = 0.15  # growth rate of curve
    # cislunar_customers = round(a * exp(-b * exp(-c * x))); <- MATLAB

    cislunar_customers = np.around(a1*math.e**(-b1*math.e**(-c1*years)))

    cust_needs_power = 60  # [kWh] daily consumption
    cust_needs_comms = 10 ** 3  # [GB] daily consumption (curiosity uses 100-250 megabits a day, person uses ~34 gb, one hour of netflix is 1 gb hour)
    # ^ Can randomise for more accurate model
    price_power = 95  # [$/kWh]
    price_comms = 30  # [$/GB]

    cost = allDesigns[designWanted].roiObj
    daily_power_provided = allDesigns[designWanted].powerObj
    daily_comms_provided = allDesigns[designWanted].commsObj[0][0] ###### Help with this one, not sure if calculating right comms figure

    # synodicPeriod = 29.523  # days
    # daily_power_provided = E_received / E_period
    # daily_comms_provided = commsOutput / synodicPeriod

    rev_power = daily_power_provided * 365 * price_power
    rev_comms = daily_comms_provided * 365 * price_comms
    yearlyRevenue = rev_power + rev_comms


    print('Return on Investment (Revenue / Cost) :', yearlyRevenue*50*cislunar_customers[-1] / cost)

    ###################################################### Trying to figure out how many years til breakeven
    sum = 0
    for g in range(len(years)):  # every year for 50 years
        revenue = yearlyRevenue * cislunar_customers[g]
        sum = sum + revenue
        if sum > cost:
            print('Design reaches Book Breakeven after', g, 'years')


    # print('Return on Investment (Revenue / Cost) :', yearlyRevenue * /cost)

    # plt.plot(year, revenue)
    # plt.xlabel('Year')
    # plt.ylabel('Revenue ($)')
    # plt.title('Revenue vs Year')
    # plt.show()

    ######################################################

    #####################################################################################################################

    # Plotting orbits
    ####################################################################################################################

    orbits = allDesigns[designWanted].orbits
    sys = System(mu=0.01215058560962404, lstar=389703.2648292776, tstar=382981.2891290545)
    fig = plt.figure()
    ax = sys.plot_system()
    for orb in orbits:
        ax.plot(orb.x, orb.y, orb.z)
        ax.set_aspect('auto')

    pickle.dump(fig, open('Good1.fig.pickle','wb'))
    plt.show()
    print('')
    ####################################################################################################################


    # Scoring
    print("Scoring")
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
        roiObj = -(currDesign.roiObj)
        currDesign.roiObj = (roiObj-minROI)/(maxROI-minROI) + 2
        # print('roiObj:', roiObj)

        designScores.append([round(currDesign.powerObj,7), round(currDesign.commsObj, 7), round(currDesign.roiObj, 7)])

    ###########################
    # def read_list():
    #     # for reading also binary mode is important
    #     with open('CRATER_MOO_DESIGN', 'rb') as fp:
    #         n_list = pickle.load(fp)
    #         return n_list
    #
    # # list of names
    #
    # bro = read_list()
    # # print(bro[10000].powerObj)
    #
    # allDesigns = allDesigns+bro #######################################################################################################################################################################################################################

    # time.sleep(100)
    # for currDesign in bro:
    #     designScores.append([round(currDesign.powerObj,7), round(currDesign.commsObj, 7), round(currDesign.roiObj, 7)])
    ###########################

    # time.sleep(100)
    def write_list(a_list):
        # store list in binary file so 'wb' mode
        with open('listfile', 'wb') as fp:
            pickle.dump(a_list, fp)
            print('Done writing list into a binary file')

    # # Read list to memory
    # def read_list():
    #     # for reading also binary mode is important
    #     with open('listfile', 'rb') as fp:
    #         n_list = pickle.load(fp)
    #         return n_list

    # list of names
    # write_list(allDesigns)


    # Pareto
    print('')
    print("Pareto Plot")
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

    # import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))  # points on pareto
    print(pp.shape,dp.shape)
    ax.scatter(dp[:,0],dp[:,1],dp[:,2])
    ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
    ax.set_xlabel('Power')
    ax.set_ylabel('Comms')
    ax.set_zlabel('ROI')

    # Remove tick values but keep the labels
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    #

    ##### UNCOMMMENT FOR SURFACE #####
    # if len(pp[:,0]) > 2:
    #     import matplotlib.tri as mtri
    #     triang = mtri.Triangulation(pp[:,0],pp[:,1])
    #     ax.plot_trisurf(triang,pp[:,2],color='red')
    ##################################

    print('New Length of allDesigns:', len(allDesigns))
    print('')

    index = []
    GoodIDs = []

    for i in range(len(paretoIndex)):
        index.append(designScores.index(paretoIndex[i]))
    print('index in design score:', index)  # index in designScores

    for i in index:
        GoodIDs.append(allDesigns[i].ID)
    print('Design IDs on Pareto Surface:', GoodIDs)  # design IDs which are on pareto
    print('')
    print(len(GoodIDs),' designs on the pareto surface')
    print('')


    print('Analyzing Pareto Surface Designs:')
    print('')


    ##### Finding highest ROI ##########################################################################################
    print('Finding highest ROI')
    roiScores = []
    roiID = []
    for k in index:
        roiScores.append(allDesigns[k].roiObj)
        roiID.append(allDesigns[k].ID)

    print('Max ROI Score', max(roiScores))
    max_index = roiScores.index(max(roiScores))
    print('Max ROI index: ', index[max_index])
    print('Max ROI ID value in allDesigns', roiID[max_index])
    ####################################################################################################################
    print('')


    ##### Finding highest Power ########################################################################################
    print('Finding highest Power')
    powerScores = []
    powerID = []
    for k in index:
        powerScores.append(allDesigns[k].powerObj)
        powerID.append(allDesigns[k].ID)

    print('Max Power Score', max(powerScores))
    max_index = powerScores.index(max(powerScores))
    print('Max Power index: ', index[max_index])
    print('Max Power ID value in allDesigns', powerID[max_index])
    ####################################################################################################################
    print('')


    ##### Finding highest Comms ########################################################################################
    print('Finding highest Comms')
    commsScores = []
    commsID = []
    for k in index:
        commsScores.append(allDesigns[k].commsObj[0])
        commsID.append(allDesigns[k].ID)

    print('Max Comms Score', max(commsScores))
    max_index = commsScores.index(max(commsScores))
    print('Max Comms index: ', index[max_index])
    print('Max Comms ID value in allDesigns', commsID[max_index])
    ####################################################################################################################
    print('')

    ##### Finding highest magnitude from origin ########################################################################
    print('Finding Highest Magnitude from Origin')
    overallScores = []
    overallID = []
    for k in index:
        mag = allDesigns[k].commsObj**2 + allDesigns[k].powerObj**2 + allDesigns[k].roiObj**2
        overallScores.append(mag)
        overallID.append(allDesigns[k].ID)

    print('Max Overall Score', max(overallScores))
    max_index = overallScores.index(max(overallScores))
    print('Max Overall index: ', index[max_index])
    print('Max Overall ID value in allDesigns', overallID[max_index])
    ####################################################################################################################
    print('')


    # print('pp', pp)
    # print('paretoIndex:', paretoIndex)  # values in pp

    pickle.dump(fig, open('Pareto.fig.pickle','wb'))
    plt.show()


# Main
if __name__ == "__main__":
    main()
