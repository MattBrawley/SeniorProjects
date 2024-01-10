from re import T
import numpy as np
from cmath import log10
from math import pi
import matplotlib.pyplot as plt
import time 
from csltk.utilities import System

# Move path to main CRATER directory
import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current) 
# adding the parent directory to the sys.path.
sys.path.append(parent)
 
from orbitDict import chosenOrbits
from design import designConfiguration

class comms:
    def SpaceLoss(self, distance,wavelength):
        #Inputs: Distance in METERS & Wavelength in Hz
        #Outputs: Spaceloss in DB
        #Method: Using FSPL formula

        loss = -1*10*log10(((4*pi*distance)/wavelength)**2)
        return(loss.real) #return only the real part

    def AntennaParameters(self, D, wavelength):
        #Inputs: Diameter (meters) Wavelength (Hz)
        #Outputs: Gain in dB, area in m^2 (trivial calculation but sped up with function implementation)
        #Method: use a form to find gain that can be sampled via
        #diameter and wavelength

        k = 1.38e-23 #boltzman's constant
        area = pi*(D/2)**2
        beamwidth = 70 * (wavelength/D) 
        area_eff = area*.55 # = effective area simplified formula
        gain = 10*log10((6.9*area_eff)/(wavelength**2))
        return [gain.real, area, area_eff, beamwidth]

    def FOM(self, orbits, numSats):
        def reorder(lst, newIdx):
            pos = newIdx #lst.index(first)
            return lst[pos:] + lst[:pos]

        start = time.time()

        ## inputs are a design 
        ## takes in list of orbits with corresponding number of sats per orbit 
        ###############################################
        ################# Constants ###################
        ###############################################

        TU = 382981.2891290545 # seconds/TU
        LU = 389703 # 1 LU (distance from Earth to Moon in km)
        moon_rad = 1740/LU # radius of the Moon in [km]
        moonPos = [0.98784942,0,0] # LU 
        lat_spacing = 10 # [deg]
        max_points = 12 
        latitude = np.linspace(90,-90,19)
        gap = 360/max_points 
        synodicPeriod = 29.523 # days 
        synodicS = synodicPeriod*24*60*60 # s 
        sToHr = (1/(60*60))

        ################################################
        ############## Grid point Calcluation ##########
        ################################################

        points = []
        counter = 0
        # northern hemisphere 
        for i in latitude: 
            radians = np.radians(i)
            points += [abs(round(max_points*np.cos(radians)))]
            counter = counter+1

        totalPoints = sum(points)
        print('Simulating',totalPoints,'grid points...')
        time.sleep(2)
        # ax = plt.axes(projection = '3d')
        sys = System(mu=0.01215058560962404, lstar=389703.2648292776, tstar=382981.2891290545)
        ax = sys.plot_system()
        coordinates = []
        i = 0
        for point in points: 
            if point == 1 or point == 0: 
                longitude = np.linspace(0,360,point)
            else: 
                longitude = np.linspace(0,(360/point)*(point-1),point)
            
            # offset longitude by 45 deg every sencond latitude 
            if not i % 2: 
                longitude = longitude + 45 
                for long in longitude:
                    if long > 360: 
                        long = long - 360

            lat = latitude[i]
            theta = -lat + 90
            for long in longitude:
                z = moon_rad*np.cos(np.radians(theta))
                x = moon_rad*np.sin(np.radians(theta))*np.cos(np.radians(long))
                y = moon_rad*np.sin(np.radians(theta))*np.sin(np.radians(long))
                r = np.sqrt(x**2 + y**2 + z**2)
                phi = long
                data = [x,y,z,r,theta,phi,lat,long] # LU LU LU LU deg deg deg deg 
                #ax.scatter(x,y,z)
                coordinates.append(data)
            i = i + 1 


        #################################################
        ############# Orbit Simulation #################
        #################################################


        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # filename = 'trajectories/orbit76.dat'
        # orb1 = chosenOrbits.load(filename)
        # filename = 'trajectories/orbit5.dat'
        # orb2 = chosenOrbits.load(filename)
        # filename = 'trajectories/orbit45.dat'
        # orb3 = chosenOrbits.load(filename)
        # orbits = [orb1,orb2,orb3]
        # numSats = [2,2,2]

        # periodTU = orb1.T # TU
        # periodDays = periodTU*TU*(1/60)*(1/60)*(1/24) # [days]
        loops = round(synodicS/100)
        rows = totalPoints
        cols = loops
        rotate = 360/(synodicS/100) # deg to rotate per loop 
        coverageMain = np.zeros((rows,cols))
        counter = 0 
        # go through each orbit

        # a = 1

        for orb in orbits:
            
            coverage = np.zeros((rows,cols)) # rows = points, cols = time 
            timeCounterCoverage = 0 # restart time counter goes with loops 
            counterLoops = 0 # restarting orbits counter 
            counterRotate = 0 # 
            if not numSats[counter] == 0:
                for i in range(loops): 
                    
                    # allocate sat position 
                    if i >= (len(orb.x)*counterLoops + len(orb.x)):
                        counterLoops = counterLoops + 1
                    i = i - len(orb.x)*counterLoops
                    sat_xPos = orb.x[i]
                    sat_yPos = orb.y[i]
                    sat_zPos = orb.z[i]


                    # go through each grid point 
                    for k in range(totalPoints):
                        currentPoint = coordinates[k]
                        point_r = currentPoint[3]
                        point_theta = currentPoint[4]
                        point_phi = currentPoint[5] + counterRotate*rotate
                        
                        point_zPos = point_r*np.cos(np.radians(point_theta))
                        point_xPos = point_r*np.sin(np.radians(point_theta))*np.cos(np.radians(point_phi)) + 0.98784942 
                        point_yPos = point_r*np.sin(np.radians(point_theta))*np.sin(np.radians(point_phi))


                        r_point = [point_xPos, point_yPos, point_zPos] # vector from center of earth to grid point 
                        r_spacecraft = [sat_xPos, sat_yPos, sat_zPos] # vector from center of earth to satellite 
                        r_spacecraftToPoint = [point_xPos - sat_xPos, point_yPos - sat_yPos, point_zPos - sat_zPos] # vector sat to point
                        r_moonToPoint = [r_point[0]-moonPos[0],r_point[1]-moonPos[1],r_point[2]-moonPos[2]]# vector from center of moon to point 
                        r_moonToSat = [r_spacecraft[0]-moonPos[0],r_spacecraft[1]-moonPos[1],r_spacecraft[2]-moonPos[2]] # vector from moon center to sat 

                        angle1 = np.arccos((np.dot(r_spacecraftToPoint,r_moonToPoint)/(np.linalg.norm(r_spacecraftToPoint)*np.linalg.norm(r_moonToPoint))))
                        angle2 = np.arccos((np.dot(r_moonToSat,r_moonToPoint)/(np.linalg.norm(r_moonToSat)*np.linalg.norm(r_moonToPoint))))
                        
                        if (int(angle1) > (np.pi / 2) and int(angle2) < (np.pi / 2)) or (coverage[k,timeCounterCoverage] == 1):
                            coverage[k,timeCounterCoverage] = 1
                            # print('k, timecounter, coverage ',k,timeCounterCoverage,coverage[k][timeCounterCoverage])    
                        else:
                            coverage[k,timeCounterCoverage] = 0
        
                    counterRotate = counterRotate+1 
                    timeCounterCoverage = timeCounterCoverage+1 

                ################ Phasing multiple satellites  ########################
                
                satellites = numSats[counter]
                if not satellites == 1:
                    satIDXstep = round(loops/satellites)
                    for a in range(satellites):
                        for b in range(totalPoints): 
                            point = coverage[b,:]
                            point = reorder(list(point),a*satIDXstep)
                            for c in range(loops): 
                                if point[c] == 1: 
                                    coverageMain[b,c] = 1
                if np.count_nonzero(coverageMain) == loops*totalPoints: 
                    print('working')
                    break          
                        
            counter = counter + 1
                


        #  def reorder(lst, newIdx):
        #     pos = newIdx #lst.index(first)
        #     return lst[pos:] + lst[:pos]       
            
        print('Calculating FOM...')
        ###################################################
        ############## calculate FOM ######################     done once per design 
        ###################################################
        if np.count_nonzero(coverageMain) == loops*totalPoints:
            percentCoverage = np.zeros(totalPoints) + 100 # % covered
            maxCoverageGap = np.zeros(totalPoints) # longest coverage gap by each point 
            meanCoverageGap = np.zeros(totalPoints) # average coverage gap for each point 
            timeAvgGap = np.zeros(totalPoints) # time avg gap for each point 
            meanResponseTime = np.zeros(totalPoints) 
        else: 
            percentCoverage = np.zeros(totalPoints) # % covered 
            maxCoverageGap = np.zeros(totalPoints) # longest coverage gap by each point 
            meanCoverageGap = np.zeros(totalPoints) # average coverage gap for each point 
            timeAvgGap = np.zeros(totalPoints) # time avg gap for each point 
            meanResponseTime = np.zeros(totalPoints) 
            for i in range(totalPoints):

                percentCoverage[i] = (np.count_nonzero(coverageMain[i,:])/loops)*100 # percent of time each grid point is seen [%]
                
                if np.count_nonzero(coverageMain[i,:]) == loops:
                    #print('all 1')
                    timeAvgGap[i] = 0 
                    meanCoverageGap[i] = 0 
                    maxCoverageGap[i] = 0
                    meanResponseTime[i] = 0 
                elif np.count_nonzero(coverageMain[i,:]) == 0:
                    #print('all 0')
                    meanCoverageGap[i] = loops*100 # mean coverage gap for grid point i [s]
                    timeAvgGap[i] = loops*100 # [s]
                    maxCoverageGap[i] = loops*100 # seconds of maximum coverage gap for grid point i [s]
                    meanResponseTime[i] = loops*100 # s
                else:
                    #print('mix')
                    counterMCG = 0 # counter consecutive 0s, coverage gap 
                    coverageGaps = [] # length of coverage gaps  
                    meanRT = []
                    for j in range(loops):
                        # if coverage[i,j] == 1: 
                        #     print(i,coverage[i,j])
                        if coverageMain[i,j] == 0: # if uncovered 
                            counterMCG = counterMCG + 1 # +1 counter of gap 
                            meanRT.append(counterMCG)  
                        else: 
                            if not counterMCG == 0:
                                coverageGaps.append(counterMCG)
                                counterMCG = 0  
                    if not counterMCG == 0:
                        coverageGaps.append(counterMCG)               
                    numGaps = len(coverageGaps)
                    if numGaps == 0: 
                        numGaps = 1
                    meanCoverageGap[i] = (sum(coverageGaps)/numGaps)*100 # mean coverage gap for grid point i [s]
                    timeAvgGap[i] = (sum(np.array(coverageGaps)**2)/loops)*100 # [s]
                    maxCoverageGap[i] = max(coverageGaps)*100 # seconds of maximum coverage gap for grid point i [s]
                    meanResponseTime[i] = (sum(meanRT)/loops)*100 # s
                    
        ###############################################
        #################### Score / Return ####################
        ###############################################

        TimeLOS = synodicS*np.mean(percentCoverage)/100, 

        return TimeLOS, np.mean(percentCoverage), np.mean(meanCoverageGap), np.mean(timeAvgGap), np.mean(maxCoverageGap), np.mean(meanResponseTime)

    def driver(self, currDesign):
        #Inputs: Diameter_TxM = antenna diameter of the reciever on the moon in meters
        #DiameterTxO = antenna diameter of the reciever on the sattelite in meters
        #Freq = frequency in Hz
        #DataRate = desired data rate in bps
        #DataRateED = desired data rate for earth downlink (Keep this above 10e6)
        #Range = distance between Tx & Rx in m
        #Range_Sidelink = distance between sattelites in m
        #Time LOS = #of seconds over a target per period

        LU = 389703 # 1 LU (distance from Earth to Moon in km)
        distMoon = [0.98784942, 0, 0] # moon pos in LU
        rMoon = 1737.4/LU # radius of the moon in LU
        orbits = currDesign.orbits
        numSats = currDesign.numSats
        alt = 0
        for orbit in orbits:
            posX = orbit.x
            posY = orbit.y
            posZ = orbit.z
            currAlt = np.sqrt( (posX - distMoon[0])**2 + (posY - distMoon[1])**2 + (posZ - distMoon[2])**2) - rMoon # distance between satellite and moon center
            currAlt = max(currAlt)
            if currAlt > alt:
                alt = currAlt

        alt = alt*389703000 #convert from LU to m  

        Range = alt
        Range_Sidelink = 114039473.3696403 # m

        Diameter_TxM = currDesign.diameterTxM
        DiameterTxO = currDesign.diameterTxO
        DataRate = currDesign.dataRate
        DataRateED = currDesign.dataRate_ED

        # Call FOM

        TimeLOS, percentCoverage, meanCoverageGap, timeAvgGap, maxCoverageGap, meanResponseTime = self.FOM(orbits, numSats)
        coverage = [percentCoverage, meanCoverageGap, timeAvgGap, maxCoverageGap, meanResponseTime]

        #Outputs:
        #margin = 4 element of array of the link margin of all 4 links in order (Lunar Uplink, Lunar downlink, Cross Link, Earth downlink)
        #margin_check = 0 or 1 if the margin will be 6db or over, same order as margin
        #data amount = data amount of the link per day
        
        EbNo = 9.8 #dB change this if BER changes
        Efficiency = .4 #Ka deployable mesh antennas are not smooth
        DesignMargin = 6 #dB - hihg confidence with 6db Margin, complete confidence with 10db link margin
        boltzman = 1.38e-23

        #Preallocating arrays
        margin = []
        margin_check = []
        data_amount = []
        DRS_list = []
        constraints = 1

        #print(TimeLOS)
        #print(DataRate)
        #TimeLOS = float(TimeLOS)
        amnt = DataRate * TimeLOS #this is the data amount that can be transmitted per period

        
        DiameterRx = DiameterTxO
        index = np.linspace(1,4, num=4)

        for Selection in index:
            #Chosing Selection Constants
            if Selection == 1: #For Lunar Uplink
                Text = 250 #k
                Tphys = 400 #k
                CableLoss = -.25 #dB
                AtmAtten = 0
                Freq = 3e10
                TransmitPower = 100
                Diameter_Tx = Diameter_TxM
                

            if Selection == 2: #For Lunar Downlink
                Text = 25
                Tphys = 400 #4
                CableLoss = -.5 #dB
                AtmAtten = 0
                Freq = 3e10
                TransmitPower = 100
                Diameter_Tx = DiameterTxO

            if Selection == 3: #For Crosslink
                Text = 25
                Tphys = 400 #k
                CableLoss = -.5 #dB
                AtmAtten = 0
                Freq = 3e10
                TransmitPower = 100
                Diameter_Tx = DiameterTxO
                Range = Range_Sidelink

            if Selection == 4: #For Earth Downlink
                Text = 300
                Tphys = 286.15 #k
                CableLoss = -.5 #dB
                AtmAtten = -7 #dB
                TransmitPower = 100
                Diameter_Tx = DiameterTxO
                Freq = 1.8e10
                Range = 3.84e8 - Range
                DataRate = DataRateED
                DiameterRx = 10 #Only link where reciever does not = transmitter

            wavelength = 3e8/Freq #convert freq to wavelength
            #Noise Constants
            Line_Loss = -.5 #dB
            #Antenna Constants
            Tref = 290 #k
            Tr = 289 #k - Noise Temp

            ################# DATA CALCULATIONS #################
            
            CNo_Min = EbNo + 10*log10(DataRate) + DesignMargin #Required CNoReq = Margin + EbnO + Datarate (in db)
            ################# NOISE CALCULATIONS #################
            Tant = Efficiency*Text + (1-Efficiency)*Tphys #Antenna temp in kelvin
            NF = 1 - (Tphys/(Tr)) #Noise Figure
            Ts = Tr + Tant + Text #Reciever System Noise Temp
            No = (10*log10(boltzman*Ts)).real #Reciever System Noise Power

            ################ NOISE PARAMETERS ################
            
            pointingLoss = -3 #dB
            spaceLoss = self.SpaceLoss(Range, wavelength)

            ################# RECEIVER PARAMETERS #################
            #Tx = Transmitter Rx = receiver
            gainTx, areaTx, areaEffRx, beamWidthRx = self.AntennaParameters(Diameter_Tx, wavelength)
            gainRx, areaRx, areaEffRx, beamWidthRx = self.AntennaParameters(DiameterRx, wavelength)

            powerTxDb = (10*log10(TransmitPower)).real
            
            EIRP = (powerTxDb + gainTx + Line_Loss).real
            
            ################## FINAL PARAMETERS ##################

            PropLosses = spaceLoss + AtmAtten 
            # print(PropLosses)
            # print(pointingLoss)
            # print(gainTx)

            ReceivedPower = EIRP + pointingLoss + PropLosses + gainRx
            CNo_received = ReceivedPower - No
            #print(CNo_Min.real)

            marg = (CNo_received - CNo_Min).real #calculating margin
            margin.append(marg)
            #print(marg)

            margin_checker = 0 #Default is 0, margin check of 1 means the link works
            
            if marg >= 3 and Range_Sidelink != 0: #assuming 3dB error correction is added
                margin_checker = 1 #do we have power and margin for the link to exist
            
            #Setting Weighted Average
            if Selection == 1:
                DRS = DataRate *  .35  * margin_checker #this is the data amount that can be transmitted per period
                #^^ sets amnt to 0 if not feasible
                
            if Selection == 2:
                DRS = DataRate *  .35 * margin_checker #this is the data amount that can be transmitted per period
                    
            if Selection == 3:
                DRS = DataRate * .20 * margin_checker #this is the data amount that can be transmitted per period
                
            if Selection == 4: #earth downlink is the least important rate
                DRS = DataRateED *  .10 * margin_checker #this is the data amount that can be transmitted per period
                
            DRS_list.append(DRS)

            if amnt < 600e6 or margin_checker != 1: #if requirement is not satisfied seat feasibility to 0
                constraints = 0

        DataRateReturn = sum(DRS_list) #return weighted average of dataRates
        currDesign.add_commsObj(coverage, DataRateReturn)
        print("FOM")
        print(coverage)
        print("Data rate")
        print(DataRateReturn)
        print("data amount")
        print(data_amount)
        return constraints, data_amount



