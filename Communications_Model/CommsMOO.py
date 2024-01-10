#Importing Math functions to speed up code


#Importing Functions



def CommsMOO(currDesign):
    from cmath import log10
    from math import pi
    from SpaceLoss import SpaceLoss
    from AntennaParameters import AntennaParameters
    import numpy as np
    #Inputs: Diameter_TxM = antenna diameter of the reciever on the moon in meters
    #DiameterTxO = antenna diameter of the reciever on the sattelite in meters
    #Freq = frequency in Hz
    #DataRate = desired data rate in bps
    #DataRateED = desired data rate for earth downlink (Keep this above 10e6)
    #Range = distance between Tx & Rx in m
    #Range_Sidelink = distance between sattelites in m
    #Time LOS = #of seconds over a target per period
    distMoon = [1, 0, 0] # moon pos in LU    orbit = currDesign.orbit
    orbit = currDesign.orbit
    posX = orbit.x
    posY = orbit.y
    posZ = orbit.z
    alt = np.sqrt( (posX - distMoon[0])**2 + (posY - distMoon[1])**2 + (posZ - distMoon[2])**2) # distance between satellite and moon center
    alt = alt*389703 #convert from LU to m 

    Range = alt
    Range_Sidelink = 1#calc goes here!!!!!!!!

    Diameter_TxM = currDesign.diameterTxM
    DiameterTxO = currDesign.diameterTxO
    DataRate = currDesign.dataRate
    DataRateED = currDesign.dataRate_ED
    


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
            Range = 3.84e8 - Range #avg dist earth surface ---> lunar surface - altitude 
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
        spaceLoss = SpaceLoss(Range, wavelength)

        ################# RECEIVER PARAMETERS #################
        #Tx = Transmitter Rx = receiver
        gainTx, areaTx, areaEffRx, beamWidthRx = AntennaParameters(Diameter_Tx, wavelength)
        gainRx, areaRx, areaEffRx, beamWidthRx = AntennaParameters(DiameterRx, wavelength)

        powerTxDb = (10*log10(TransmitPower)).real
        
        EIRP = (powerTxDb + gainTx + Line_Loss).real
        
        ################## FINAL PARAMETERS ##################

        PropLosses = spaceLoss + AtmAtten 
        # print(PropLosses)
        # print(pointingLoss)
        # print(gainTx)

        ReceivedPower = EIRP + pointingLoss + PropLosses + gainRx
        CNo_received = ReceivedPower - No
        print(CNo_Min)

        marg = (CNo_received - CNo_Min).real #calculating margin
        margin.append(marg)
        margin_checker = 0 #Default is 0, margin check of 1 means the link works

        
        
        if marg >= 3 and Range_Sidelink != 0: #assuming 3dB error correction is added
            margin_checker = 1 #do we have power and margin for the link to exist
        if data_amount < 600e6: #if requirement is not satisfied seat feasibility to 0
            margin_checker = 0
        margin_check.append(margin_checker)

        #Setting Weighted Average
        if Selection == 1:
            amnt = DataRate *  .35  * margin_checker #this is the data amount that can be transmitted per period
            #^^ sets amnt to 0 if not feasible
            data_amount.append(amnt.real)

        if Selection == 2:
            amnt = DataRate *  .35 * margin_checker #this is the data amount that can be transmitted per period
            data_amount.append(amnt.real)     

        if Selection == 3:
            amnt = DataRate * .20 * margin_checker #this is the data amount that can be transmitted per period
            data_amount.append(amnt.real)    

        if Selection == 4: #earth downlink is the least important rate
            amnt = DataRate *  .10 * margin_checker #this is the data amount that can be transmitted per period
            data_amount.append(amnt.real)  

        DataRateReturn = sum(data_amount) #return weighted average of dataRates


    return(margin_check,DataRateReturn)