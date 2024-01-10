# FUllPowerModelFunction - the genuine FULL MODEL
# Created: 1/12/2023
# Author: Cameron King
# Finalized: Cameron King; 2/2/2023
import GS_Stuff


# description: This model takes in the orbits and numsats variables from MOO, as well as other individual design variables, to output
# the total energy received for this design. It chooses the two groundstation locations based on the number of sc on the orbit, and their distances at approaches


def FullPowerModelFunction(orbits, numSats,   panelSize,   LI_battery_mass_total,   laser_intake_wattage,  r_aperture,  r):

    # imports
    import numpy
    import math
    import PowerMain_Func3D
    import Plotting

    # simple magnitude function, used throughout
    def MagFunc(vec):
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

    # constants for adjusting from lunar to metric
    r_m = 1737400  # radius of the moon, m
    LU = 389703000 # distance of Earth to Moon in m
    T_m = 2360448 # lunar day in seconds

    # loop through all orbits extracting their data
    NumOrbits = len(orbits)
    PosnTime = []
    eclipse_percent = []
    Period = []

    for i in range(0,NumOrbits):
        orbit_curr = orbits[i]
        x = orbit_curr.x
        y = orbit_curr.y
        z = orbit_curr.z
        vx = orbit_curr.vx
        vy = orbit_curr.vy
        vz = orbit_curr.vz
        eclipse_percent.append(orbit_curr.eclipse)
        Period.append(orbit_curr.T*T_m) # period in seconds
            
        t = numpy.linspace(0,Period[i],len(x)) # creates time vector, seconds
        PosnTime.append([LU*x,LU*y,LU*z,t]) # convert lunar units to seconds and meters




    ##### Ground Station Locating #####
    # From PosnTime, list of each orbit's [x,y,z,t] for this design, & NumOrbits determine which 2 orbits
    # are the ones transmitting to ground stations. Need to find transmitting orbit indices, ground station coordinates,
    # and pass over index

    current_GS_sich = GS_Stuff.GroundStationLocator(PosnTime,numSats)
    orbit_range = current_GS_sich[0] # list of the 1 or 2 chosen orbit's indices in PosnTime/numSats
    GS_index = current_GS_sich[1] # list of indices of gs cross over for the 1 or 2 chosen orbits

    comm_orbit_range = []
    for i in range(0,NumOrbits-1):
        if len(orbit_range) != 1:
            for j in range(0,len(orbit_range)-1):
                if i != orbit_range[j]:
                    comm_orbit_range.append(i)
        else:
            if i != orbit_range:
                comm_orbit_range.append(i)

    PownComm = [orbit_range,comm_orbit_range]

    ###### Energy Calculation for each of the two chosen orbits #####

    E_rec_perT = []
    E_rec_perD = []
    Eff = []
    numSats_transmitting = []
    alpha_ave = []
    periods = []
    d_ave = []
    theta_r_transmission = []
    pos_rec = []
    t_ends = []
    latlon = []
    TransPosnTime = []
    efficiency_profs = []
    theta_r_profs = []
    d_profs = []
    t_profs = []

    for i in range(0,len(orbit_range)):

        # extract this orbit's position and time, groundstation cross over index, period, and eclipse percent
        thisPosnTime = PosnTime[orbit_range[i]]
        thisGS_index = GS_index[i]

        if thisGS_index > len(thisPosnTime[0]): # rare case where GS_locator goes around the orbit too far, this just subtracts a total number of indices
            thisGS_index = thisGS_index - len(thisPosnTime[0])

        thisPeriod = Period[orbit_range[i]]
        thiseclipse_percent = eclipse_percent[orbit_range[i]]
        # quick receiver position calculation for this orbit
        d_t = MagFunc([thisPosnTime[0][thisGS_index]-LU, thisPosnTime[1][thisGS_index],thisPosnTime[2][thisGS_index]])  # distance of transmission
        u_t = [(thisPosnTime[0][thisGS_index]-LU)/d_t, thisPosnTime[1][thisGS_index]/d_t, thisPosnTime[2][thisGS_index]/d_t]  # unit vector of transmission
        thispos_rec = [u_t[0]*r_m + LU, u_t[1]*r_m, u_t[2]*r_m]  # position of receiver, radius of moon * unit vector of transmission point
        pos_rec.append(thispos_rec) # store this for later

        # unpacking of Current_Power_sich:

        if numSats[i] != 0: # only append if the current orbit actually has sc on it

            Current_Power_sich = PowerMain_Func3D.PowerMain_func3D(thisPosnTime, thispos_rec, thisGS_index, thisPeriod, thiseclipse_percent, panelSize, LI_battery_mass_total, laser_intake_wattage, r_aperture, r)

            E_rec_perT.append(round(Current_Power_sich[1], 3))
            E_rec_perD.append(round(Current_Power_sich[1]*numSats[i]/(thisPeriod/(3600*24)),8)) # scaled by current period and num of sc
            Eff.append(Current_Power_sich[2])
            numSats_transmitting.append(numSats[i])
            alpha_ave.append(Current_Power_sich[4])
            periods.append(round(thisPeriod/(3600*24),2))
            d_ave.append(round(Current_Power_sich[5],3))
            theta_r_transmission.append(round(Current_Power_sich[6],3))
            t_ends.append(round(Current_Power_sich[0],3))

            TransPosnTime.append(Current_Power_sich[7])
            efficiency_profs.append(Current_Power_sich[8])
            theta_r_profs.append(Current_Power_sich[9])
            d_profs.append(Current_Power_sich[10])
            t_profs.append(Current_Power_sich[11])

            # convert pos_rec to lat and lon
            thislat = round(math.asin(thispos_rec[2]/r_m)*180/numpy.pi,3)
            thislon = round(math.atan2(thispos_rec[1],thispos_rec[0]-LU)*180/numpy.pi-180,3)
            latlon.append([thislat,thislon])

    # energy to each gs

    E_rec_perD_totdesign = sum(E_rec_perD)



    ##### PRINTING ######

    print('Transmission Point & Orbit Parameters:')
    print('   View Angles: ',alpha_ave,'radians')
    print('   Distances: ',d_ave,'km')
    print('   Transmission times: ',t_ends,'seconds')
    print('   Max incident angle: ',theta_r_transmission,'degrees')
    print('   Number of SC: ',numSats_transmitting)
    print('   Number of SC on Comms: ',len(PownComm[1]))
    print('   Periods:',periods,'hours')

    print('Results:')
    print('   Ground Station Locations: ',latlon,'degrees lat,lon')
    print('   Energy Received, 24 hour average:',E_rec_perD,'kWh/24h')
    print('   Energy Received, single transmission:',E_rec_perT,'kWh')

    ##### PLOTTING #####
    #print(efficiency_profs)
    #Plotting.PlotThisDesign(PosnTime,TransPosnTime,PownComm,pos_rec,efficiency_profs,theta_r_profs,d_profs,t_profs)

    return E_rec_perD_totdesign # return total kWh/24hr delivered to the surface