# PowerMain_Func3D - The Power TRANSMISSION model
# Created: 10/06/2022
# Author: Cameron King
# Finalized: Cameron King; 2/2/2023
# Description:
#
# This is the main function for modeling the power TRANSMISSION. Given coordinates of a full orbit period, the coordinates of the groundstation,
# and the remaining power design variables, the power received for ONE transmission of ONE sc on the given orbit is calculated. Other values
# necessary to output for hardware design and design verification are also calculated and output.


def PowerMain_func3D(FULLpos_sc, pos_rec,trans_center_index, Period, eclipse_percent,   panelSize,   LI_battery_mass_total,   laser_intake_wattage, r_aperture,    r  ):

    # imports
    import OrbitAssumptions
    import GaussNCone
    import Current_Orbit_Values3D
    import efficiency_funcs
    import math
    import mpmath
    import numpy


    # small magnitude function, used throughout
    def MagFunc(vec):
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

    #####################################################################################
    #######################   Constants & Assumed Values   ##############################
    #####################################################################################

    # moon
    mu_m = 4.905E12; # Gravitational parameter of the moon, 
    r_m = 1737400; # radius of the moon, m

    # attitude & position errors - hardcoded to 0 because we assume no error
    pos_err = [0, 0, 0]
    point_err = [0, 0]

    # battery & pane constants
    satLife = float(10); # years
    degPYear = float(0.01); # 1% 
    thetaMax = float(0); # informs peak power production
    I_d = float(0.77); # inherent degradation (0.49-0.88)----------SMAD
    BOLEff = float(0.3); #Beginning of Life Efficiency, 30 % ----- https://www.nasa.gov/smallsat-institute/sst-soa/power 
    BOLPmp = float(400); # W/m^2 ----------------------------------https://www.nasa.gov/smallsat-institute/sst-soa/power 
    specPow = float(100); # W/kg ----------------------------------https://www.nasa.gov/smallsat-institute/sst-soa/power
    DoD = 0.4; # Depth pf Discharge
    LI_battery_upperBound = 0.15; # Battery can't allocate this capacity to anything else
    LI_battery_lowerBound = 0.15; # Battery can't allocate this capacity to anything else
    bounds = LI_battery_upperBound + LI_battery_lowerBound # total sum of the bounds
    SatSurvival = 0.05; # Battery dedicated to onboard computing
    LI_EOL_const = 0.85;  #0.85 is from EOL modeling
    Panel_Eff = 0.32 # solar panel efficiency within the assumed range of +/- 22.5 Degrees
    theta_panel = float(0.4); # Influences cosine loss 22.5 deg worst case -> 0.4 rad

    # MAIN BAT SPECS:
    P_per_kg = 1500 # W/kg
    E_per_kg = 200 # Wh/kg 

    # Comms
    Comm_Power = 100 # watts of constant power draw for comms system
    
    # laser
    laser_loss = 0.55 # percentage of loss of power in the laser itself
    
    # receiver
    rec_zleff = 0.30 # receiver's zero loss efficiency (normal / maximum efficiency of flux power to elec power) 
    rec_b_0 = 0.1 # reflectivity constant, 0.1 for 1 sheet of glass, 0.2 for 2
    rec_I_cutoff = 1380*450 # W/m^2 Max flux receiver can withstand, any higher flux and this is the power accepted. This caps the flux allowed.
    
    #####################################################################################
    #####################################################################################
    #####################################################################################

    Feasible = 1;

    ###### Calculate power generated by the solar panels ######

    L_d = (1-degPYear)**satLife; # % (How much the satellite degrades over a lifetime)

    P_eol = Panel_Eff* BOLPmp * L_d * math.cos(theta_panel); # Specific power at end of life
    P_0 = P_eol*panelSize; # power available at end of life, assume this is the power available during the whole lifetime

    P_0_noComm = P_0 - Comm_Power

    ###### Battery losses and allocation ######

    LI_usablebattery_mass = LI_battery_mass_total * 0.5; # Redundancy: Makes sure that there is a secondary battery system if the first fails for some reason

    LI_battery_capacity_total = LI_usablebattery_mass*E_per_kg* LI_EOL_const *(1-SatSurvival-bounds); # [Wh] # same assumption of end of life power output of panels; battery at end of life has LI_EOL_const amount of initial value,
    LI_battery_discharge = LI_usablebattery_mass*P_per_kg * LI_EOL_const;     # [W]  # so entire lifetime we assume we are operating at end of life conditions

    LI_battery_capacity_laser = DoD*LI_battery_capacity_total # energy capacity for the laser
    
    ###### feasibility checks

    if LI_battery_discharge < laser_intake_wattage: 
        Feasible = 0; # not enough mass of batt to power high wattage laser

    #Satellite can charge panels in half an orbital period
    E2Batt = P_0_noComm * Period * (1-eclipse_percent); # [Wh] Assume battery charges for half of the orbit period
    if E2Batt < LI_battery_capacity_total:
        Feasible = 0;
        print('laser')


    ###### calculate max transmission time for current battery specs & receiver specs ######

    # time step accuracy
    N_full = len(FULLpos_sc[0])

    # laser loss and maximum discharge time
    L_W = laser_intake_wattage*(1-laser_loss) # Laser Wattage, this is the battery/capaciter AVERAGE watt output possible, minus the power loss of the laser
    t_max_battery = LI_battery_capacity_laser/laser_intake_wattage*3600; # max discharge time, equal to maximum transmission time for this battery

    # receiver maximum trasnmission time given receiver reflectance, this takes time of total view into account

    theta_r_max = mpmath.acos(rec_b_0/(1+rec_b_0))

    # Preallocation
    pos_full_orbit = []
    t_full_orbit = []
    z_list = []
    t_max_receiver = 0
    theta_tmp = []

    for i_full in range(0,N_full):

        pos_full_orbit.append([FULLpos_sc[0][i_full],FULLpos_sc[1][i_full],FULLpos_sc[2][i_full]]) # x y z position matrix
        z_list.append(FULLpos_sc[2][i_full]) # z array
        t_full_orbit.append(FULLpos_sc[3][i_full]) # t array
        current_sich = Current_Orbit_Values3D.Current_Orbit_Values3D(pos_full_orbit[i_full], pos_rec, r) # [d,theta_s,theta_r,FOV,r_prime,h], matrix

        this_theta_r = current_sich[2]
        theta_tmp.append(current_sich[2])

        if i_full != 0:
            this_t_step = t_full_orbit[1] - t_full_orbit[0]
            if this_theta_r < theta_r_max:
                # we are within transmission, just count that this time step was part of it
                t_max_receiver = t_max_receiver + this_t_step

    if t_max_receiver != 0:
        t_end = min([t_max_battery,t_max_receiver]) # choose the smallest maximum time possible with given orbit & battery
    else:
        t_end = t_max_battery

    ###### do one pass simulation, calculate orbit averages and focal length to define beam conditions ######
        
    # loop through time period to figure out average distance and average size of the receiver
    # do this JUST FOR TRANSMISSION PERIOD CENTERED ABOVE RECEIVER
    # find center time point, and then start and stop indices
    # loads and defines current sich:

    ##### Here is where we need to generate a new position matrix, with just the transmission period
    # the time should start at 0 and go to about t_end, might be a little bit off

    elements_beforenafter = round((t_end/2)/this_t_step)

    TransPosnTimeVec = []
    transmission_time_prof = []
    if elements_beforenafter < 1: # this is when the time of transmission is short compared to the time step from given positions

        TransPosnTimeVec.append([FULLpos_sc[0][trans_center_index], FULLpos_sc[1][trans_center_index], FULLpos_sc[2][trans_center_index],0])
        ref_pos = [FULLpos_sc[0][trans_center_index+1],FULLpos_sc[1][trans_center_index+1],FULLpos_sc[2][trans_center_index+1],this_t_step]
        TransPosnTimeVec.append(ref_pos) # now we have a 2 element long posntime mat

    else: # we have more than 1 point during transmission time
        count = 0
        for i in range(trans_center_index-elements_beforenafter,trans_center_index+elements_beforenafter):

            TransPosnTimeVec.append([FULLpos_sc[0][i],FULLpos_sc[1][i],FULLpos_sc[2][i],count*this_t_step])
            count = count+1

    if len(TransPosnTimeVec) <= 2: # this is the case where we dont have enough points in the current vec, need more
        TransPosnTimeVec = OrbitAssumptions.StraitLineOrbitApprox(TransPosnTimeVec,t_end) # this adds elements between, and returns a new matrix

    N = len(TransPosnTimeVec)
    t_end = TransPosnTimeVec[-1][3] # reset t_end to the slightly different t_end from above
    t_step = TransPosnTimeVec[1][3]-TransPosnTimeVec[0][3]
    # preallocate
    d = []
    theta_s = []
    theta_r = numpy.zeros(N)
    t = numpy.zeros(N)
    FOV = numpy.zeros(N)
    r_prime = numpy.zeros(N)
    dtheta_s = numpy.zeros(N)
    h = numpy.zeros(N)
    dtheta_s_approx = numpy.zeros(N)
    ddtheta_s_approx = numpy.zeros(N)
    
    # for the below loop, i is the index for the transmission period, ranging from 0 to length of the just the transmission period.
    # i_full is different. this is the index that corresponds to i in FULLpos_sc

    for i in range(0,N):

        current_pos = [TransPosnTimeVec[i][0],TransPosnTimeVec[i][1],TransPosnTimeVec[i][2]]
        current_sich = Current_Orbit_Values3D.Current_Orbit_Values3D(current_pos, pos_rec, r)
        transmission_time_prof.append(TransPosnTimeVec[i][3])
        # split up the output from Current_Orbit_Values into useful values to save, defined by i

        if i > 0:
            t[i] = t[i-1]+t_step

        # print("d: ", current_sich[0])
        d.append(current_sich[0])
        #theta_s.append(current_sich[1])
        theta_r[i] = current_sich[2]
        # print("theta_r: ", theta_r[i])
        FOV[i] = current_sich[3]
        r_prime[i] = current_sich[4]
        h[i] = current_sich[5]


    # calculate values to define beam shape, and in turn HARDWARE NECESSARY VALUES
    d_ave_index = -1
    # distance average, with min and max and std
    d_ave = numpy.mean(d)

    for i in range(0,len(d)):
        if d[i] < d_ave:

            d_ave_index = i
            break

    d_std = numpy.std(d)

    # Receiver angle at average distance

    theta_max_transmission = (max(abs(theta_r)))*180/numpy.pi

    # apparent radius at transmission time
    r_b = numpy.mean(d_ave_index)

    alpha_ave = mpmath.atan((r_b-r_aperture)/d_ave) # ideal view angle, this is the defining angle of the shape of the beam
    focal_length = -r_aperture/mpmath.tan(alpha_ave) # focal length of necessary diverging lens, neg cause diverging

    ###### using defined beam conditions, simulate the beam at every point on reciever during transmission ######

    # preallocations
    F_disp = [];
    P_T = [];
    UA_F_disp = [];
    I_ave = numpy.zeros([N-1,100]);
    I_max = numpy.zeros(N-1)
    
    # loop through the transmission period 
    # this section also accounts for the max intensity, but also measures the UA = UNADJUSTED flux dispursion to determin soley position error down the line
    for i in range(0,N-1):
        current_disp = GaussNCone.gaussNcone_transmission_func(r_aperture,r_b,d_ave,d[i],L_W)
        current_disp = numpy.array(current_disp)

        # check if intensity of shell is above the maximum, adjust the percentage within to keep I_ave[shell] < rec_I_cutoff
        for j in range(1,len(current_disp[1,:])):
            P_perc_old = current_disp[1,j]
            A_shell = (numpy.pi*(current_disp[0,j]**2-current_disp[0,j-1]**2)) # area of the shell rn

            I_ave[i,j] = (current_disp[1,0]*current_disp[1,j])/A_shell # (total power * percent of power within shell) / area of shell
    
            if I_ave[i,j] >= rec_I_cutoff: # we need to reassign the second row of current_disp to rec_I_cutoff = P_within / A_shell
                P_allowed = rec_I_cutoff*A_shell
                P_perc_new = P_allowed/current_disp[1,0]
                current_disp[1,j] = P_perc_new
    
        I_max[i] = max(I_ave[i,:])  
        F_disp.append(current_disp) 
        P_T.append(current_disp[1,0])

    F_disp = numpy.array(F_disp) # flux dispursion for each time step in a matrix
    UA_F_disp = numpy.array(UA_F_disp) # this is the incident flux, unadjusted (UA) for the receiver's max intensity
    
    
    ###### Using the flux dispursion at every time step, determine efficiency at every step, and in total ######
    
    # preallocations
    n_rec = numpy.zeros(N);
    n_pos = numpy.zeros(N);
    UA_n_pos = numpy.zeros(N);
    E_R = numpy.zeros(N);
    E_T = numpy.zeros(N);

    eff_prof = []
    
    for i in range(0,N-1):

        # this function calculates the efficiency associated with incidence angle and receiver efficiency
        n_rec[i] = efficiency_funcs.receiver_eff_func(theta_r[i], rec_zleff, rec_b_0);
        
    
        # this is the position error, without taking max intensity into account -> useful for checking position error effects
        UA_F_disp = GaussNCone.gaussNcone_transmission_func(r_aperture,r_b,d_ave,d[i],L_W)
        UA_n_pos[i] = efficiency_funcs.position_eff_func(theta_r[i], pos_err, point_err, UA_F_disp, h[i], r);
        
        # this function will determine the efficiency associated with the pointing and position error of the satellite
        # This also incorperates lost energy from changing apperent reciever size
        n_pos[i] = efficiency_funcs.position_eff_func(theta_r[i], pos_err, point_err, F_disp[i,:,:], h[i], r);

        E_R[i] = t_step*P_T[i]*n_rec[i]*n_pos[i] # total energy recieved, per time step
        E_T[i] = t_step*P_T[i] # Total energy transmit, per time step

        eff_prof.append(E_R[i]/E_T[i]) # efficiency profile
    

    E_R_tot = sum(E_R); # This is in Joules ->
    E_R_tot = E_R_tot*2.7778*10**-7 # kWh

    E_T_tot = sum(E_T); # This is in Joules ->
    E_T_tot = E_T_tot*2.7778*10**-7 # kWh

    Total_eff = E_R_tot/E_T_tot*100

    return [t_end, E_R_tot, Total_eff, Feasible, alpha_ave, d_ave, theta_max_transmission, TransPosnTimeVec, eff_prof, theta_r, d, transmission_time_prof]
