import numpy
import math
import mpmath


# Move path to main CRATER directory to import design and orbit classes
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

from collections import Counter
from os import listdir
from os.path import isfile, join


class power:

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # New Shit
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    def position_eff_func(self, theta, pos_err, point_err, F_disp, h, r):
        # This function determines the efficiency associated with the error in pointing and position
        # The efficiency calculated is specifically the energy hitting the receiver / Energy leaving the laser at the given time

        # imports

        # define errors in each dimension, these are input
        xpos_err = pos_err[0];
        ypos_err = pos_err[1];
        hpos_err = pos_err[2];
        theta_err = point_err[0];
        phi_err = point_err[1];

        # Define/calculate other necessary values, taking errors in
        x = h * mpmath.tan(theta);
        d = math.sqrt((x + xpos_err) ** 2 + (h + hpos_err) ** 2)
        FOV = 2 * (mpmath.atan((x + xpos_err + r) / (h + hpos_err)) - mpmath.atan((x + xpos_err) / (h + hpos_err)))
        r_prime = d * mpmath.atan(FOV / 2)

        # find total errors
        xtheta_err = h * mpmath.tan(theta + theta_err) - x;
        x_err = xpos_err + xtheta_err;
        yphi_err = d * mpmath.tan(phi_err);
        y_err = ypos_err + yphi_err;

        # define shell radii, & their respective flux %
        shell_num = len(F_disp[0]) - 1;
        r_shell = F_disp[0];
        Fperc_shell = F_disp[1];

        # Preallocation
        A_hit = numpy.zeros(shell_num);
        A_avail = numpy.zeros(shell_num);
        hit_eff = numpy.zeros(shell_num);
        shell_eff = numpy.zeros(shell_num);

        # Loop through each shell, determine the area available to hit and the area actually hit
        for shell_index in range(0, shell_num):

            # define this shells inner and outer radii
            if shell_index == 0:
                r_outer = r_shell[shell_index + 1];
                r_inner = 0;
            else:
                r_outer = r_shell[shell_index + 1];
                r_inner = r_shell[shell_index];

            dx = r_outer / 10  # define the distance and area steps
            dA = dx ** 2

            # loop through the total range in x dimension of the shell
            for x in numpy.arange(-r_outer, r_outer, dx):

                y_outer = math.sqrt(r_outer ** 2 - x ** 2);  # y of edges of shell at current x

                # loop through the max and min y at current x, not including within the shell when x < r_inner
                if abs(x) > abs(r_inner):
                    # if we are not needing to 'jump' over the center of the shell, start and end of scan in x

                    for y in numpy.arange(-y_outer, y_outer, dx):

                        # check if point is on the receiver 1, add to area hit if it is, add to area available to hit either way
                        hit_value = ((x - x_err) ** 2) / (r_prime ** 2) + ((y - y_err) ** 2) / (r ** 2)
                        if hit_value <= 1:
                            A_hit[shell_index] = A_hit[shell_index] + dA
                            A_avail[shell_index] = A_avail[shell_index] + dA
                        else:
                            A_avail[shell_index] = A_avail[shell_index] + dA

                else:
                    # if we get here, we are in the range where we have to scan the top and bottom halves of the shell in y

                    y_inner = math.sqrt(r_inner ** 2 - x ** 2)  # this is the inner bound on the y range

                    for y in numpy.arange(-y_outer, -y_inner, dx):  # for the bottom half of the shell...

                        # check if point is on the receiver 1, add to area hit if it is, add to area available to hit either way
                        hit_value = ((x - x_err) ** 2) / (r_prime ** 2) + ((y - y_err) ** 2) / (r ** 2)
                        if hit_value <= 1:
                            A_hit[shell_index] = A_hit[shell_index] + dA
                            A_avail[shell_index] = A_avail[shell_index] + dA
                        else:
                            A_avail[shell_index] = A_avail[shell_index] + dA

                    for y in numpy.arange(y_inner, y_outer, dx):  # for the top half of the shell...

                        # check if point is on the receiver 1, add to area hit if it is, add to area available to hit either way
                        hit_value = ((x - x_err) ** 2) / (r_prime ** 2) + ((y - y_err) ** 2) / (r ** 2)
                        if hit_value <= 1:
                            A_hit[shell_index] = A_hit[shell_index] + dA
                            A_avail[shell_index] = A_avail[shell_index] + dA
                        else:
                            A_avail[shell_index] = A_avail[shell_index] + dA

            # now, per shell, find out efficiency
            if A_avail[
                shell_index] == 0:  # This only happens when the area available to hit is so small the step size rounds it to zero

                # check if center of beam is on the ellipse or not, 100% efficiency if it is and 0% efficiency if it isnt
                if ((0 - x_err) ** 2) / (r_prime ** 2) + ((0 - y_err) ** 2) / (r ** 2) <= 1:
                    hit_eff[shell_index] = 1
                else:
                    hit_eff[shell_index] = 0

            else:  # in all other cases, with out A_avail = 0, below is solved to be the efficiency of position per shell
                hit_eff[shell_index] = A_hit[shell_index] / A_avail[
                    shell_index]  # if this is close to 1, then the shell is 100% on the receiver, even with error

            # now, adjust the percent of total flux contained in a shell by the hit efficiency of that shell
            # the sum of this is going to be the total position efficiency, a sum which would have been 100% with complete position efficiency
            shell_eff[shell_index] = hit_eff[shell_index] * Fperc_shell[shell_index + 1]

        return sum(shell_eff)

    def receiver_eff_func(self, theta, zero_loss_eff, b_0):
        # This function determines the efficiency of a receiver as a function of incident angle

        # this is the cutoff of usable power as a function of reflectance
        theta_cutoff = mpmath.acos(b_0 / (1 + b_0))

        if theta > theta_cutoff:  # if the current angle is above cutoff, set efficiency to 0
            n_rec = 0
        else:  # if we are within angle range...
            K_theta = 1 - b_0 * ((1 / mpmath.cos(theta)) - 1)  # incident angle modifier, scales zero loss efficiency
            n_rec = zero_loss_eff * K_theta  # receiver associated efficiency when in angle range

        if n_rec < 0:  # catch just in case
            n_rec = 0

        return n_rec  # output the receiver efficiency

    def Current_Orbit_Values3D(self, pos_sc, pos_rec, r):
        # imports

        # quick magnitude function used throughout
        # quick magnitude function used throughout
        def MagFunc(vec):
            return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

        # Constants
        r_m = 1737400;  # radius of the moon, m

        # current angle between receiver, SC, and center of moon
        alpha = mpmath.acos(numpy.dot(pos_sc, pos_rec) / (MagFunc(pos_sc) * MagFunc(pos_rec)))

        # current distance vector from SC to Receiver
        d_vec = [pos_sc[0] - pos_rec[0], pos_sc[1] - pos_rec[1], pos_sc[2] - pos_rec[2]]
        d = MagFunc(d_vec)  # distance magnitude

        # calculate receiver incident angle, negetive d vector calculated for it
        d_vec_neg = [-pos_sc[0] + pos_rec[0], -pos_sc[1] + pos_rec[1], -pos_sc[2] + pos_rec[2]]
        theta_r = mpmath.acos(numpy.dot(d_vec_neg, pos_rec) / (MagFunc(d_vec) * MagFunc(pos_rec)))
        # print("numerator: ", numpy.dot(d_vec_neg,pos_rec))
        # print("denominator: ", (MagFunc(d_vec)*MagFunc(pos_rec)))
        theta_r = mpmath.re(theta_r)

        while theta_r >= numpy.pi / 2 or theta_r < -numpy.pi / 2:
            if theta_r >= numpy.pi / 2:
                theta_r = theta_r - numpy.pi
            if theta_r < -numpy.pi / 2:
                theta_r = theta_r + numpy.pi

        # calculate SC pointing angle
        theta_s = theta_r - alpha

        # calculate adjusted radius
        r_prime = r * mpmath.sin(math.pi / 2 - theta_r)

        # calculate field of view
        # print("r_prime: ",r_prime)
        # print("r", r)

        FOV = 2 * (mpmath.atan(r_prime / (d - math.sqrt(r ** 2 - r_prime ** 2))))  # field of view of the receiver

        # calculate altitude
        h = MagFunc(pos_sc) - r_m

        return [d, theta_s, theta_r, FOV, r_prime, h]

    def gaussNcone_transmission_func(self, r_aperture, r_ave, d_ave, d, P0):
        # This function will combine the gaussian distribution output from gaussian_transmission_func, with a conular dispursion.
        # The flux dispursion from gaussian_transmission_func is projected onto a cone, scaling up the radius corresponding to a given
        # shell as a function of distance. The output is the flux dispursion, [r_vec,P_within_vec], accuratly modeled with a cone.

        # imports

        mpmath.dps = 30
        # 99% of the beam is in 1.52*r, where r is the radius of the beam IN THE GAUSSIAN FORMULATION
        # dividing by 1.52 here allows for the calculation of a beam that contains 99% in r_aperature = r*1.52
        r = r_aperture / 1.52

        # hardcoded value for distance from the laser to the output lens. Doesn't matter too much, as it is a cylinder during that time and does not dispurse
        d_lens = 1

        # Get the flux distribution and radii for gaussian beam with radius of the output lens, at the output lens
        # this does the gaussian formulation of the math to determine power in each shell WITHOUT EXPANDING IT ON A CONE
        F_disp = self.gaussian_transmission_func(r, d_lens, d_lens, P0)

        # determine cone shape and radius ratio
        alpha = mpmath.atan((r_ave - r_aperture) / d_ave)  # view angle, half of FOV
        focal_length = r_aperture / abs(mpmath.tan(alpha))  # focal length of necessary lens

        for i in range(0, len(
                F_disp[0])):  # ranges through all radii given from gaussian_transmission_func, scaling up as it goes

            r_new = F_disp[0][i] * (d_ave / focal_length + 1)  # radius of this ring at average distance
            F_disp[0][i] = r_new / d_ave * d  # scale this new radius by current distance

        # CHECK: is the sum of the shell percentages close to 100%? throw error if not
        tot_shell_perc = sum(F_disp[1][:]) - F_disp[1][0]
        if tot_shell_perc < 0.98:
            print('Total shell percentage = ', tot_shell_perc)
            raise ValueError('Total Shell Percentage doesnt sum to >98%. Check gaussian_transmission_func & calls.')

        return F_disp  # return the flux dispursion on a cone in the same format as given from gaussian_transmission_func

    def gaussian_transmission_func(self, radius, av_d, curr_d, P0):

        # This function takes in the input of the radius at aperture, the average distance to the beam, the current distance, and the initial power
        # the goal is to output a matrix with a radius in one column and a power value in another. The power corresponds to the power contained within
        # the current 'shell', or the ring between the current radius and the last radius.
        # It is important to note that this function does not make the beam a cone, just a cylinder.
        import numpy as np
        # constants
        Lambda = 532 * 10 ** (-12)  # meters

        # Waist of the beam is the size of the aperture
        w0 = radius

        # Calculate Rayleigh length needed for the average distance, wavelength, and target radius
        z_R = (w0 ** 2 * np.pi / Lambda)

        # Calculate the actual radius of the beam at the changing distance (near average distance)
        w = w0 * np.sqrt(1 + (curr_d / z_R) ** 2)

        # Choose max radius to calculate intensity to, should be equal to radius of beam at surface & current time
        r_max = w * 1.52

        # preallocation
        N = 10  # number of rings generated
        r_step = r_max / N  # distance between edges of shells
        P_within = []  # power contained within the shell, in PERCENT
        r = np.arange(0, r_max + r_step, r_step)  # range of radii used in generation of P_within
        r_vec = []  # vector of radii corresponding to P_within

        # Loop through range of radii, r
        for i in range(0, len(r)):
            if i == 0:  # for the first row, assign a 0 radii and the total initial power, for extraction later
                P_within.append(P0)
            elif i == 1:  # for the first radii, it is a circle not a shell, therefore we use this simplified equation
                P_within.append((1 - np.exp((-2 * r[i] ** 2) / (w) ** 2)))
            else:  # for all other cases, use a shell calculation which includes a term with the previous radius r[i-1]
                P_within.append(
                    (1 - np.exp((-2 * r[i] ** 2) / (w) ** 2)) - (1 - np.exp((-2 * r[i - 1] ** 2) / (w) ** 2)))

            r_vec.append(r[i])  # assign the r used to the storage vector of r, r_vec

        return [r_vec, P_within]  # a list of lists, the radius vector and percent power contained vector

    def GroundStationLocator(self, PosnTime, numSats):
        import math
        # find transmitting orbit indices and pass over index
        LU = 389703000  # distance of Earth to Moon in m

        ###### find the transmitting orbit indices first:

        if sum(numSats) != max(numSats):  # when all spacecraft are NOT on one orbit

            max_orbit_indices = []
            for i in range(0, len(numSats)):  # find all orbits that contain the max num of sc
                if numSats[i] == max(numSats):
                    max_orbit_indices.append(i)  # store the indices of the times we are at maximum

            if len(max_orbit_indices) == 1:  # if the max is truly the max, we need to find the second most.

                second_most_sc = 0
                for i in range(0, len(numSats)):
                    if i != max_orbit_indices:

                        if numSats[i] > second_most_sc:
                            second_most_sc = numSats[i]
                            second_most_index = i

                transmitting_orbits = [max_orbit_indices[0], second_most_index]

            else:  # this is when there are more than 1 orbit with the max amount of sc, arbitrarily take the first two with the max

                transmitting_orbits = [max_orbit_indices[0], max_orbit_indices[1]]

        else:  # when there is only one orbit with all sc

            transmitting_orbits = [numSats.index(max(numSats))]

        ##### now we need to find the passover indices of the two given orbits

        pass_index = []
        if len(transmitting_orbits) == 2:

            for i in range(0, len(transmitting_orbits) - 1):  # calculate the pass index for each orbit

                thisOrbit_index = transmitting_orbits[i]
                thisPosnTime = PosnTime[thisOrbit_index]
                x_vec = thisPosnTime[0]
                y_vec = thisPosnTime[1]
                z_vec = thisPosnTime[2]

                min_z_indices = []
                for j in range(0, len(z_vec)):  # find all orbits that contain the max num of sc
                    if z_vec[j] == min(z_vec):
                        min_z_indices.append(j)  # store the indices of the times we are at maximum

                pass_index.append(min_z_indices[0])

                min_d_indices = []
                if len(min_z_indices) != 1:  # this is when our min z isnt unique, and we need to find another time to transmit
                    d = []
                    for j in range(0, len(z_vec)):
                        d.append(math.sqrt((x_vec[j] + LU) ** 2 + y_vec[j] ** 2 + z_vec[j] ** 2))

                    for j in range(0, len(d)):  # find all orbits that contain the max num of sc
                        if d[j] == min(d):
                            min_d_indices.append(j)  # store the indices of the times we are at maximum

                    pass_index.append(min_d_indices[0])
        else:

            thisPosnTime = PosnTime[transmitting_orbits[0]]
            x_vec = thisPosnTime[0]
            y_vec = thisPosnTime[1]
            z_vec = thisPosnTime[2]

            min_z_indices = []
            for j in range(0, len(z_vec)):  # find all orbits that contain the max num of sc
                if z_vec[j] == min(z_vec):
                    min_z_indices.append(j)  # store the indices of the times we are at maximum

            pass_index.append(min_z_indices[0])

            min_d_indices = []
            if len(min_z_indices) != 1:  # this is when our min z isnt unique, and we need to find another time to transmit
                d = []
                for j in range(0, len(z_vec)):
                    d.append(math.sqrt((x_vec[j] + LU) ** 2 + y_vec[j] ** 2 + z_vec[j] ** 2))

                for j in range(0, len(d)):  # find all orbits that contain the max num of sc
                    if d[j] == min(d):
                        min_d_indices.append(j)  # store the indices of the times we are at maximum

                pass_index.append(min_d_indices[0])

        if len(transmitting_orbits) != len(
                pass_index):  # the rare case where we dont find a second pass index fsr, so we set it to 0.
            if len(transmitting_orbits) == 1:
                pass_index = 0
            if len(transmitting_orbits) == 2:
                if len(pass_index) == 0:
                    pass_index = [0, 0]
                if len(pass_index) == 1:
                    pass_index.append(0)

        return [transmitting_orbits, pass_index]

    def StraitLineOrbitApprox(self, pos_vec, t_end):
        # This function takes position and velocity vectors, at least 2 points but any length,
        # and output the [x,y,z,t] vector of the transmission period, with AT LEAST 15 ELEMENTS
        import math
        def MagFunc(vec):
            return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

        N_required = 20
        N_initial = len(pos_vec)

        new_pos_vec = []

        if len(pos_vec) > 2:

            # linearize the area between each given points, generate new points between, create new position vector
            t_current = 0
            add_between = round(N_required / N_initial) + 1
            for gap in range(0, N_initial - 1):

                vec_to_next = [pos_vec[gap + 1][0] - pos_vec[gap][0], pos_vec[gap + 1][1] - pos_vec[gap][1],
                               pos_vec[gap + 1][2] - pos_vec[gap][2]]
                d_mag_to_next = MagFunc(vec_to_next)
                unit_to_next = [vec_to_next[0] / d_mag_to_next, vec_to_next[1] / d_mag_to_next,
                                vec_to_next[2] / d_mag_to_next]
                d_to_addition = d_mag_to_next / (add_between + 1)
                this_t_step = pos_vec[gap + 1][3] - pos_vec[gap][3]

                print(d_to_addition)

                new_pos_vec.append([pos_vec[gap][0], pos_vec[gap][1], pos_vec[gap][2], t_current])
                t_current = t_current + this_t_step

                for addition in range(0, add_between - 1):
                    new_pos_vec.append([unit_to_next[0] * d_to_addition * (addition + 1),
                                        unit_to_next[1] * d_to_addition * (addition + 1),
                                        unit_to_next[2] * d_to_addition * (addition + 1), t_current])
                    t_current = t_current + this_t_step

            new_pos_vec.append([pos_vec[gap + 1][0], pos_vec[gap + 1][1], pos_vec[gap + 1][2], t_current])

        else:

            ref_vec = [pos_vec[1][0] - pos_vec[0][0], pos_vec[1][1] - pos_vec[0][1], pos_vec[1][2] - pos_vec[0][2]]
            center_pos_vec = pos_vec[0]
            ref_d = MagFunc(ref_vec)
            ref_unit = [ref_vec[0] / ref_d, ref_vec[1] / ref_d, ref_vec[2] / ref_d]
            V_trans = ref_d / pos_vec[1][3]
            t_step = t_end / 20
            d_between = V_trans * t_step
            count = 0

            this_pos = []
            for i in range(-20, 20):
                this_time = count * t_step
                count = count + 1
                dist_fr_center = i * d_between
                this_x = center_pos_vec[0] + ref_unit[0] * dist_fr_center
                this_y = center_pos_vec[1] + ref_unit[1] * dist_fr_center
                this_z = center_pos_vec[2] + ref_unit[2] * dist_fr_center

                # this_pos  [center_pos_vec[0]+ref_unit[0]*dist_fr_center,center_pos_vec[1]+ref_unit[1]*dist_fr_center,center_pos_vec[2]+ref_unit[2]*dist_fr_center,this_time]
                # print('this_pos[0]:',center_pos_vec[0]+ref_unit[0]*dist_fr_center)
                # print('this_pos:',this_pos)
                new_pos_vec.append([this_x, this_y, this_z, this_time])

        lists = [[val for val in arr] for arr in new_pos_vec]

        return lists

    def PowerMain_func3D(self, FULLpos_sc, pos_rec, trans_center_index, Period, eclipse_percent, panelSize,
                         LI_battery_mass_total, laser_intake_wattage, r_aperture, r):

        # imports
        # import OrbitAssumptions
        # import GaussNCone
        # import Current_Orbit_Values3D
        # import efficiency_funcs
        # import math
        # import mpmath
        # import numpy

        # small magnitude function, used throughout
        def MagFunc(vec):
            return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

        #####################################################################################
        #######################   Constants & Assumed Values   ##############################
        #####################################################################################

        # moon
        mu_m = 4.905E12;  # Gravitational parameter of the moon,
        r_m = 1737400;  # radius of the moon, m

        # attitude & position errors - hardcoded to 0 because we assume no error
        pos_err = [0, 0, 0]
        point_err = [0, 0]

        # battery & pane constants
        satLife = float(10);  # years
        degPYear = float(0.01);  # 1%
        thetaMax = float(0);  # informs peak power production
        I_d = float(0.77);  # inherent degradation (0.49-0.88)----------SMAD
        BOLEff = float(
            0.3);  # Beginning of Life Efficiency, 30 % ----- https://www.nasa.gov/smallsat-institute/sst-soa/power
        BOLPmp = float(
            400);  # W/m^2 ----------------------------------https://www.nasa.gov/smallsat-institute/sst-soa/power
        specPow = float(
            100);  # W/kg ----------------------------------https://www.nasa.gov/smallsat-institute/sst-soa/power
        DoD = 0.4;  # Depth pf Discharge
        LI_battery_upperBound = 0.15;  # Battery can't allocate this capacity to anything else
        LI_battery_lowerBound = 0.15;  # Battery can't allocate this capacity to anything else
        bounds = LI_battery_upperBound + LI_battery_lowerBound  # total sum of the bounds
        SatSurvival = 0.05;  # Battery dedicated to onboard computing
        LI_EOL_const = 0.85;  # 0.85 is from EOL modeling
        Panel_Eff = 0.32  # solar panel efficiency within the assumed range of +/- 22.5 Degrees
        theta_panel = float(0.4);  # Influences cosine loss 22.5 deg worst case -> 0.4 rad

        # MAIN BAT SPECS:
        P_per_kg = 1500  # W/kg
        E_per_kg = 200  # Wh/kg

        # Comms
        Comm_Power = 100  # watts of constant power draw for comms system

        # laser
        laser_loss = 0.55  # percentage of loss of power in the laser itself

        # receiver
        rec_zleff = 0.30  # receiver's zero loss efficiency (normal / maximum efficiency of flux power to elec power)
        rec_b_0 = 0.1  # reflectivity constant, 0.1 for 1 sheet of glass, 0.2 for 2
        rec_I_cutoff = 1380 * 450  # W/m^2 Max flux receiver can withstand, any higher flux and this is the power accepted. This caps the flux allowed.

        #####################################################################################
        #####################################################################################
        #####################################################################################

        Feasible = 1;

        ###### Calculate power generated by the solar panels ######

        L_d = (1 - degPYear) ** satLife;  # % (How much the satellite degrades over a lifetime)

        P_eol = Panel_Eff * BOLPmp * L_d * math.cos(theta_panel);  # Specific power at end of life
        P_0 = P_eol * panelSize;  # power available at end of life, assume this is the power available during the whole lifetime

        P_0_noComm = P_0 - Comm_Power

        ###### Battery losses and allocation ######

        LI_usablebattery_mass = LI_battery_mass_total * 0.5;  # Redundancy: Makes sure that there is a secondary battery system if the first fails for some reason

        LI_battery_capacity_total = LI_usablebattery_mass * E_per_kg * LI_EOL_const * (
                    1 - SatSurvival - bounds);  # [Wh] # same assumption of end of life power output of panels; battery at end of life has LI_EOL_const amount of initial value,
        LI_battery_discharge = LI_usablebattery_mass * P_per_kg * LI_EOL_const;  # [W]  # so entire lifetime we assume we are operating at end of life conditions

        LI_battery_capacity_laser = DoD * LI_battery_capacity_total  # energy capacity for the laser

        ###### feasibility checks

        if LI_battery_discharge < laser_intake_wattage:
            Feasible = 0;  # not enough mass of batt to power high wattage laser

        # Satellite can charge panels in half an orbital period
        E2Batt = P_0_noComm * Period * (
                    1 - eclipse_percent);  # [Wh] Assume battery charges for half of the orbit period
        if E2Batt < LI_battery_capacity_total:
            Feasible = 0;
            print('laser')

        ###### calculate max transmission time for current battery specs & receiver specs ######

        # time step accuracy
        N_full = len(FULLpos_sc[0])

        # laser loss and maximum discharge time
        L_W = laser_intake_wattage * (
                    1 - laser_loss)  # Laser Wattage, this is the battery/capaciter AVERAGE watt output possible, minus the power loss of the laser
        t_max_battery = LI_battery_capacity_laser / laser_intake_wattage * 3600;  # max discharge time, equal to maximum transmission time for this battery

        # receiver maximum trasnmission time given receiver reflectance, this takes time of total view into account

        theta_r_max = mpmath.acos(rec_b_0 / (1 + rec_b_0))

        # Preallocation
        pos_full_orbit = []
        t_full_orbit = []
        z_list = []
        t_max_receiver = 0
        theta_tmp = []

        for i_full in range(0, N_full):

            pos_full_orbit.append(
                [FULLpos_sc[0][i_full], FULLpos_sc[1][i_full], FULLpos_sc[2][i_full]])  # x y z position matrix
            z_list.append(FULLpos_sc[2][i_full])  # z array
            t_full_orbit.append(FULLpos_sc[3][i_full])  # t array
            current_sich = self.Current_Orbit_Values3D(pos_full_orbit[i_full], pos_rec,
                                                                         r)  # [d,theta_s,theta_r,FOV,r_prime,h], matrix

            this_theta_r = current_sich[2]
            theta_tmp.append(current_sich[2])

            if i_full != 0:
                this_t_step = t_full_orbit[1] - t_full_orbit[0]
                if this_theta_r < theta_r_max:
                    # we are within transmission, just count that this time step was part of it
                    t_max_receiver = t_max_receiver + this_t_step

        if t_max_receiver != 0:
            t_end = min(
                [t_max_battery, t_max_receiver])  # choose the smallest maximum time possible with given orbit & battery
        else:
            t_end = t_max_battery

        ###### do one pass simulation, calculate orbit averages and focal length to define beam conditions ######

        # loop through time period to figure out average distance and average size of the receiver
        # do this JUST FOR TRANSMISSION PERIOD CENTERED ABOVE RECEIVER
        # find center time point, and then start and stop indices
        # loads and defines current sich:

        ##### Here is where we need to generate a new position matrix, with just the transmission period
        # the time should start at 0 and go to about t_end, might be a little bit off

        elements_beforenafter = round((t_end / 2) / this_t_step)

        TransPosnTimeVec = []
        transmission_time_prof = []
        if elements_beforenafter < 1:  # this is when the time of transmission is short compared to the time step from given positions

            TransPosnTimeVec.append([FULLpos_sc[0][trans_center_index], FULLpos_sc[1][trans_center_index],
                                     FULLpos_sc[2][trans_center_index], 0])
            ref_pos = [FULLpos_sc[0][trans_center_index + 1], FULLpos_sc[1][trans_center_index + 1],
                       FULLpos_sc[2][trans_center_index + 1], this_t_step]
            TransPosnTimeVec.append(ref_pos)  # now we have a 2 element long posntime mat

        else:  # we have more than 1 point during transmission time
            count = 0
            for i in range(trans_center_index - elements_beforenafter, trans_center_index + elements_beforenafter):
                TransPosnTimeVec.append([FULLpos_sc[0][i], FULLpos_sc[1][i], FULLpos_sc[2][i], count * this_t_step])
                count = count + 1

        if len(TransPosnTimeVec) <= 2:  # this is the case where we dont have enough points in the current vec, need more
            TransPosnTimeVec = self.StraitLineOrbitApprox(TransPosnTimeVec,
                                                                      t_end)  # this adds elements between, and returns a new matrix

        N = len(TransPosnTimeVec)
        t_end = TransPosnTimeVec[-1][3]  # reset t_end to the slightly different t_end from above
        t_step = TransPosnTimeVec[1][3] - TransPosnTimeVec[0][3]
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

        for i in range(0, N):

            current_pos = [TransPosnTimeVec[i][0], TransPosnTimeVec[i][1], TransPosnTimeVec[i][2]]
            current_sich = self.Current_Orbit_Values3D(current_pos, pos_rec, r)
            transmission_time_prof.append(TransPosnTimeVec[i][3])
            # split up the output from Current_Orbit_Values into useful values to save, defined by i

            if i > 0:
                t[i] = t[i - 1] + t_step

            # print("d: ", current_sich[0])
            d.append(current_sich[0])
            # theta_s.append(current_sich[1])
            theta_r[i] = current_sich[2]
            # print("theta_r: ", theta_r[i])
            FOV[i] = current_sich[3]
            r_prime[i] = current_sich[4]
            h[i] = current_sich[5]

        # calculate values to define beam shape, and in turn HARDWARE NECESSARY VALUES
        d_ave_index = -1
        # distance average, with min and max and std
        d_ave = numpy.mean(d)

        for i in range(0, len(d)):
            if d[i] < d_ave:
                d_ave_index = i
                break

        d_std = numpy.std(d)

        # Receiver angle at average distance

        theta_max_transmission = (max(abs(theta_r))) * 180 / numpy.pi

        # apparent radius at transmission time
        r_b = numpy.mean(d_ave_index)

        alpha_ave = mpmath.atan(
            (r_b - r_aperture) / d_ave)  # ideal view angle, this is the defining angle of the shape of the beam
        focal_length = -r_aperture / mpmath.tan(
            alpha_ave)  # focal length of necessary diverging lens, neg cause diverging

        ###### using defined beam conditions, simulate the beam at every point on reciever during transmission ######

        # preallocations
        F_disp = [];
        P_T = [];
        UA_F_disp = [];
        I_ave = numpy.zeros([N - 1, 100]);
        I_max = numpy.zeros(N - 1)

        # loop through the transmission period
        # this section also accounts for the max intensity, but also measures the UA = UNADJUSTED flux dispursion to determin soley position error down the line
        for i in range(0, N - 1):
            current_disp = self.gaussNcone_transmission_func(r_aperture, r_b, d_ave, d[i], L_W)
            current_disp = numpy.array(current_disp)

            # check if intensity of shell is above the maximum, adjust the percentage within to keep I_ave[shell] < rec_I_cutoff
            for j in range(1, len(current_disp[1, :])):
                P_perc_old = current_disp[1, j]
                A_shell = (numpy.pi * (current_disp[0, j] ** 2 - current_disp[0, j - 1] ** 2))  # area of the shell rn

                I_ave[i, j] = (current_disp[1, 0] * current_disp[
                    1, j]) / A_shell  # (total power * percent of power within shell) / area of shell

                if I_ave[
                    i, j] >= rec_I_cutoff:  # we need to reassign the second row of current_disp to rec_I_cutoff = P_within / A_shell
                    P_allowed = rec_I_cutoff * A_shell
                    P_perc_new = P_allowed / current_disp[1, 0]
                    current_disp[1, j] = P_perc_new

            I_max[i] = max(I_ave[i, :])
            F_disp.append(current_disp)
            P_T.append(current_disp[1, 0])

        F_disp = numpy.array(F_disp)  # flux dispursion for each time step in a matrix
        UA_F_disp = numpy.array(
            UA_F_disp)  # this is the incident flux, unadjusted (UA) for the receiver's max intensity

        ###### Using the flux dispursion at every time step, determine efficiency at every step, and in total ######

        # preallocations
        n_rec = numpy.zeros(N);
        n_pos = numpy.zeros(N);
        UA_n_pos = numpy.zeros(N);
        E_R = numpy.zeros(N);
        E_T = numpy.zeros(N);

        eff_prof = []

        for i in range(0, N - 1):
            # this function calculates the efficiency associated with incidence angle and receiver efficiency
            n_rec[i] = self.receiver_eff_func(theta_r[i], rec_zleff, rec_b_0);

            # this is the position error, without taking max intensity into account -> useful for checking position error effects
            UA_F_disp = self.gaussNcone_transmission_func(r_aperture, r_b, d_ave, d[i], L_W)
            UA_n_pos[i] = self.position_eff_func(theta_r[i], pos_err, point_err, UA_F_disp, h[i], r);

            # this function will determine the efficiency associated with the pointing and position error of the satellite
            # This also incorperates lost energy from changing apperent reciever size
            n_pos[i] = self.position_eff_func(theta_r[i], pos_err, point_err, F_disp[i, :, :], h[i], r);

            E_R[i] = t_step * P_T[i] * n_rec[i] * n_pos[i]  # total energy recieved, per time step
            E_T[i] = t_step * P_T[i]  # Total energy transmit, per time step

            eff_prof.append(E_R[i] / E_T[i])  # efficiency profile

        E_R_tot = sum(E_R);  # This is in Joules ->
        E_R_tot = E_R_tot * 2.7778 * 10 ** -7  # kWh

        E_T_tot = sum(E_T);  # This is in Joules ->
        E_T_tot = E_T_tot * 2.7778 * 10 ** -7  # kWh

        Total_eff = E_R_tot / E_T_tot * 100

        return [t_end, E_R_tot, Total_eff, Feasible, alpha_ave, d_ave, theta_max_transmission, TransPosnTimeVec,
                eff_prof, theta_r, d, transmission_time_prof]

    def driver(self, currDesign):

        # orbits, numSats, panelSize, LI_battery_mass_total, laser_intake_wattage, r_aperture, r):

        # imports
        # import numpy
        # import math
        # import Power_Function_V4.PowerMain_Func3D

        # ID = allDesigns[design_ID].ID
        orbits = currDesign.orbits  # List of all orbits (family, trajectory, velocity, period, percent eclipsed) in current design
        numSats = currDesign.numSats  # Number of satellites on each orbit as list
        # totSats = allDesigns[design_ID].totSats  # Total number of satellites in constellation
        # solarPanelSize = allDesigns[design_ID].solarPanelSize  # Solar panel area [m^2]
        # batterySize = allDesigns[design_ID].batterySize  # Battery mass [kg]
        # laserPower = allDesigns[design_ID].laserPower  # Wattage required to power the laser [W]
        r_aperture = currDesign.apetRad  # radius of output lens on SC [m]
        r = currDesign.receiverRad_power  # radius of ground receiver [m]
        panelSize = currDesign.solarPanelSize
        LI_battery_mass_total = currDesign.batterySize
        laser_intake_wattage = currDesign.laserPower

        # simple magnitude function, used throughout
        def MagFunc(vec):
            return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

        # constants for adjusting from lunar to metric
        r_m = 1737400  # radius of the moon, m
        LU = 389703000  # distance of Earth to Moon in m
        T_m = 2360448  # lunar day in seconds

        # loop through all orbits extracting their data
        NumOrbits = len(orbits)
        PosnTime = []
        eclipse_percent = []
        Period = []
        for i in range(0, NumOrbits):
            orbit_curr = orbits[i]
            x = orbit_curr.x
            y = orbit_curr.y
            z = orbit_curr.z
            vx = orbit_curr.vx
            vy = orbit_curr.vy
            vz = orbit_curr.vz
            eclipse_percent.append(orbit_curr.eclipse)
            Period.append(orbit_curr.T * T_m)  # period in seconds

            t = numpy.linspace(0, Period[i], len(x))  # creates time vector, seconds
            PosnTime.append([LU * x, LU * y, LU * z, t])  # convert lunar units to seconds and meters

        ##### Ground Station Locating #####
        # From PosnTime, list of each orbit's [x,y,z,t] for this design, & NumOrbits determine which 2 orbits
        # are the ones transmitting to ground stations. Need to find transmitting orbit indices, ground station coordinates,
        # and pass over index

        current_GS_sich = self.GroundStationLocator(PosnTime, numSats)
        orbit_range = current_GS_sich[0]  # list of the 1 or 2 chosen orbit's indices in PosnTime/numSats
        GS_index = current_GS_sich[1]  # list of indices of gs cross over for the 1 or 2 chosen orbits

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

        for i in range(0, len(orbit_range)):

            # extract this orbit's position and time, groundstation cross over index, period, and eclipse percent
            thisPosnTime = PosnTime[orbit_range[i]]
            thisGS_index = GS_index[i]

            if thisGS_index > len(thisPosnTime[
                                      0]):  # rare case where GS_locator goes around the orbit too far, this just subtracts a total number of indices
                thisGS_index = thisGS_index - len(thisPosnTime[0])

            thisPeriod = Period[orbit_range[i]]
            thiseclipse_percent = eclipse_percent[orbit_range[i]]
            # quick receiver position calculation for this orbit
            d_t = MagFunc([thisPosnTime[0][thisGS_index], thisPosnTime[1][thisGS_index],
                           thisPosnTime[2][thisGS_index]])  # distance of transmission
            u_t = [thisPosnTime[0][thisGS_index] / d_t, thisPosnTime[1][thisGS_index] / d_t,
                   thisPosnTime[2][thisGS_index] / d_t]  # unit vector of transmission
            thispos_rec = [u_t[0] * r_m, u_t[1] * r_m,
                           u_t[2] * r_m]  # position of receiver, radius of moon * unit vector of transmission point
            pos_rec.append(thispos_rec)  # store this for later

            # unpacking of Current_Power_sich:

            if numSats[i] != 0:  # only append if the current orbit actually has sc on it

                Current_Power_sich = self.PowerMain_func3D(thisPosnTime, thispos_rec,
                                                                                         thisGS_index, thisPeriod,
                                                                                         thiseclipse_percent, panelSize,
                                                                                         LI_battery_mass_total,
                                                                                         laser_intake_wattage,
                                                                                         r_aperture, r)

                E_rec_perT.append(round(Current_Power_sich[1], 3))
                E_rec_perD.append(round(Current_Power_sich[1] * numSats[i] / (thisPeriod / (3600 * 24)),
                                        8))  # scaled by current period and num of sc
                Eff.append(Current_Power_sich[2])
                numSats_transmitting.append(numSats[i])
                alpha_ave.append(Current_Power_sich[4])
                periods.append(round(thisPeriod / (3600 * 24), 2))
                d_ave.append(round(Current_Power_sich[5], 3))
                theta_r_transmission.append(round(Current_Power_sich[6], 3))
                t_ends.append(round(Current_Power_sich[0], 3))

                # convert pos_rec to lat and lon
                thislat = round(math.asin(thispos_rec[2] / r_m) * 180 / numpy.pi, 3)
                thislon = round(math.atan2(thispos_rec[1], thispos_rec[0]) * 180 / numpy.pi, 3)
                latlon.append([thislat, thislon])

        # energy to each gs

        E_rec_perD_totdesign = sum(E_rec_perD)

        ##### PRINTING ######

        #
        print('Transmission Point & Orbit Parameters:')
        print('   View Angles: ',alpha_ave,'radians')
        print('   Distances: ',d_ave,'km')
        print('   Transmission times: ',t_ends,'seconds')
        print('   Max incident angle: ',theta_r_transmission,'degrees')
        print('   Number of SC: ',numSats_transmitting)
        print('   This designs two periods:',periods,'hours')

        print('Results:')
        print('   Ground Station Locations: ',latlon,'degrees lat,lon')
        print('   Energy Received, 24 hour average:',E_rec_perD,'kWh/24h')
        print('   Energy Received, single transmission:',E_rec_perT,'kWh')
        #

        return E_rec_perD_totdesign  # return total kWh/24hr delivered to the surface

##################

p = power()

allDesigns = []
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

lengthOfFiles = len(files) - 1

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
N_end = lengthOfFiles
# N_end = 1
print(N_end - N_start, 'designs being tested')
for i in range(N_start, N_end):
    design_ID = i

    design = allDesigns[design_ID]

    # design variable extraction
    # ID = allDesigns[design_ID].ID
    # orbits = allDesigns[design_ID].orbits  # List of all orbits (family, trajectory, velocity, period, percent eclipsed) in current design
    # numSats = allDesigns[design_ID].numSats  # Number of satellites on each orbit as list
    # totSats = allDesigns[design_ID].totSats  # Total number of satellites in constellation
    # solarPanelSize = allDesigns[design_ID].solarPanelSize  # Solar panel area [m^2]
    # batterySize = allDesigns[design_ID].batterySize  # Battery mass [kg]
    # laserPower = allDesigns[design_ID].laserPower  # Wattage required to power the laser [W]
    # r_aperture = allDesigns[design_ID].apetRad  # radius of output lens on SC [m]
    # r = allDesigns[design_ID].receiverRad_power  # radius of ground receiver [m]
    # panelSize = allDesigns[design_ID].solarPanelSize
    # LI_battery_mass_total = allDesigns[design_ID].batterySize
    # laser_intake_wattage = allDesigns[design_ID].laserPower

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
    print('Design', design_ID, ':\n')
    E_tot.append(p.driver(design))

max_E = max(E_tot)
max_E_ID = E_tot.index(max_E)

print("______________________________________________________")
print('\nFinal Results for All Designs:')
print('Maximum Energy achieved by a design:', max_E, 'kWh/24h')
print('Maximum Energy Orbit ID:', max_E_ID)
