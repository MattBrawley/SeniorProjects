# Efficiency Functions:
# Created: Cameron King; 10/20/2022
# Finalized: Cameron King; 2/2/2023
# this file contains the two efficiency functions needed in the power model: position_eff_func & receiver_eff_func
# each of them calculate an efficiency that corresponds to a given aspect of transmission

def position_eff_func(theta, pos_err, point_err, F_disp, h, r):
    # This function determines the efficiency associated with the error in pointing and position
    # The efficiency calculated is specifically the energy hitting the receiver / Energy leaving the laser at the given time

    # imports
    import mpmath
    import math
    import numpy

    # define errors in each dimension, these are input
    xpos_err = pos_err[0];
    ypos_err = pos_err[1];
    hpos_err = pos_err[2];
    theta_err = point_err[0];
    phi_err = point_err[1];

    # Define/calculate other necessary values, taking errors in
    x = h*mpmath.tan(theta);
    d = math.sqrt((x+xpos_err)**2 + (h+hpos_err)**2)
    FOV = 2*(mpmath.atan((x+xpos_err+r)/(h+hpos_err))-mpmath.atan((x+xpos_err)/(h+hpos_err)))
    r_prime = d*mpmath.atan(FOV/2)
    
    # find total errors
    xtheta_err = h*mpmath.tan(theta+theta_err)-x;
    x_err = xpos_err + xtheta_err;
    yphi_err = d*mpmath.tan(phi_err);
    y_err = ypos_err + yphi_err;

    # define shell radii, & their respective flux %
    shell_num = len(F_disp[0])-1;
    r_shell = F_disp[0];
    Fperc_shell = F_disp[1];

    # Preallocation
    A_hit = numpy.zeros(shell_num);
    A_avail = numpy.zeros(shell_num);
    hit_eff = numpy.zeros(shell_num);
    shell_eff = numpy.zeros(shell_num);
    
    # Loop through each shell, determine the area available to hit and the area actually hit
    for shell_index in range(0,shell_num):

        # define this shells inner and outer radii
        if shell_index == 0:
            r_outer = r_shell[shell_index+1];
            r_inner = 0;
        else:
            r_outer = r_shell[shell_index+1];
            r_inner = r_shell[shell_index];
    
        
        dx = r_outer/10 # define the distance and area steps
        dA = dx**2
            
        # loop through the total range in x dimension of the shell
        for x in numpy.arange(-r_outer,r_outer,dx):

            y_outer = math.sqrt(r_outer**2 - x**2); # y of edges of shell at current x

            # loop through the max and min y at current x, not including within the shell when x < r_inner
            if abs(x) > abs(r_inner):
                # if we are not needing to 'jump' over the center of the shell, start and end of scan in x

                for y in numpy.arange(-y_outer,y_outer,dx):
                
                    # check if point is on the receiver 1, add to area hit if it is, add to area available to hit either way
                    hit_value = ((x-x_err)**2)/(r_prime**2) + ((y-y_err)**2)/(r**2)
                    if hit_value <= 1:
                        A_hit[shell_index] = A_hit[shell_index] + dA
                        A_avail[shell_index] = A_avail[shell_index] + dA
                    else:
                        A_avail[shell_index] = A_avail[shell_index] + dA

            else:
                # if we get here, we are in the range where we have to scan the top and bottom halves of the shell in y
                
                y_inner = math.sqrt(r_inner**2 - x**2) # this is the inner bound on the y range
                
                for y in numpy.arange(-y_outer,-y_inner,dx): # for the bottom half of the shell...
                
                    # check if point is on the receiver 1, add to area hit if it is, add to area available to hit either way
                    hit_value = ((x-x_err)**2)/(r_prime**2) + ((y-y_err)**2)/(r**2)
                    if hit_value <= 1:
                        A_hit[shell_index] = A_hit[shell_index] + dA
                        A_avail[shell_index] = A_avail[shell_index] + dA
                    else:
                        A_avail[shell_index] = A_avail[shell_index] + dA
                        
                for y in numpy.arange(y_inner,y_outer,dx): # for the top half of the shell...
                                    
                    # check if point is on the receiver 1, add to area hit if it is, add to area available to hit either way
                    hit_value = ((x-x_err)**2)/(r_prime**2) + ((y-y_err)**2)/(r**2)
                    if hit_value <= 1:
                        A_hit[shell_index] = A_hit[shell_index] + dA
                        A_avail[shell_index] = A_avail[shell_index] + dA
                    else:
                        A_avail[shell_index] = A_avail[shell_index] + dA
                
        # now, per shell, find out efficiency
        if A_avail[shell_index] == 0: # This only happens when the area available to hit is so small the step size rounds it to zero

            # check if center of beam is on the ellipse or not, 100% efficiency if it is and 0% efficiency if it isnt
            if ((0-x_err)**2)/(r_prime**2) + ((0-y_err)**2)/(r**2) <= 1:
                hit_eff[shell_index] = 1
            else:
                hit_eff[shell_index] = 0

        else: # in all other cases, with out A_avail = 0, below is solved to be the efficiency of position per shell
            hit_eff[shell_index] = A_hit[shell_index]/A_avail[shell_index] # if this is close to 1, then the shell is 100% on the receiver, even with error

        # now, adjust the percent of total flux contained in a shell by the hit efficiency of that shell
        # the sum of this is going to be the total position efficiency, a sum which would have been 100% with complete position efficiency
        shell_eff[shell_index] = hit_eff[shell_index]*Fperc_shell[shell_index+1]

    return sum(shell_eff)



def receiver_eff_func(theta, zero_loss_eff, b_0):
# This function determines the efficiency of a receiver as a function of incident angle

    # imports
    import mpmath

    # this is the cutoff of usable power as a function of reflectance
    theta_cutoff = mpmath.acos(b_0/(1+b_0))

    if theta > theta_cutoff: # if the current angle is above cutoff, set efficiency to 0
        n_rec = 0
    else: # if we are within angle range...
        K_theta = 1-b_0*((1/mpmath.cos(theta))-1) # incident angle modifier, scales zero loss efficiency
        n_rec = zero_loss_eff*K_theta # receiver associated efficiency when in angle range

    if n_rec < 0: # catch just in case
        n_rec = 0

    return n_rec # output the receiver efficiency