# GAUSS & CONULAR Combination
# Created: 10/24/2022
# Finalized: 2/2/2023
# Author: Cameron King

# Sub Functions Needed :
# transmission_funcs.gaussian_transmission_func

def gaussNcone_transmission_func(r_aperture, r_ave, d_ave, d, P0):
    # This function will combine the gaussian distribution output from gaussian_transmission_func, with a conular dispursion.
    # The flux dispursion from gaussian_transmission_func is projected onto a cone, scaling up the radius corresponding to a given
    # shell as a function of distance. The output is the flux dispursion, [r_vec,P_within_vec], accuratly modeled with a cone.

    # imports
    import transmission_funcs
    import mpmath
    import math

    mpmath.dps = 30
    # 99% of the beam is in 1.52*r, where r is the radius of the beam IN THE GAUSSIAN FORMULATION
    # dividing by 1.52 here allows for the calculation of a beam that contains 99% in r_aperature = r*1.52
    r = r_aperture/1.52

    # hardcoded value for distance from the laser to the output lens. Doesn't matter too much, as it is a cylinder during that time and does not dispurse
    d_lens = 1

    # Get the flux distribution and radii for gaussian beam with radius of the output lens, at the output lens
    # this does the gaussian formulation of the math to determine power in each shell WITHOUT EXPANDING IT ON A CONE
    F_disp = transmission_funcs.gaussian_transmission_func(r, d_lens, d_lens, P0)

    # determine cone shape and radius ratio
    alpha = mpmath.atan((r_ave-r_aperture)/d_ave) # view angle, half of FOV
    focal_length = r_aperture/abs(mpmath.tan(alpha)) # focal length of necessary lens
        
    for i in range(0,len(F_disp[0])): # ranges through all radii given from gaussian_transmission_func, scaling up as it goes
        
        r_new = F_disp[0][i]*(d_ave/focal_length + 1) # radius of this ring at average distance
        F_disp[0][i] = r_new/d_ave*d # scale this new radius by current distance
    
    # CHECK: is the sum of the shell percentages close to 100%? throw error if not
    tot_shell_perc = sum(F_disp[1][:])-F_disp[1][0]    
    if tot_shell_perc < 0.98:
        print('Total shell percentage = ',tot_shell_perc)
        raise ValueError('Total Shell Percentage doesnt sum to >98%. Check gaussian_transmission_func & calls.')

    return F_disp # return the flux dispursion on a cone in the same format as given from gaussian_transmission_func
