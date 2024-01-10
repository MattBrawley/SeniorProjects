# Creator: Alec Church; Created file 10/22
# Author: Cameron King; Edited 10/24, converted from skeleton to current configuration
# Completion: Cameron King; 11/18 Completed the function to the ability necessary for MOO
# Finalized 2/2: Cameron King; finalized comments, cleaned and proofread code, deemed no longer necessary to edit

def gaussian_transmission_func(radius, av_d, curr_d, P0):
    # This function takes in the input of the radius at aperture, the average distance to the beam, the current distance, and the initial power
    # the goal is to output a matrix with a radius in one column and a power value in another. The power corresponds to the power contained within
    # the current 'shell', or the ring between the current radius and the last radius.
    # It is important to note that this function does not make the beam a cone, just a cylinder.

    # imports
    import numpy as np
    
    # constants
    Lambda = 532 * 10**(-12) # meters
    
    # Waist of the beam is the size of the aperture
    w0 = radius

    # Calculate Rayleigh length needed for the average distance, wavelength, and target radius
    z_R = (w0**2 * np.pi / Lambda)
        
    # Calculate the actual radius of the beam at the changing distance (near average distance)
    w = w0 * np.sqrt(1 + (curr_d / z_R)**2)
    
    # Choose max radius to calculate intensity to, should be equal to radius of beam at surface & current time
    r_max = w*1.52

    # preallocation
    N = 10 # number of rings generated
    r_step = r_max/N # distance between edges of shells
    P_within = [] # power contained within the shell, in PERCENT
    r = np.arange(0,r_max+r_step,r_step) # range of radii used in generation of P_within
    r_vec = [] # vector of radii corresponding to P_within
    
    # Loop through range of radii, r
    for i in range(0,len(r)):
        if i == 0: # for the first row, assign a 0 radii and the total initial power, for extraction later
            P_within.append(P0)
        elif i == 1: # for the first radii, it is a circle not a shell, therefore we use this simplified equation
            P_within.append((1-np.exp((-2 * r[i]**2) / (w)**2)))
        else: # for all other cases, use a shell calculation which includes a term with the previous radius r[i-1]
            P_within.append((1-np.exp((-2 * r[i]**2) / (w)**2)) - (1-np.exp((-2 * r[i-1]**2) / (w)**2)))
            
        r_vec.append(r[i]) # assign the r used to the storage vector of r, r_vec

    return [r_vec, P_within] # a list of lists, the radius vector and percent power contained vector