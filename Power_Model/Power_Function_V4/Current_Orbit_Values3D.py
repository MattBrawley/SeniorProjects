# Current Orbit Values Function
# Created: 10/27/2022
# Author: Cameron King
# finalized: Cameron King; 2/2/2023

# Description:
#
# This function takes in the current positon vector of the SC, the position vector of the groundstation, and the receiver radius
# outputs the current distance, sat view angle, receiver incident angle, FOV, adjusted radius, and altitude

def Current_Orbit_Values3D(pos_sc,pos_rec,r):

    # imports
    import numpy
    import math
    import mpmath
    
    # quick magnitude function used throughout
    def MagFunc(vec):
        return math.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
    
    # Constants
    r_m = 1737400; # radius of the moon, m

    # current angle between receiver, SC, and center of moon
    alpha = mpmath.acos(numpy.dot(pos_sc,pos_rec)/(MagFunc(pos_sc)*MagFunc(pos_rec)))

    # current distance vector from SC to Receiver
    d_vec = [pos_sc[0]-pos_rec[0],pos_sc[1]-pos_rec[1],pos_sc[2]-pos_rec[2]]
    d = MagFunc(d_vec) # distance magnitude

    # calculate receiver incident angle, negetive d vector calculated for it
    d_vec_neg = [-pos_sc[0]+pos_rec[0],-pos_sc[1]+pos_rec[1],-pos_sc[2]+pos_rec[2]]
    theta_r = mpmath.acos(numpy.dot(d_vec_neg,pos_rec)/(MagFunc(d_vec)*MagFunc(pos_rec)))
    # print("numerator: ", numpy.dot(d_vec_neg,pos_rec))
    # print("denominator: ", (MagFunc(d_vec)*MagFunc(pos_rec)))
    theta_r = mpmath.re(theta_r)

    while theta_r >= numpy.pi/2 or theta_r < -numpy.pi/2:
        if theta_r >= numpy.pi/2:
            theta_r = theta_r - numpy.pi
        if theta_r < -numpy.pi/2:
            theta_r = theta_r + numpy.pi

    # calculate SC pointing angle
    theta_s = theta_r-alpha

    # calculate adjusted radius
    r_prime = r*mpmath.sin(math.pi/2 - theta_r)

    # calculate field of view
    # print("r_prime: ",r_prime)
    # print("r", r)

    FOV = 2*(mpmath.atan(r_prime/(d-math.sqrt(r**2-r_prime**2)))) # field of view of the receiver

    # calculate altitude
    h = MagFunc(pos_sc) - r_m
        
    return [d,theta_s,theta_r,FOV,r_prime,h]