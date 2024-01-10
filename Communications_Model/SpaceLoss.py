from cmath import log10
from math import pi

def SpaceLoss(distance,wavelength):
    #Inputs: Distance in METERS & Wavelength in Hz
    #Outputs: Spaceloss in DB
    #Method: Using FSPL formula

    loss = -1*10*log10(((4*pi*distance)/wavelength)**2)
    return(loss.real) #return only the real part