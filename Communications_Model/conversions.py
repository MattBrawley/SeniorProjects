#This function covers conversions from linear to dB
#Nathan Tonella
#10/31/2022

import math as m

def linear2DB(linear):
    dB = 10*m.log10(linear);
    return dB

def dB2linear(dB):
    linear = 10^(dB / 10);
    return linear    