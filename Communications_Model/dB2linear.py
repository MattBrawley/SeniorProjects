#Function to convert dB values to linear values
#Nathan Tonella
#10/29/2022

import math as m

def dB2linear(dB):
    linear = 10^(dB / 10);
    return linear