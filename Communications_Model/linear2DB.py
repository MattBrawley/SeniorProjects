#Function to convert linear values to dB
#Nathan Tonella
#10/29/2022
import math as m

def linear2DB(linear):
    dB = 10*m.log10(linear);
    return dB
