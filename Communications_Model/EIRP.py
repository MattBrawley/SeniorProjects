#Function to calculate EIRP (db)
#Nathan Tonella
#10/31/2022

import math as m

def EIRP(Pt,Gt):
    #assume Pt, Gt are in dB
    return Pt + Gt