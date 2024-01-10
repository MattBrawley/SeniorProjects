from cmath import log10
from math import pi
from pickletools import float8
import wave

#Importing Functions
from CommsMOO import CommsMOO
from AntennaParameters import AntennaParameters
from SpaceLoss import SpaceLoss
from numpy import linspace

Altitude = 30000e3
Range_SideLink = 4000e3 
TimeLOS = 28800

margin,check,amnt = CommsMOO(currDesign)

#DRIVER INPUTS: CURR DESIGN, CONSTRAINTS
    #IF it meets comms constraint: constraint = 1 ONLY FOCUS ON CONSTRAINTS RELEVANT TO ME
    #do this in the moo
    #FIX FBD, NO LONGER DISCARD 
    #alt is calculated from xyz wrt, xyz --> pythagorize and subtract


################ Transmission Amount/ Coverage #################
#    LunarTransReq = 600e6 # 600 MB in bytes
#    LunarTransAmt = DataRate * TimeLOS

######### Design Variables Needed for currDesign ##########
#diameterTxm
#diameterTxo
#dataRate
#dataRate_Ed
#Altitude off the lunar surface
#Rangebtwn Sidelinks
