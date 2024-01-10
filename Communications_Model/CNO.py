#Function to calculate the carrier to noise ratio
#Nathan Tonella
#10/31/2022

import math as m
#10*LOG10(10^(pwrReceived/10) / 10^(receiverSysPwr/10))
def CNO(Pr,rSP):
    carrierNO = 10*m.log10(10^(Pr / 10) / 10^(rSP / 10));
    return carrierNO