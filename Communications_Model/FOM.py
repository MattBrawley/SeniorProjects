#Function to calculate figure of merit for receiver
#Nathan Tonella
#10/31/2022

import math as m

def FOM(Gr,Ts):
    FOM = Gr - 10 * m.log10(Ts);
    return FOM