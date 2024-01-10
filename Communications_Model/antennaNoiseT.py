#Function to find the antenna noise temperature
#Nathan Tonella
#10/31/2022

def antennaNoiseT(n,Tphys,Text):
    Tant = n * Text + (1 - n) * Tphys;
    return Tant