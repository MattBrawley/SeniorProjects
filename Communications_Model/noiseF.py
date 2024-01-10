#Function to calculate noise factor of the system
#Nathan Tonella
#10/29/22

def noiseF(Tphys,Tr):
    F = 1 + (Tphys / Tr);
    return F