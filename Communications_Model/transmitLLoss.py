#Function to calculate the transmitter line loss given length of cord and the loss/ft
#Nathan Tonella
#10/31/2022

def lineLoss(l,anttenuation):
    loss = l*anttenuation;
    return loss