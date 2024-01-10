#Function to calculate the miminum CNO from the required CNO and Design Margin
#Nathan Tonella
#10/29/2022

def minCNO(reqCNO,D):
    minCNO = reqCNO - D;
    return minCNO