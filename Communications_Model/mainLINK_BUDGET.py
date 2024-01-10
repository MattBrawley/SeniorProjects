#Main function: Runs through entire link budget to calculate CNO
#Nathan Tonella
#10/31/2022

#This main code is the link budget. It is divided into sections of the budget. In each section important
#variables for said section are declared, and values computed at the end

#Import all functions
from urllib.robotparser import RequestRate
import antennaNoiseT
import CNO
import dB2linear
import EIRP
import FOM
import linear2DB
import minCNO
import noiseF
import noiseT
import pwrReceived
import recieverSysPwr
import reqCNO
import transmitLLoss
import math as m
import conversions
#Define initial constants
c = 3 * 10**8; #speed of light [m/s]
f = float(input("What is the frequency [hz]\n"));  #frequency [Hz]
print("Type f is: ", type(f));
print("F is ", f);
k = 1.38 * 10**(-23);  #boltzman's constant [W/(Hz-K)]
linkLambda = float(input("What is the wavelength [m]\n")); #wavelength [m] (change)
#range is going to be different once we take into account the orbits: change this later for lvl2?
linkRange = float(input("What is the range [m]\n"));  #range for downlink,crosslink,uplink, etc.


#DATA PARAMETERS
BER = 10**(-5);    #bit error rate
reqEbNO = 9.8;  #required bit energy to noise ratio [dB]
R = float(input("What is the data rate [bps]\n"));    #data rate [bps]
reqMargin = 6;  #required design margin [dB]

#find the required C/Noise
requiredCNO = reqCNO.reqCNO(reqEbNO,R);
minCNO_link = minCNO.minCNO(requiredCNO,reqMargin);

#NOISE
T0 = 290;   #reference temperature [K]
n = 0.4;    #antenna efficiency
Tphys = float(input("What is the physical temperature of the antenna [K]\n")); #physical temperature: based on surrounding environment
Text = float(input("What is the external noise temperature [K]\n"));   #temperature based on what antenna is looking at [K]
Tr = 289;   #receiver noise temp [K]
Tant = antennaNoiseT.antennaNoiseT(n,Tphys,Text);

Lc = -1;    #receiver cable loss [dB]
F = noiseF.noiseF(Tphys,Tr);
Ts = noiseT.noiseT(Text,Tant,Tr);
No = recieverSysPwr.recieverSysPower(k,Ts);

#RECEIVER ANTENNA FUNCTIONS (TODO)
#Need to implement Mark's functions for antenna design
diameter_receive = float(input("Enter receiver antenna diameter [m]\n"));
er = 0.1;   #pointing accuracy [deg]
Lpr = -3;   #loss from angle errors/pointing loss [dB]
Lc = -1;    #cable loss [db]
area_receive = m.pi()*(diameter_receive / 2)**2;
Gr = 2 * conversions.linear2DB(4 * m.pi() * area_receive / linkLambda ** 2); #HARDCODE: Change when antenna functions input
Ae_receive = 0.55 * area_receive;   #effective area
qr = 70 * (linkLambda / diameter_receive);  #beamwidth

FigOM = FOM.FOM(Gr,Ts); 


#PROPAGATION PARAMETERS
Ls = -2 * conversions.linear2DB(4 * m.pi() * linkRange / linkLambda);
L = Ls;
#TRANSMIT PARAMETERS
diameter_transmit = float(input("Enter transmitter antenna diameter [m]\n"));
area_transmit = m.pi()*(diameter_transmit / 2) ** 2;
et = 0.1;   #pointing accuracy
Lpt = -3;   #pointing loss
Lt = -1;    #line loss (transmitter)
Gt = 2 * conversions.linear2DB(4 * m.pi() * area_receive / linkLambda ** 2); #HARDCODE: Change when antenna functions input
Ae_transmit = 0.55 * area_receive;   #effective area
qt = 70 * (linkLambda / diameter_receive);  #beamwidth
Pt = float(input("Enter transmitter power [W]\n"));
Pt = conversions.dB2linear(Pt);
eirp_link = EIRP.EIRP(Pt,Gt);

#LINK BUDGET
Pr = pwrReceived.pwrReceived(eirp_link,Lpt,Ls,Gr);
cno_link = CNO.CNO(Pr,No);
link_margin = cno_link - minCNO_link;


