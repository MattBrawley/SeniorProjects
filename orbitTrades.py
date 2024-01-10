"""
Script to demonstrate API for requesting periodic 3BP orbits from JPL database:
https://ssd.jpl.nasa.gov/tools/periodic_orbits.html#/periodic
"""


from logging.config import listen
from csltk.jpl_query import JPLFamily
from csltk import cr3bp
from csltk.utilities import System
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pickle
import orbitDict
from csltk.utilities import emSystem




# define Earth-Moon system and orbit parameters
# sys = System(mu=0.01215058560962404, lstar=389703.2648292776, tstar=382981.2891290545)
# ax = sys.plot_system()
LU = 389703 # 1 LU (distance from Earth to Moon in km)
xM = 0.98784942
distMoon = [xM, 0, 0] # moon pos in LU
# moonSun_vector = [-1,0,0] # vector from Moon to Sun LU 
moonSun_vector = [-150000000/LU,0,0] # vector from Moon to Sun LU 
moonSun_vector = moonSun_vector/np.linalg.norm(moonSun_vector) 
rMoonKm = 1737.4 
rMoon = 1737.4/LU # radius of the moon in LU
distL1 = abs(0.83691513 - distMoon[0]) # distance from moon to L1
distL2 = abs(1.15568217 - distMoon[0]) # distance from moon to L2
alt_UL = max(distL1, distL2) + max(distL1, distL2)*0.2# maximum allowable orbit altitude 
alt_LL = rMoon + 10/LU
maxStability = 30 # maximum allowable orbit stability
synodicPeriod = 29.523 # days 


families = ['lpo', 'dro', 'halo', 'dpo','butterfly'] # families to analyze (from trades)
orbitDict2 = []
# iterate throgh each family
counterRuns = 0
totOrbits = 0
orbPerFam = []
for i in range(len(families)):
    numOrbits = 0 # number of acceptable orbits
    print(families[i])
    # set necessary lagrange points for each family
    if families[i] == 'halo':
        lagrange = ['1', '2']
    else:
        lagrange = ['']

    # set necessary branches for each family
    if families[i] == 'lpo':
        branches = ['E', 'W']
    #elif families[i] == 'resonant':
    #    branches = ['pq']
    elif families[i] == 'halo' or families[i] == 'butterfly':
        branches = ['N', 'S']
    else: 
        branches = ['']

    for j in range(len(lagrange)):
        for k in range(len(branches)):
            # supported values: lpo, dro, dragonfly, halo, dpo, resonant, butterfly
            # pull JPL orbit initial conditions
            fam = JPLFamily(sys='earth-moon', fam = families[i], libr = lagrange[j], branch = branches[k])
            fam.set_stab_bounds(1,maxStability)
        
            
            orbits = fam.request()
            if families[i] == 'halo':
                num_to_plot = len(orbits)/20 # number of orbits in family to plot
            else: 
                num_to_plot = len(orbits)/200 # number of orbits in family to plot


            for orbit in orbits[::int(len(orbits)/num_to_plot)]:
                print('Run: ',counterRuns)
                # propagate orbit
                x0 = orbit.iState # initial conditions
                stability = orbit.stab # stability index 
                t_span = np.array([0, orbit.T]) # time span [TU]
                # print(x0[0],x0[1],x0[2])
                # print((orbit.T*orbit.system.tstar)*(1/60)*(1/60)*(1/24),'days')
                # print(orbit.T,'TU')
                t_eval = np.arange(0, orbit.T, 100/orbit.system.tstar) # evaluate orbit at every 100 seconds [TU]
            
                sol = cr3bp.propagate(orbit.system, x0, t_span, eval_times = t_eval) # propagate orbit 
                posX = sol.y[0,:]  # allocate x pos LU               
                posY = sol.y[1,:]  # allocate y pos LU 
                posZ = sol.y[2,:]  # allocate z pos LU 
    
                # # calculate orbit altitude from Moon
                alt = np.sqrt( (posX - distMoon[0])**2 + (posY - distMoon[1])**2 + (posZ - distMoon[2])**2) # distance between satellite and moon center  
                #print(alt)
                altMax = max(alt) # LU
                altMin = min(alt) # LU

                ## claculate % period time eclipsed 
                # vectors from moon to sc
                # https://gssc.esa.int/navipedia/index.php/Satellite_Eclipses  
                moonSC_vector = [posX - distMoon[0], posY - distMoon[1],posZ - distMoon[2]] # vector from moon to satellite 
                moonSC_vector = moonSC_vector/np.linalg.norm(moonSC_vector) # unit vector
                # calculate orbit altitude from Moon 
                check1 = np.dot(moonSun_vector,moonSC_vector) # = cos phi, both vectors are unit vectors , phi is angle between vectors 
                check2 = alt*np.sqrt(1 - check1**2) # = sin phi * alt 
                sunCounter = 0 # keeps track of sun view 
                

                # check if in view of the sun 
                for jj in range(len(check1)): 
                    if check1[jj] < 0 and check2[jj] < rMoon: 
                        sunCounter = sunCounter +1
                eclipse = (sunCounter/len(check1)) * 100 # percent time being eclipsed 

                # Check if orbit is within set boundaries: altitude and stability constraints 
                # print(altMax, '<',alt_UL,altMin,'>',alt_LL,stability,'<',maxStability)
                if (altMax < alt_UL)  and (altMin > alt_LL):  
                    numOrbits = numOrbits + 1 # counter of number of acceptable orbits
                    totOrbits = totOrbits + 1
                    filename = 'trajectories/orbit' 
                    filename = filename + str(totOrbits) + '.dat'
                    orb = orbitDict.chosenOrbits(families[i], posX, posY, posZ, sol.y[3,:], sol.y[4,:], sol.y[5,:], orbit.T, eclipse, stability) # save orbit 
                    orb.save(filename)
                    orbitDict2.append( orb )
                counterRuns = counterRuns + 1
    orbPerFam.append([numOrbits,families[i]])
