import numpy as np


# Move path to main CRATER directory
import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current) 
# adding the parent directory to the sys.path.
sys.path.append(parent)
 
from orbitDict import chosenOrbits
from design import designConfiguration



class roi:
    def cost(self, solarPanelSize, batterySize, powerReceiverRad, laserPower, diameterTxM, diameterTxO, totSats, orbits):
        # - WEIGHT COSTS - #
        year = 10
        sat_mass = 3200  # kg max
        cost_kg = 58000  # $58000/kg
        weight_area_con = 1.76  # kg/m² for 3 mil thickness of coverglass
        # ^ https://www.spectrolab.com/DataSheets/Panel/panels.pdf
        receiver_area = np.pi*powerReceiverRad**2  # m^2
        antenna_area = (diameterTxO/2)**2*np.pi+(diameterTxM/2)**2*np.pi
        antennaWeight = antenna_area*10/0.7
        weight = weight_area_con*(solarPanelSize + receiver_area) + totSats*sat_mass + batterySize + antennaWeight
        cost_weight = cost_kg*weight
        print("weight")
        print(weight)

        # - Component COSTS - #
        antenna_cost = (diameterTxM+diameterTxO)*10000*totSats
        laser_cost = 50*laserPower/(1e4*np.pi*(powerReceiverRad)**2)

        # - LAUNCH COSTS - #
        numOrbits = len(orbits)     
        launch_cost = (4*10**9)*numOrbits  # 4 billion for every rocket launch   
        launch_cost = launch_cost*(weight/64000)

        # - PROPELLANT AND THRUST COSTS - #
        propellant_cost = 850  # USD/kg for Xenon (high molecular weight)
        thrust = 150*10**(-3)  # N 40 to 600 mN
        acceleration = thrust/sat_mass
        delta_V =( 3 / 1.5 )  # 3 m/s of delta_V for a stability index of 1.5
        # ^ (rough estimate, JWST is in halo orbit and requires 2 - 4 m/s A YEAR)
        # We readjust every orbit, roughly 12 times more often than JWST but same burn)
        delta_V_needed = 0
        for s in range(len(orbits)):
            delta_V_needed += delta_V * orbits[s].stability
        burn_time = delta_V_needed/acceleration
        prop_mass_usage = 0.2*10**(-6)*burn_time  # 0.2 mg/s Xenon mass flow rate (10^-6 is for mg to kg)
        propellant_mass = prop_mass_usage*12*year*1.05  # times number of corrections
        # propellant_mass=425 kg for Dawn Spacecraft + 5% to account for delta V mass
        # ^ http://aerospace.mtu.edu
        # stability index of 1 is no maintenance
        propellant_total_cost = propellant_mass*propellant_cost*totSats

        # - SATELLITE BUILD COSTS - #
        cost_sat = 150*10**6
        cost_ground_station_total = 2*cost_sat

        # - TOTAL COSTS - #
        # Other model charges launch as a % of 64000 then multiplies by 4 billy
        total_cost = cost_weight + (cost_sat*totSats) + cost_ground_station_total + launch_cost + propellant_total_cost + antenna_cost + laser_cost
        # total_cost_no_launch = total_cost-launch_cost
        return total_cost
  
    def revenue(self, powerOutputs, commsOutput, totSats):
        constraints = 1
        num_cust = 6
        year = 10

        cust_needs_power = 60  # [kWh] daily consumption
        cust_needs_comms = 10**3  # [GB] daily consumption (curiosity uses 100-250 megabits a day, person uses ~34 gb, one hour of netflix is 1 gb hour)
        # ^ Can randomise for more accurate model
        price_power = 95  # [$/kWh]
        price_comms = 30  # [$/GB]

        TU_to_days = 382981.2891290545/86400 # days/TU
        E_received = powerOutputs[0]
        E_period = powerOutputs[1]*TU_to_days

        synodicPeriod = 29.523 # days 
        daily_power_provided = E_received/E_period
        daily_comms_provided = commsOutput/synodicPeriod


        rev_power = year * (daily_power_provided * 365 * price_power)
        rev_comms = year * (daily_comms_provided * 365 * price_comms)

        if (daily_power_provided > cust_needs_power) or (daily_comms_provided > cust_needs_comms):
            constraints = 0
            
        revenue = rev_power + rev_comms

        return revenue, constraints
    
    def driver(self, currDesign, powerOutput, commsOutput):
        # design variable extraction
        # design variable extraction
        ID = currDesign.ID
        orbits = currDesign.orbits # List of all orbits (family, trajectory, velocity, period, percent eclipsed) in current design
        totSats = currDesign.totSats # Total number of satellites in constellation 
        solarPanelSize = currDesign.solarPanel # Solar panel area [m^2]
        laserPower = currDesign.laserPower
        batterySize = currDesign.batterySize # Battery mass [kg]
        receiverRad_power = currDesign.receiverRad_power # radius of ground receiver [m]
        diameterTxM = currDesign.diameterTxM # antenna diameter of the receiver on the moon [m]
        diameterTxO = currDesign.diameterTxO # antenna diameter of the receiver on the satellite [m]

        # Calculate ROI objective
        totRevenue, contraints = self.revenue(powerOutput, commsOutput, totSats)
        totCost = self.cost(solarPanelSize, batterySize, receiverRad_power, laserPower, diameterTxM, diameterTxO, totSats, orbits)
        print("cost")
        print(totCost)
        rev_cost = totRevenue/totCost

        print("rev/cost")
        print(rev_cost)
        currDesign.add_roiObj(rev_cost)
        return contraints
