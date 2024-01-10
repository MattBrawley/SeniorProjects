import numpy as np

# ROI
# Cost as a func of weight and then weight as a func of size

year = 10  # For revenue @ this year

# - ASHWIN THESE ARE THE VARIABLES I GAVE RANDOM VARIABLES TO TEST - #

daily_power_provided = 500  # [kWh] daily capability
daily_comms_provided = 10000  # [GB] daily capability

solar_area = 20  # m^2 (Example)
battery_mass = 1000  # 200 to 5000 kg according to power team
receiver_radius = 4  # 1 to 20 m
communication_antenna_diameter = 1.5  # 1 to 5m according to comms team
ground_station_antenna_diameter = 1  # m

num_cust = 6  # number of customers
number_of_satellites = 10  # satellites
number_of_orbits = 5  # orbits

cust_needs_power = 50  # [kWh] daily consumption
cust_needs_comms = 10**3  # [GB] daily consumption (curiosity uses 100-250 megabits a day, person uses ~34 gb, one hour of netflix is 1 gb hour)
# ^ Can randomise for more accurate model
price_power = 95  # [$/kWh]
price_comms = 30  # [$/GB]

# - WEIGHT (AREA) COSTS - #

sat_mass = 3200  # kg max

cost_kg = 58000  # $58000/kg

weight_area_con = 1.76  # kg/m² for 3 mil thickness of coverglass
weight_area_lib = 2.06  # kg/m² for 6 mil thickness of coverglass
# ^ https://www.spectrolab.com/DataSheets/Panel/panels.pdf

receiver_area = np.pi*receiver_radius**2  # m^2
antenna_area = (communication_antenna_diameter/2)**2*np.pi+(ground_station_antenna_diameter/2)**2*np.pi

# - LAUNCH COSTS - #

# launch_cost = (4*10**9)*number_of_orbits  # 4 billion for every rocket launch (SLS Artemis system basically)
launc_cost = (7.5*10**6)*number_of_orbits # Rocket lab electron

# - PROPELLANT AND THRUST COSTS - #

stability = 1

propellant_cost = 850  # USD/kg for Xenon (high molecular weight)
thrust = 150*10**(-3)  # N 40 to 600 mN
acceleration = thrust/sat_mass
delta_V =( 3 / 1.5 )  # 3 m/s of delta_V for a stability index of 1.5
# ^ (rough estimate, JWST is in halo orbit and requires 2 - 4 m/s A YEAR)
# We readjust every orbit, roughly 12 times more often than JWST but same burn)
delta_V_needed = delta_V * stability
burn_time = delta_V_needed/acceleration
prop_mass_usage = 0.2*10**(-6)*burn_time  # 0.2 mg/s Xenon mass flow rate (10^-6 is for mg to kg)
propellant_mass = prop_mass_usage*12*year  # times number of corrections
print('propellant mass =', propellant_mass, 'kg')
# propellant_mass=425 kg for Dawn Spacecraft
# ^ http://aerospace.mtu.edu
# stability index of 1 is no maintenance
propellant_total_cost = propellant_mass*propellant_cost*number_of_satellites

# - SATELLITE BUILD COSTS - #

cost_sat = 150*10**6
cost_ground_station_total = 2*cost_sat


# - TOTAL COSTS - #

weight = weight_area_con*(solar_area+receiver_area)+number_of_satellites*sat_mass+battery_mass+(antenna_area*14.2857143)  # 10 kg per 0.7 m^2 for antenna
cost_weight = cost_kg*weight

cost_antennas = (ground_station_antenna_diameter+communication_antenna_diameter)*10000*number_of_satellites

print('total weight =', weight, '<64000 to be launchable')
# Other model charges launch as a % of 64000 then multiplies by 4 billy

launch_cost = launch_cost*(weight/64000)

total_cost = cost_weight+(cost_sat*number_of_satellites)+cost_ground_station_total+launch_cost+propellant_total_cost+cost_antennas

total_cost_no_launch = total_cost-launch_cost

print('total cost =', total_cost)

print('total cost without launch costs =', total_cost_no_launch)

# - REVENUE - #

# profit_power = year * (cust_needs_power * num_cust * number_of_satellites * 365 * price_power);
# profit_comms = year * (cust_needs_comms * num_cust * number_of_satellites * 365 * price_comms);
# ^ Daddy Ashwin was unhappy with those :(

profit_power = year * (daily_power_provided * 365 * price_power)
profit_comms = year * (daily_comms_provided * 365 * price_comms)

if daily_power_provided > cust_needs_power:
    print('Customer Power Demands Met :)')
else:
    print('Customer Power Demands NOT Met >:( ')

if daily_comms_provided > cust_needs_comms:
    print('Customer Comms Demands Met :)')
else:
    print('Customer Comms Demands NOT Met >:( ')

percent_power_used = (cust_needs_power * num_cust)/daily_power_provided
print('%', percent_power_used, 'of Power Available being used')

percent_comms_used = (cust_needs_comms * num_cust)/(daily_comms_provided)
print('%', percent_comms_used, 'of Comms Available being used')

profits = profit_power + profit_comms - total_cost

print('Revenue =', profits)