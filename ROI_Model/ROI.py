#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:29:07 2022

@author: alecchurch
"""


# Power ROI Analysis Setup for MOO

num_sat = 10 # MOO

num_cust = [2,4] # [constant, half month] # Trade study

cust_needs = 5 # (kWh) # Trade study, some half month, some all the time

price_power = 10 # ($/kWh) # We choose reasonable value

price_comms = 200000 # $/year # Trade study

cost_sat = 1.5 * 10**6 # ($/sat) # Trade study

sat_mass = 350 # (kg) # Trade study

cost_launch = 10 * 10**6 # ($) # Trade study ~ >$5e6

mass_launch = 13000 # (kg) # Max mass per launch

cost_overhead = 800000 # ($) # Overhead annual costs


def ROI(num_sat,num_cust,cust_needs,price_power,cost_sat,sat_mass,cost_launch,mass_launch,overhead):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Startup cost
    roundup_launch = np.ceil(num_sat * sat_mass / mass_launch)
    invest = (cost_launch * roundup_launch + cost_sat * num_sat) 
    # Annual costs
    roundup_launch = np.ceil(8 * sat_mass / mass_launch)
    relaunch = (cost_launch/2 * roundup_launch + cost_sat * 8)
        
    # Power income
    period_power = (365.25/29.531)*(num_cust[0] + 0.5*num_cust[1])
    annual_income_P = period_power*cust_needs*29.531*24*price_power
    # Comms income
    total_cust = num_cust[0] + num_cust[1]
    annual_income_C = total_cust * price_comms
        
    # Total income
    income = annual_income_P + annual_income_C
    
    lock = False
    j = 1
    time = 44 #years
    
    bank = np.zeros(2*time)
    incomes = np.zeros(2*time//5)
    relaunches = np.zeros(2*time//5)
    overheads = np.zeros(2*time//5)
    
    bank[0] = -invest
    incomes[0] = income
    relaunches[0] = relaunch
    overheads[0] = overhead
    
    for i in range(2*time):  
        if (i%10) == 0:
            incomes[j] = incomes[j-1] * 1.05
            relaunches[j] = relaunches[j-1] * 1.05
            overheads[j] = overheads[j-1] * 1.05
            
            income = incomes[j]
            relaunch = relaunches[j]
            overhead = overheads[j]
            j += 1
            
        if (i/2)%10 == 0:
            print('10 year Internal rate of return: {}%.'.format(100*income/invest))
        if i%2 == 0:
            bank[i] = bank[i-1] + income/2
        else:
            bank[i] = bank[i-1] + income/2 - overhead
        if (i/2)%15 == 0: # Every 15 years, launch new constellation
            bank[i] -= relaunch
        
    
    roiFit = np.polyfit(range(0,2*time),bank,3)
    roiPolyfit = np.polyval(roiFit,range(0,2*time))  
    
    for i in range(1,2*time):
        if (roiPolyfit[i] > 0) and (lock == False):
            print('Predicted break even in {} years.'.format(i/2))
            lock = True
    
    x = np.linspace(0,time,2*time)
    
    plt.figure(0)
    plt.plot(x,bank/(10**6),'k', label = 'Predicted annual costs')
    plt.plot(x,roiPolyfit/(10**6),'r', label = 'Costs fit over time')
    plt.xlabel('Years in Operation')
    plt.ylabel('CRATER Profit ($ 10^6)')
    plt.title('CRATER ROI')
    plt.grid()
    plt.legend()
    plt.show()
    
    
    
    return bank, roiPolyfit


bank, roiPolyfit = ROI(num_sat,num_cust,cust_needs,price_power,cost_sat,sat_mass,cost_launch,mass_launch,cost_overhead)
