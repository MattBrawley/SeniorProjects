#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:44:34 2022

@author: alecchurch
"""

def powerFunc(radius, av_d, curr_d, P0):
    import numpy as np
    
    '''  
    https://www.keyence.com/ss/products/marking/laser-marking-central/basics/principle.jsp
    # Constants & Calculated Variables    
           Second Harmonic Generation (SHG) lasers use a 532 nm wavelength. This laser light is visible to humans, appearing green, and is produced by transmitting a 1064 nm wavelength through a nonlinear crystal. As the light passes through the crystal, it's wavelength is reduced by half. A YVO4 medium is normally used because the characteristics of the beam are well suited for intricate processing.
           
           High absorption rates in materials that do not react well with typical IR wavelengths and those that reflect IR light (such as gold and copper)
Intricate processing is possible due to a smaller beam spot than IR lasers
Transparent objects are typically not able to be processed
High peak power and limited heat transfer make 532 nm lasers ideal for micro machining and intricate designs
    '''
    # Chosen constants
    Lambda = 532 * 10**(-12) # meters (nanometers)
    # n = 1 # Refractive index in a vacuum (assume for space)

    '''
    FOR MIRRORS:
        https://www.edmundoptics.com/knowledge-center/application-notes/lasers/gaussian-beam-propagation/
        
    f = 1/3 * av_d # focal length of lens ---ARBITRARY---
    laser2lens = 1 # laser tube to lens ---ARBITRARY---
    
    Equations sourced from
    https://iopscience.iop.org/article/10.1088/1361-6404/aa57cb

    d1 = laser2lens
    d2 = 
    # _p or ' values are used to calculate new values when moving the lens
    #d1_p =
    #d2_p =
    
    A = 1 - d2/f
    B = d1 + d2*(1 - d2/f)
    C = -1/f
    D = 1 - d1/f
    
    #A_p = 1 - d2_p/f
    #B_p = d1_p + d2_p * (1 - d2_p/f)
    #C_p = -1/f
    #D_p = 1 - d1_p/f
    
    x = -(((A_p*B*D)-(A*B_p*D_p))/((A*A_p*D)+(A_p*B*C)-(A*A_p*D_p)-(A*B_p*C)))
    
    y = np.sqrt(-(x)**2 - x*(((A*C) + (B*C)) / (A*C)) - ((B*D)/(A*C)))
    
    w0 = np.sqrt((Lambda/np.pi) * (( (A*x + B)**2 + (A*y)**2) / y))
    
    #w0_p = np.sqrt((Lambda/np.pi) * (( (A_p*x + B_p)**2 + (A_p*y)**2) / y))
    
    z_R = 0.5 * k * w0**2
    z0 = np.pi * w0**2 / Lambda
    #z_R_p = 0.5 * k * w0_p**2
    
    w = w0 * np.sqrt(1 + (laser2lens/z_R)**2)
    '''
    
    # Calculate Rayleigh length needed for the average distance, wavelength, and target radius
    z_R = (radius**2 * np.pi / Lambda) - av_d**2
    
    # Calculate the beam waist needed for the calculated Rayleigh length
    w0 = np.sqrt(z_R * Lambda / np.pi)
    
    # Calculate the actual radius of the beam at the changing distance (near average distance)
    w = w0 * np.sqrt(1 + (curr_d / z_R)**2)
    
    # Calculate the intensity at the center of the beam
    I0 = 2 * P0 / (np.pi * w0**2)
    
    # Initialize variables prior to for loop
    I = []
    radius = []
    
    # Choose max radius to calculate intensity out to -- ARBITRARY --
    r_max = 3/2 * radius
    
    # Loop through a range from 1 to selected end radius
    for r in range(1,r_max):
        # Calculate intensity at that radius
        I.append(I0 * (w0/w)**2 * np.exp((-2 * r**2) / (w)**2)) # calculate intensity/irradiance distribution
        # Save radius to list
        radius.append(r)
        
    # Convert lists to numpy arrays
    I = np.array(I)
    radius = np.array(radius)
    return [radius, I]


    
    


    
