

def AntennaParameters(D, wavelength):
    from cmath import log10
    from math import pi
    #Inputs: Diameter (meters) Wavelength (Hz)
    #Outputs: Gain in dB, area in m^2 (trivial calculation but sped up with function implementation)
    #Method: use a form to find gain that can be sampled via
    #diameter and wavelength

    k = 1.38e-23 #boltzman's constant
    area = pi*(D/2)**2
    beamwidth = 70 * (wavelength/D) 
    area_eff = area*.55 # = effective area simplified formula
    gain = 10*log10((6.9*area_eff)/(wavelength**2))
    return [gain.real, area, area_eff, beamwidth]

