#%%

import numpy as np
import matplotlib.pyplot as plt

'''

Things needed:
- Concentraions of different species
- Focusing factor for different gas-phase species
- Size distribution of grains (MRN distribution)
- Charge distribution of grains 

'''

# Initializing constants for chemical network
k_B = 8.617e-5 # Boltzmann constant in eV/K
h = 6.626e-34 # Planck's constant in J s
lambda_R = 0.5 # Richardson constant
n_H2 = 10e14 # number density of H2 in cm^-3

# Density and dust-to-gas ratio
rho_gr = 3.3 # density of grains in g cm^-3
f_dg = 0.01 # dust-to-gas mass ratio

# Temperature
T = 1000 # temperature in K (Arbitrary value for testing)

#%%

# MRN grain size distribution
q = 2.5 # power law index for grain size distribution
mu = 2.34 # mean molecular weight in g/mol
m_H = 1.67e-24 # mass of hydrogen atom in g
x_H = 9.21e-1 # mass fraction of hydrogen in the gas
rho = mu * m_H * ((2-x_H)/x_H) * n_H2
a_min = 10e-5 # minimum grain size in cm
a_max = 10e-1 # maximum grain size in cm
A = ((f_dg*rho)/rho_gr) * (3/(4*np.pi)) * (4-q) * (1/a_min**(4-q) - 1/a_max**(4-q)) # normalization constant for MRN distribution

a = np.linspace(a_min, a_max, 100) # grain sizes in cm

def grain_size_distribution(a):
    return A * a**(-q)

# plt.figure(figsize=(10, 6))
# plt.plot(a, grain_size_distribution(a), label='Grain Size Distribution', color='blue')
# plt.xlabel('Grain Size (cm)')
# plt.ylabel('Distribution Function')
# plt.title('MRN Grain Size Distribution')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.show()

#%%

Z = np.array([]) # charges of the grains

# Rate coefficients
zeta = 7.6e-19 # ionization rate of H2 in s^-1
beta = 3e-9 # rate coefficient for charge transfer in cm^3 s^-1
alpha = 3e-6 # rate coefficient for dissociative recombination in T^-1/2 cm^3 s^-1
gamma = 3e-11 # rate coefficient for radiative recombination in T^-1/2 cm^3 s^-1
IP = 4.34  # ionization potential in eV (Taken from Desch & Turner 2015)
k_2 = (9.9e-9)*np.exp(-IP/k_B*T) # rate coefficient for collisional ionisation in T^1/2 cm^3 s^-1
k_minus_2 = 4.4e-24 # rate coefficient for 3-body recombination in T^-1 cm^6 s^-1

# Electrostatic potential, effective work function and fraction of alkali ions evaporated
E_a = 3.25 # activation energy in eV
W = 5 # work function in eV
e = 1.602e-19 # elementary charge in C
phi = -(Z*e)/a # electrostatic potential of the grain in V
W_eff = W - (e*phi) # effective work function in eV
g_plus = 1 # degeneracy factor for ions (Taken from Desch & Turner 2015)
g_0 = 2 # degeneracy factor for neutrals (Taken from Desch & Turner 2015)
f_plus = 1/(1+((g_0/g_plus)*np.exp(IP-W_eff/(k_B*T)))) # Fraction of alkali ions evaporated

# Initial concentrations of different species
n_e = 
n_alk_0 = 
n_alk_plus =
n_alk_cond = np.array([])
n_m_plus =
n_M_0 =
n_M_plus = 
n_gr = np.array([])

n_alk_0 + n_alk_plus + n_alk_cond = 3.04e-7 * n_H2

n_alk_plus = n_e

x_M = 3.67e-5
x_H = 9.21e-1
n_M_0 + n_M_plus = ((2*x_M)/x_H) * n_H2

# Mass of different gas-phase species (Used chatgpt for values)
m_e = 9.109e-31 # mass of electron in kg
m_alk_plus = 6.491e-26 # mass of alkali ion (K+) in kg
m_m_plus = 4.814e-26 # mass of molecular ion (HCO+) in kg
m_M_plus = 4.037e-26 # mass of metal ion (Mg+) in kg

# Focusing factor and sticking coefficient of different gas-phase species
J_e = 
J_alk_plus = 
J_m_plus =
J_M_plus =
s_electrons = 0.6 # sticking coefficient for electrons
s_ions = 1 # sticking coefficient for ions and neutrals

# Frequencies of different gas-phase species
nu_evap = 3.7e13 * np.exp(-E_a/(k_B*T)) # frequency of alk vibration resulting in evaporation in s^-1
nu_e = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_e))**0.5 * s_electrons * J_e
nu_alk_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_plus))**0.5 * s_ions * J_alk_plus
nu_m_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_m_plus))**0.5 * s_ions * J_m_plus
nu_M_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_M_plus))**0.5 * s_ions * J_M_plus

# Rates equations
# Gas-phase reactions
R_ct = beta * n_m_plus * n_M_0 # rate of charge transfer in cm^-3 s^-1
R_dissrec = alpha * n_m_plus * n_e # rate of dissociative recombination in T^-1/2 cm^-3 s^-1
R_gas_2rec_M_plus = gamma * n_M_plus * n_e # rate of radiative recombination for metal ions in T^-1/2 cm^-3 s^-1
R_gas_collion = k_2 * n_H2 * n_alk_0 # rate of collisional ionization in T^1/2 cm^-3 s^-1
R_gas_2rec_alk_plus = gamma * n_alk_plus * n_e # rate of radiative recombination for alkali ions in T^-1/2 cm^-3 s^-1
R_gas_3rec_alk_plus = k_minus_2 * n_alk_plus * n_e * n_H2 # rate of 3-body recombination for alkali ions in T^-1 cm^6 s^-1

# Dust-phase reactions (surface of grains)
R_e_ads = n_e * nu_e # rate of electron adsorption in cm^-3 s^-1
R_alk_plus_ads = n_alk_plus * nu_alk_plus # rate of alkali ion (K+) adsorption in cm^-3 s^-1
R_m_plus_ads = n_m_plus * nu_m_plus # rate of molecular ion (HCO+) adsorption in cm^-3 s^-1
R_M_plus_ads = n_M_plus * nu_M_plus # rate of metal ion (Mg+) adsorption in cm^-3 s^-1
R_alk_evap = n_alk_cond * nu_evap # rate of total alkali evaporation in cm^-3 s^-1
R_alk_plus_evap = f_plus * R_alk_evap # rate of alkali ion evaporation in cm^-3 s^-1
R_alk_0_evap = (1-f_plus) * R_alk_evap # rate of neutral alkali evaporation in cm^-3 s^-1
R_therm = n_gr * 4*np.pi * a**2 * lambda_R * ((4*np.pi*m_e*((k_B*T)**2))/(h**3)) * np.exp(-W_eff/(k_B*T)) # rate of thermionic emission of electrons in cm^-3 s^-1
