#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

def calculate_focusing_factor_J(nu, tau): # Taken from M.Williams C code
    if nu < -1:
        # Equation (3.4 D+S 87)
        return (1 - nu / tau) * (1 + np.sqrt(2 / (tau - 2 * nu)))
    elif nu > 1:
        # Equation (3.5)
        return (1 + 1 / np.sqrt(4 * tau + 3 * nu)) ** 2 * np.exp(-(nu / (1 + 1 / np.sqrt(nu))) / tau)
    elif nu < 0:
        # Interpolation for -1 < nu < 0
        term1 = (1 + np.sqrt(np.pi / (2 * tau))) * (1 + nu)
        term2 = ((1 - (-1) / tau) * (1 + np.sqrt(2 / (tau - 2 * (-1))))) * (-nu)
        return term1 - term2
    else:
        # Interpolation for 0 ≤ nu ≤ 1
        term1 = (1 + np.sqrt(np.pi / (2 * tau))) * (1 - nu)
        term2 = (1 + 1 / np.sqrt(4 * tau + 3)) ** 2 * np.exp(-(1 / (1 + 1 / np.sqrt(1))) / tau) * nu
        return term1 + term2

# Rates equations
# Gas-phase reactions
def R_ct(n_m_plus, n_M_0, beta): # rate of charge transfer in cm^-3 s^-1
    return beta * n_m_plus * n_M_0

def R_dissrec(n_m_plus, n_e, alpha): # rate of dissociative recombination in T^-1/2 cm^-3 s^-1
    return alpha * n_m_plus * n_e

def R_gas_2rec_M_plus(n_M_plus, n_e, gamma): # rate of radiative recombination for metal ions in T^-1/2 cm^-3 s^-1
    return gamma * n_M_plus * n_e

def R_gas_collion(n_H2, n_alk_0, k_2): # rate of collisional ionization in T^1/2 cm^-3 s^-1
    return k_2 * n_H2 * n_alk_0

def R_gas_2rec_alk_plus(n_alk_plus, n_e, gamma): # rate of radiative recombination for alkali ions in T^-1/2 cm^-3 s^-1
    return gamma * n_alk_plus * n_e

def R_gas_3rec_alk_plus(n_alk_plus, n_e, n_H2, k_minus_2): # rate of 3-body recombination for alkali ions in T^-1 cm^6 s^-1
    return k_minus_2 * n_alk_plus * n_e * n_H2

# Dust-phase reactions (surface of grains)
def R_e_ads(n_e, nu_e): # rate of electron adsorption in cm^-3 s^-1
    return n_e * nu_e

def R_alk_plus_ads(n_alk_plus, nu_alk_plus): # rate of alkali ion (K+) adsorption in cm^-3 s^-1
    return n_alk_plus * nu_alk_plus

def R_m_plus_ads(n_m_plus, nu_m_plus): # rate of molecular ion (HCO+) adsorption in cm^-3 s^-1
    return n_m_plus * nu_m_plus

def R_M_plus_ads(n_M_plus, nu_M_plus): # rate of metal ion (Mg+) adsorption in cm^-3 s^-1
    return n_M_plus * nu_M_plus

def R_alk_evap(n_alk_cond, nu_evap): # rate of total alkali evaporation in cm^-3 s^-1
    return n_alk_cond * nu_evap

def R_alk_plus_evap(n_alk_cond, nu_evap, f_plus): # rate of alkali ion evaporation in cm^-3 s^-1
    return f_plus * R_alk_evap(n_alk_cond, nu_evap)

def R_alk_0_evap(n_alk_cond, nu_evap, f_plus): # rate of neutral alkali evaporation in cm^-3 s^-1
    return (1 - f_plus) * R_alk_evap(n_alk_cond, nu_evap)

def R_therm(n_gr, a, lambda_R, T, W_eff, m_e, k_B, h): # rate of thermionic emission of electrons in cm^-3 s^-1
    coeff = n_gr * 4 * np.pi * a**2 * lambda_R
    density_of_states = (4 * np.pi * m_e * (k_B * T)**2) / (h**3)
    return coeff * density_of_states * np.exp(-W_eff / (k_B * T))


#%%


# Initializing constants for chemical network
k_B = 1.38e-23 # Boltzmann constant in J/K
h = 6.626e-34 # Planck's constant in Js
lambda_R = 0.5 # Richardson constant
n_H2 = 10e14 # number density of H2 in cm^-3
e = 1.602e-19 # elementary charge in C

# Density and dust-to-gas ratio
rho_gr = 3.3 # density of grains in g cm^-3
f_dg = 0.01 # dust-to-gas mass ratio

# Temperature
T = 1000 # temperature in K (Arbitrary value for testing)

# MRN grain size distribution
q = 2.5 # power law index for grain size distribution
mu = 2.34 # mean molecular weight in g/mol
m_H = 1.67e-24 # mass of hydrogen atom in g
x_H = 9.21e-1 # mass fraction of hydrogen in the gas
rho = mu * m_H * ((2-x_H)/x_H) * n_H2
a_min = 1e-5 # minimum grain size in cm
a_max = 1e-1 # maximum grain size in cm
A = ((f_dg*rho)/rho_gr) * (3/(4*np.pi)) * (4-q) * (1/a_min**(4-q) - 1/a_max**(4-q)) # normalization constant for MRN distribution
a = np.logspace(np.log10(a_min), np.log10(a_max), 100) # grain sizes in cm
n_gr = A * a**(-q)

plt.figure(figsize=(10, 6))
plt.plot(a, n_gr, label='Grain Size Distribution')
plt.xlabel('Grain Size (cm)')
plt.ylabel('Number Density (cm$^{-3}$)')
plt.xscale('log')
plt.yscale('log')
plt.title('MRN Grain Size Distribution (log scale)')
plt.grid(True)
plt.legend()
plt.show()

# Initial heuristic charge distribution of grains
Z = np.full_like(a, -1e-6)
# Z = -(a/1e-5)**0.1

plt.figure(figsize=(10, 6))
plt.plot(a, Z, label='Initial Charge Distribution')
plt.xlabel('Grain Size (cm)')
plt.ylabel('Charge (C)')
plt.title('Initial Heuristic Charge Distribution of Grains')
plt.grid(True)
plt.legend()
plt.show()

# Rate coefficients
zeta = 7.6e-19 # ionization rate of H2 in s^-1
beta = 3e-9 # rate coefficient for charge transfer in cm^3 s^-1
alpha = 3e-6 # rate coefficient for dissociative recombination in T^-1/2 cm^3 s^-1
gamma = 3e-11 # rate coefficient for radiative recombination in T^-1/2 cm^3 s^-1
IP = 4.34 * e  # ionization potential in J (Taken from Desch & Turner 2015)
k_2 = (9.9e-9)*np.exp(-IP/k_B*T) # rate coefficient for collisional ionisation in T^1/2 cm^3 s^-1
k_minus_2 = 4.4e-24 # rate coefficient for 3-body recombination in T^-1 cm^6 s^-1

# Electrostatic potential, effective work function and fraction of alkali ions evaporated
E_a = 3.25 * e # activation energy in J
W = 5 * e # work function in J
phi = -(Z*e)/a # electrostatic potential of the grain in V
W_eff = W - (e*phi) # effective work function in J
g_plus = 1 # degeneracy factor for ions (Taken from Desch & Turner 2015)
g_0 = 2 # degeneracy factor for neutrals (Taken from Desch & Turner 2015)
f_plus = 1/(1+((g_0/g_plus)*np.exp(IP-W_eff/(k_B*T)))) # Fraction of alkali ions evaporated

# Mass of different gas-phase species (Used chatgpt for values)
m_e = 9.109e-28 # electron mass in g
m_alk_plus = 6.491e-23 # alkali ion (K+) mass in g
m_alk_0 = 6.491e-23 # neutral alkali (K) mass in g
m_m_plus = 4.814e-23 # molecular ion (HCO+) mass in g
m_M_plus = 4.037e-23 # metal ion (Mg+) mass in g

# Charge of different gas-phase species
q_e = -e # charge of electron in C
q_alk_plus = e # charge of alkali ion (K+) in C
q_m_plus = e # charge of molecular ion (HCO+) in C
q_M_plus = e # charge of metal ion (Mg+) in C

def calculate_nu_ff(Z,q,e):
    return Z * (e/q)

def calculate_tau(a,k_B,T,q):
    return a * k_B * T / q**2

nu_ff_e = calculate_nu_ff(Z, q_e, e) 
nu_ff_alk_plus = calculate_nu_ff(Z, q_alk_plus, e)
nu_ff_m_plus = calculate_nu_ff(Z, q_m_plus, e)
nu_ff_M_plus = calculate_nu_ff(Z, q_M_plus, e)

tau_e = calculate_tau(a, k_B, T, q_e)
tau_alk_plus = calculate_tau(a, k_B, T, q_alk_plus)
tau_m_plus = calculate_tau(a, k_B, T, q_m_plus)
tau_M_plus = calculate_tau(a, k_B, T, q_M_plus)

# Focusing factor and sticking coefficient of different gas-phase species
J_e = np.zeros_like(a) # Focusing factor for electrons
J_alk_plus = np.zeros_like(a) # Focusing factor for alkali ions (K+)
J_m_plus = np.zeros_like(a) # Focusing factor for molecular ions (HCO+)
J_M_plus = np.zeros_like(a) # Focusing factor for metal ions (Mg+)

for i in range(len(a)):
    J_e[i] = calculate_focusing_factor_J(nu_ff_e[i], tau_e[i]) # Focusing factor for electrons
    J_alk_plus[i] = calculate_focusing_factor_J(nu_ff_alk_plus[i], tau_alk_plus[i]) # Focusing factor for alkali ions (K+)
    J_m_plus[i] = calculate_focusing_factor_J(nu_ff_m_plus[i], tau_m_plus[i]) # Focusing factor for molecular ions (HCO+)
    J_M_plus[i] = calculate_focusing_factor_J(nu_ff_M_plus[i], tau_M_plus[i]) # Focusing factor for metal ions (Mg+)

J_alk_0 = 1
s_electrons = 0.6 # sticking coefficient for electrons
s_ions = 1 # sticking coefficient for ions and neutrals

# Frequencies of different gas-phase species
nu_evap = 3.7e13 * np.exp(-E_a/(k_B*T)) # frequency of alk vibration resulting in evaporation in s^-1
nu_e = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_e))**0.5 * s_electrons * J_e 
nu_alk_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_plus))**0.5 * s_ions * J_alk_plus 
nu_alk_0 = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_0))**0.5 * s_ions * J_alk_0
nu_m_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_m_plus))**0.5 * s_ions * J_m_plus
nu_M_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_M_plus))**0.5 * s_ions * J_M_plus

# Initial concentrations of different species
n_alk_tot = 3.04e-7 * n_H2 # total concentration of alkali (K) in cm^-3
n_alk_plus = 1e-7 * n_H2 # initial concentration of alkali ion (K+) in cm^-3
n_e = 1e-7 * n_H2 # initial concentration of free electrons in cm^-3
n_m_plus = 1e-10 * n_H2 # initial concentration of molecular ion (HCO+) in cm^-3

n_alk_0 = n_alk_tot - (1+(np.sum(nu_alk_plus)/nu_evap)*n_alk_plus)/(1+(np.sum(nu_alk_0))/nu_evap) # initial concentration of neutral alkali (K) in cm^-3
n_alk_cond = (1/nu_evap) * (nu_alk_plus*n_alk_plus + nu_alk_0*n_alk_0) # initial concentration of condensed alkali (K) on grains in cm^-3

n_M_plus = n_e - np.sum(Z*n_gr) - n_alk_plus - n_m_plus # initial concentration of metal ion (Mg+) in cm^-3

x_M = 3.67e-5
x_H = 9.21e-1
n_M_tot = (((2*x_M)/x_H) * n_H2)
n_M_0 = n_M_tot - n_M_plus # initial concentration of neutral metal (Mg) in cm^-3


#%%





#%%


# # Rates equations
# # Gas-phase reactions
# R_ct = beta * n_m_plus * n_M_0 # rate of charge transfer in cm^-3 s^-1
# R_dissrec = alpha * n_m_plus * n_e # rate of dissociative recombination in T^-1/2 cm^-3 s^-1
# R_gas_2rec_M_plus = gamma * n_M_plus * n_e # rate of radiative recombination for metal ions in T^-1/2 cm^-3 s^-1
# R_gas_collion = k_2 * n_H2 * n_alk_0 # rate of collisional ionization in T^1/2 cm^-3 s^-1
# R_gas_2rec_alk_plus = gamma * n_alk_plus * n_e # rate of radiative recombination for alkali ions in T^-1/2 cm^-3 s^-1
# R_gas_3rec_alk_plus = k_minus_2 * n_alk_plus * n_e * n_H2 # rate of 3-body recombination for alkali ions in T^-1 cm^6 s^-1

# # Dust-phase reactions (surface of grains)
# R_e_ads = n_e * nu_e # rate of electron adsorption in cm^-3 s^-1
# R_alk_plus_ads = n_alk_plus * nu_alk_plus # rate of alkali ion (K+) adsorption in cm^-3 s^-1
# R_m_plus_ads = n_m_plus * nu_m_plus # rate of molecular ion (HCO+) adsorption in cm^-3 s^-1
# R_M_plus_ads = n_M_plus * nu_M_plus # rate of metal ion (Mg+) adsorption in cm^-3 s^-1
# R_alk_evap = n_alk_cond * nu_evap # rate of total alkali evaporation in cm^-3 s^-1
# R_alk_plus_evap = f_plus * R_alk_evap # rate of alkali ion evaporation in cm^-3 s^-1
# R_alk_0_evap = (1-f_plus) * R_alk_evap # rate of neutral alkali evaporation in cm^-3 s^-1
# R_therm = n_gr * 4*np.pi * a**2 * lambda_R * ((4*np.pi*m_e*((k_B*T)**2))/(h**3)) * np.exp(-W_eff/(k_B*T)) # rate of thermionic emission of electrons in cm^-3 s^-1


#%%
