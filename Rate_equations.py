#%%

import numpy as np

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

def R_therm(n_gr, a, lambda_R, T, W_eff, m_e, k_B, h): # rate of thermionic emission of electrons in cm^-3 s^-1
    coeff = n_gr * 4 * np.pi * a**2 * lambda_R
    density_of_states = (4 * np.pi * m_e * (k_B * T)**2) / (h**3)
    return coeff * density_of_states * np.exp(-W_eff / (k_B * T))