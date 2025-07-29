#%%

import numpy as np
import scipy as sp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200)
np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.2e}"})

#%%

def calculate_nu_ff(Z,q,e):
    return Z * (e/q)

def calculate_tau(a,k_B,T):
    return (a * k_B * T) / (e**2)

def calculate_focusing_factor_J(nu, tau):
    if nu < -1:
        # Equation (3.4) from D+S (1987)
        factor = (1 - nu / tau)
        sqrt_term = np.sqrt(2 / (tau - 2 * nu))
        return factor * (1 + sqrt_term)
    elif nu > 1:
        # Equation (3.5)
        sqrt_term = 1 / np.sqrt(4 * tau + 3 * nu)
        pre_factor = (1 + sqrt_term) ** 2
        exp_arg = -(nu / (1 + 1 / np.sqrt(nu))) / tau
        return pre_factor * np.exp(exp_arg)
    elif nu < 0:
        # Interpolation for -1 < nu < 0
        left_term = (1 + np.sqrt(np.pi / (2 * tau))) * (1 + nu)
        right_term = (1 + np.sqrt(2 / (tau + 2))) * (1 + 1 / tau) * (-nu)
        return left_term - right_term
    elif nu == 0:
        return 1 + np.sqrt(np.pi / (2 * tau))
    else:
        # Interpolation for 0 ≤ nu ≤ 1
        left_term = (1 + np.sqrt(np.pi / (2 * tau))) * (1 - nu)
        pre_factor = (1 + 1 / np.sqrt(4 * tau + 3)) ** 2
        exp_arg = -(1 / (1 + 1 / np.sqrt(1))) / tau
        right_term = pre_factor * np.exp(exp_arg) * nu
        return left_term + right_term

def calculate_focusing_factor_derivative_dJ_dnu(nu, tau):
    if nu < -1:
        # Derivative of Equation (3.4)
        sqrt_term = np.sqrt(2 / (tau - 2 * nu))
        dsqrt_dnu = (2 * sqrt_term) / ((tau - 2 * nu) ** 1.5)
        dJ_dnu = (-1 / tau) * (1 + sqrt_term) + (1 - nu / tau) * dsqrt_dnu
        return dJ_dnu
    elif nu > 1:
        # Derivative of Equation (3.5)
        A = 4 * tau + 3 * nu
        sqrt_A = np.sqrt(A)
        sqrt_nu = np.sqrt(nu)
        denom_inner = 1 + 1 / sqrt_nu
        exp_arg = -(nu / denom_inner) / tau
        exp_term = np.exp(exp_arg)
        d_prefactor_dnu = -3 * A ** (-1.5) * (1 + 1 / sqrt_A)
        pre_factor = (1 + 1 / sqrt_A) ** 2
        d_exp_arg_dnu = (0.5 / sqrt_nu + 1) / (denom_inner ** 2 * tau)
        d_exp_term_dnu = -exp_term * d_exp_arg_dnu
        return d_prefactor_dnu * exp_term + pre_factor * d_exp_term_dnu
    elif nu < 0:
        # Linear interpolation derivative in (-1, 0)
        left_slope = (1 + np.sqrt(np.pi / (2 * tau)))
        right_slope = (1 + np.sqrt(2 / (tau + 2))) * (1 + 1 / tau)
        return left_slope - right_slope
    else:
        # Linear interpolation derivative in (0, 1)
        left_slope = -(1 + np.sqrt(np.pi / (2 * tau)))
        sqrt_term = np.sqrt(4 * tau + 3)
        pre_factor = (1 + 1 / sqrt_term) ** 2
        exp_arg = -(1 / (1 + 1 / np.sqrt(1))) / tau
        right_slope = pre_factor * np.exp(exp_arg)
        return left_slope + right_slope

# def calculate_focusing_factor_J(nu, tau):
#     if nu < 0:
#         return 1-(nu/tau)
#     else:
#         return np.exp(-nu/tau)
    
# def calculate_focusing_factor_derivative_dJ_dnu(nu, tau):
#     if nu < 0:
#         return -1/tau
#     else:
#         return -np.exp(-nu/tau)/tau

def calc_W_eff(Z):
    phi = -(Z*e)/a  # electrostatic potential of the grain in V
    W_eff = W-(e*phi)  # effective work function in J
    return W_eff

def calc_f_plus(Z):
    phi = -(Z*e)/a # electrostatic potential of the grain in V
    W_eff = W-(e*phi) # effective work function in J
    g_plus = 1 # degeneracy factor for ions (Taken from Desch & Turner 2015)
    g_0 = 2 # degeneracy factor for neutrals (Taken from Desch & Turner 2015)
    f_plus = 1/(1+((g_0/g_plus)*np.exp((IP-W_eff)/(k_B*T)))) # Fraction of alkali ions evaporated
    return f_plus

def calc_df_plus_dZ(f_plus,tau):
    df_plus_dz = (1/tau)*(1-f_plus)*f_plus
    return df_plus_dz

def calc_dn_alk_0_dZ(n_alk_plus, dnu_alk_plus, nu_alk_0, nu_evap):
    B = np.sum(nu_alk_0)
    D = 1 + (B / nu_evap)
    dn_alk_0_dZ = - (n_alk_plus / (D * nu_evap)) * dnu_alk_plus
    return dn_alk_0_dZ



#%%


# Initializing constants for chemical network
k_B = 1.38e-16 # Boltzmann constant in erg K^-1
h = 6.626e-27 # Planck's constant in erg s
lambda_R = 0.5 # Richardson constant
n_H2 = 1e14 # number density of H2 in cm^-3
e = 4.803e-10 # elementary charge in esu (erg^0.5 cm^1/2 s^-1)
eV = 1.602e-12 # 1 eV in erg

# Density and dust-to-gas ratio
rho_gr = 3.3 # density of grains in g cm^-3
f_dg = 0.01 # dust-to-gas mass ratio

# Temperature
T = 1000 # temperature in K (Arbitrary value for testing)

# MRN grain size distribution
q = 3.5 # power law index for grain size distribution
mu = 2.34 # mean molecular weight in amu
m_H = 1.67e-24 # mass of hydrogen atom in g
x_H = 9.21e-1 # mass fraction of hydrogen in the gas
rho = mu * m_H * ((2-x_H)/x_H) * n_H2
a_min = 1e-5 # minimum grain size in cm
a_max = 1e-1 # maximum grain size in cm
A = ((f_dg*rho)/rho_gr) * (3/(4*np.pi)) * (4-q) * ((1/(a_min**(4-q)))-(1/(a_max**(4-q)))) # normalization constant for MRN distribution

# print("rho")
# print(f"{rho:.2e} g cm^-3")
# print()

# print("A:")
# print(f"{A:.2e}")
# print()

N_gr = 4
a_edges = np.logspace(-5, -1, N_gr+1) # grain sizes in cm
a = np.sqrt(a_edges[:-1] * a_edges[1:])  # geometric mean

n_gr = np.zeros(N_gr)
for i in range(N_gr):
    a1 = a_edges[i]
    a2 = a_edges[i + 1]
    n_gr[i] = (A / (1 - q)) * (a2**(1 - q) - a1**(1 - q))

n_gr_total = np.sum(n_gr)  # total number density of grains

# print("a_edges:")
# print([f"{a:.2e}" for a in a_edges])
# print()
# print("a_centers:")
# print([f"{i:.2e}" for i in a])
# print()
# print("n_gr:")
# print([f"{ngr:.2e}" for ngr in n_gr])
# print()
# print("n_gr_total:")
# print(f"{n_gr_total:.2e}")
# print()

# plt.figure(figsize=(10, 6))
# plt.step(a_edges[:], np.append(n_gr, 0), where='post', label='Grain Size Distribution')
# for value in a:
#     plt.axvline(x=value, color='red', linestyle='--', linewidth=1.5, label=f'a = {value}')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Grain Size (cm)')
# plt.ylabel('Number Density per Bin (cm$^{-3}$)')
# plt.title('Binned MRN Grain Size Distribution')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# Initial heuristic charge distribution of grains
Z_edges = np.logspace(2, 6, N_gr+1) # charge distribution in esu (erg^0.5 cm^1/2 s^-1)
Z = -np.sqrt(Z_edges[:-1] * Z_edges[1:])

# print("Z_edges:")
# print([f"{z:.2e}" for z in Z_edges])
# print()  # geometric mean of charge distribution
# print("Z:")
# print([f"{z:.2e}" for z in Z])
# print()

# Rate coefficients
zeta = 7.6e-19 # ionization rate of H2 in s^-1
# zeta = 1.4e-22 # ionization rate of H2 in s^-1 (Desch & Turner 2015)
beta = 3e-9 # rate coefficient for charge transfer in cm^3 s^-1
alpha = (3e-6)*(T**-0.5) # rate coefficient for dissociative recombination in T^-1/2 cm^3 s^-1
gamma = (3e-11)*(T**-0.5) # rate coefficient for radiative recombination in T^-1/2 cm^3 s^-1
IP = 4.34 * eV  # ionization potential in erg
k_2 = (9.9e-9)*(T**0.5)*np.exp(-IP/(k_B*T)) # rate coefficient for collisional ionisation in T^1/2 cm^3 s^-1
k_minus_2 = 4.4e-24*(T**-1) # rate coefficient for 3-body recombination in T^-1 cm^6 s^-1

# Electrostatic potential, effective work function and fraction of alkali ions evaporated
E_a = 3.25 * eV # activation energy in erg
W = 5 * eV # work function in erg
nu_evap = 3.7e13 * np.exp(-E_a/(k_B*T)) # frequency of alk vibration resulting in evaporation in s^-1

# print()
# print("E_a/(k_B*T)")
# print(E_a/(k_B*T))
# print("")

# print("np.exp(-E_a/(k_B*T)")
# print(np.exp(-E_a/(k_B*T)))
# print("")

# Mass of different gas-phase species (Used chatgpt for values)
m_e = 9.109e-28 # electron mass in g
m_alk_plus = 6.491e-23 # alkali ion (K+) mass in g
m_alk_0 = 6.491e-23 # neutral alkali (K) mass in g
m_m_plus = 4.814e-23 # molecular ion (HCO+) mass in g
m_M_plus = 4.037e-23 # metal ion (Mg+) mass in g

# Charge and sticking coefficients of different gas-phase species
q_e = -e # charge of electron in esu (erg^0.5 cm^1/2 s^-1)
q_ion = e # charge of ion in esu (erg^0.5 cm^1/2 s^-1)
s_electrons = 0.6 # sticking coefficient for electrons
s_ions = 1 # sticking coefficient for ions and neutrals

# Initial concentrations of different species
# x_alk = 3.04e-7 # mass fraction of alkali (K) in the gas
# x_alk = 9.87e-8
# x_H = 9.21e-1
# n_alk_tot = (((2*x_alk)/x_H) * n_H2)

n_alk_tot = 3.04e-7 * n_H2 # total concentration of alkali (K) in cm^-3 (Desch & Turner 2015)
n_alk_plus = 1e-10 * n_H2 # initial concentration of alkali ion (K+) in cm^-3
n_e = 1e-10 * n_H2 # initial concentration of free electrons in cm^-3
n_m_plus = 1e-19 * n_H2 # initial concentration of molecular ion (HCO+) in cm^-3
n_M_plus = n_e - np.sum(Z*n_gr) - n_alk_plus - n_m_plus # initial concentration of metal ion (Mg+) in cm^-3

# print("Initial concentrations:\n")
# print("Z:")
# print([f"{z:.2e}" for z in Z])
# print()
# print("n_gr:")
# print([f"{ngr:.2e}" for ngr in n_gr])
# print()
# print("n_gr/n_H2:")
# print([f"{ngr/n_H2:.2e}" for ngr in n_gr])
# print()
# print("Z*n_gr:")
# Zn_gr = Z * n_gr
# print([f"{prod/n_H2:.2e}" for prod in Zn_gr])
# print()
# print("np.sum(Z*n_gr):")
# print(f"{np.sum(Zn_gr/n_H2):.2e}")
# print()
# print("n_alk_plus:")
# print(f"{n_alk_plus/n_H2:.2e}")
# print()
# print("n_m_plus:")
# print(f"{n_m_plus/n_H2:.2e}")
# print()
# print("n_M_plus:")
# print(f"{n_M_plus/n_H2:.2e}")
# print()
# print("n_e:")
# print(f"{n_e/n_H2:.2e}")
# print()

x = np.concatenate((Z, np.array([n_alk_plus, n_e, n_m_plus])))

# print("Initial concentrations:")
# print(x)


#%%

# Function to calculate the chemical network
def calculate_F(x):

    Z = x[:N_gr] 
    n_alk_plus = x[N_gr]
    n_e = x[N_gr + 1]
    n_m_plus = x[N_gr + 2]
    
    F = np.zeros_like(x)

    nu_ff_e = calculate_nu_ff(Z, q_e, e) 
    nu_ff_ion = calculate_nu_ff(Z, q_ion, e)

    # print("nu_ff_e:")
    # print(nu_ff_e)
    # print("")
    # print("nu_ff_ion:")
    # print(nu_ff_ion)
    # print("")

    tau = calculate_tau(a, k_B, T)

    # print("tau:")
    # print(tau)
    # print("")

    # Focusing factor and sticking coefficient of different gas-phase species
    J_e = np.zeros_like(a) # Focusing factor for electrons
    J_ion = np.zeros_like(a) # Focusing factor for ions
    J_neutral = np.zeros_like(a) # Focusing factor for neutral alkali (K)

    for i in range(len(a)):
        J_e[i] = calculate_focusing_factor_J(nu_ff_e[i], tau[i]) # Focusing factor for electrons
        J_ion[i] = calculate_focusing_factor_J(nu_ff_ion[i], tau[i]) # Focusing factor for ions
        J_neutral[i] = calculate_focusing_factor_J(0, tau[i]) # Focusing factor for neutrals

    # print("J_e:")   
    # print(J_e)
    # print("")
    # print("J_ion:")
    # print(J_ion)
    # print("")
    # print("J_neutral:")
    # print(J_neutral)
    # print("")

    # Frequencies of different gas-phase species
    nu_e = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_e))**0.5 * s_electrons * J_e 
    nu_alk_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_plus))**0.5 * s_ions * J_ion
    nu_alk_0 = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_0))**0.5 * s_ions * J_neutral
    nu_m_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_m_plus))**0.5 * s_ions * J_ion
    nu_M_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_M_plus))**0.5 * s_ions * J_ion

    # print("nu_e:")
    # print(nu_e)
    # print("")
    # print("nu_alk_plus:")
    # print(nu_alk_plus)
    # print("")
    # print("nu_alk_0:")
    # print(nu_alk_0)
    # print("")
    # print("nu_m_plus:")
    # print(nu_m_plus)
    # print("")
    # print("nu_M_plus:")
    # print(nu_M_plus)
    # print("")

    n_alk_0 = (n_alk_tot - ((1+(np.sum(nu_alk_plus)/nu_evap))*n_alk_plus))/(1+(np.sum(nu_alk_0)/nu_evap)) # initial concentration of neutral alkali (K) in cm^-3
    n_alk_cond = (1/nu_evap) * ((nu_alk_plus*n_alk_plus) + (nu_alk_0*n_alk_0)) # initial concentration of condensed alkali (K) on grains in cm^-3
    n_M_plus = n_e - np.sum(Z*n_gr) - n_alk_plus - n_m_plus # initial concentration of metal ion (Mg+) in cm^-3

    # print("n_alk_tot:")
    # print(f"{n_alk_tot:.2e}")
    # print("")
    # print("minus term:")
    # print(f"{(1+(np.sum(nu_alk_plus)/nu_evap))*n_alk_plus:.2e}")
    # print("")

    # print("np.sum(nu_alk_plus)")
    # print(np.sum(nu_alk_plus))
    # print("")

    # print("nu_evap")
    # print(f"{nu_evap:.2e}")
    # print("")

    # print("n_alk_plus")
    # print(f"{n_alk_plus:.2e}")
    # print("")

    # print("n_alk_0:")
    # print(f"{n_alk_0:.2e}")
    # print()
    # print("n_alk_cond:")
    # print(f"{n_alk_cond}")
    # print()
    # print("n_M_plus:")
    # print(f"{n_M_plus:.2e}")
    # print()
    
    W_eff = calc_W_eff(Z) # effective work function in erg

    # print("W_eff (eV):")
    # print(W_eff/eV)
    # print("")
    # print("W_eff/(k_B*T):")
    # print(W_eff/(k_B*T))
    # print("")

    f_plus = calc_f_plus(Z) # fraction of alkali ions evaporated

    x_M = 3.67e-5
    x_H = 9.21e-1
    n_M_tot = (((2*x_M)/x_H) * n_H2)
    n_M_0 = n_M_tot - n_M_plus # initial concentration of neutral metal (Mg) in cm^-3

    # print("n_M_tot:")
    # print(f"{n_M_tot:.2e}")
    # print()
    # print("n_M_0:")
    # print(f"{n_M_0:.2e}")
    # print()

    # Rates equations
    # Gas-phase reactions
    R_ct = beta * n_m_plus * n_M_0 # rate of charge transfer in cm^-3 s^-1
    R_dissrec = alpha * n_m_plus * n_e # rate of dissociative recombination in T^-1/2 cm^-3 s^-1
    R_gas_2rec_M_plus = gamma * n_M_plus * n_e # rate of radiative recombination for metal ions in T^-1/2 cm^-3 s^-1
    R_gas_collion = k_2 * n_H2 * n_alk_0 # rate of collisional ionization in T^1/2 cm^-3 s^-1
    R_gas_2rec_alk_plus = gamma * n_alk_plus * n_e # rate of radiative recombination for alkali ions in T^-1/2 cm^-3 s^-1
    R_gas_3rec_alk_plus = k_minus_2 * n_alk_plus * n_e * n_H2 # rate of 3-body recombination for alkali ions in T^-1 cm^6 s^-1

    # print("Rates of gas-phase reactions:")
    # print()
    # print("R_ct:") 
    # print(f"{R_ct:.2e}")
    # print("R_dissrec:")
    # print(f"{R_dissrec:.2e}")
    # print("R_gas_2rec_M_plus:")
    # print(f"{R_gas_2rec_M_plus:.2e}")
    # print("R_gas_collion:")
    # print(f"{R_gas_collion:.2e}")
    # print("R_gas_2rec_alk_plus:")
    # print(f"{R_gas_2rec_alk_plus:.2e}")
    # print("R_gas_3rec_alk_plus:")
    # print(f"{R_gas_3rec_alk_plus:.2e}")
    # print()

    # Dust-phase reactions (surface of grains)
    R_e_ads = n_e * nu_e # rate of electron adsorption in cm^-3 s^-1
    R_alk_plus_ads = n_alk_plus * nu_alk_plus # rate of alkali ion (K+) adsorption in cm^-3 s^-1
    R_m_plus_ads = n_m_plus * nu_m_plus # rate of molecular ion (HCO+) adsorption in cm^-3 s^-1
    R_M_plus_ads = n_M_plus * nu_M_plus # rate of metal ion (Mg+) adsorption in cm^-3 s^-1
    R_alk_evap = n_alk_cond * nu_evap # rate of total alkali evaporation in cm^-3 s^-1
    R_alk_plus_evap = f_plus * R_alk_evap # rate of alkali ion evaporation in cm^-3 s^-1
    R_therm = n_gr * 4*np.pi * a**2 * lambda_R * ((4*np.pi*m_e*((k_B*T)**2))/(h**3)) * np.exp(-W_eff/(k_B*T)) # rate of thermionic emission of electrons in cm^-3 s^-1

    # print("Rates of dust-phase reactions:")
    # print()
    # print("R_e_ads:")
    # print(R_e_ads)
    # print("R_alk_plus_ads:")
    # print(R_alk_plus_ads)
    # print("R_m_plus_ads:")
    # print(R_m_plus_ads)
    # print("R_M_plus_ads:")
    # print(R_M_plus_ads)
    # print("R_alk_evap:")
    # print(R_alk_evap)
    # print("R_alk_plus_evap:")
    # print(R_alk_plus_evap)
    # print("R_therm:")
    # print(R_therm)
    # print()

    for i in range(N_gr):
        F[i] = R_alk_plus_ads[i] + R_m_plus_ads[i] + R_M_plus_ads[i] - R_e_ads[i] + R_therm[i] - R_alk_plus_evap[i]

    F[N_gr] = - R_gas_3rec_alk_plus - R_gas_2rec_alk_plus + R_gas_collion + np.sum(R_alk_plus_evap) - np.sum(R_alk_plus_ads)

    F[N_gr + 1] = (zeta * n_H2) - R_gas_2rec_M_plus - np.sum(R_e_ads) - R_dissrec - R_gas_2rec_alk_plus - R_gas_3rec_alk_plus + R_gas_collion + np.sum(R_therm)

    F[N_gr + 2] = (zeta * n_H2) - R_ct - R_dissrec - np.sum(R_m_plus_ads)

    return F


def calculate_Jacobian(x):

    Z = x[:N_gr] 
    n_alk_plus = x[N_gr]
    n_e = x[N_gr + 1]
    n_m_plus = x[N_gr + 2]

    J = np.zeros((len(x), len(x)))

    nu_ff_e = calculate_nu_ff(Z, q_e, e) 
    nu_ff_ion = calculate_nu_ff(Z, q_ion, e)

    # print("nu_ff_e:")
    # print(nu_ff_e)
    # print("")
    # print("nu_ff_ion:")
    # print(nu_ff_ion)
    # print("")

    tau = calculate_tau(a, k_B, T)

    # print("tau:")
    # print(tau)
    # print("")

    # Focusing factor for different gas-phase species
    J_e = np.zeros_like(a) # Focusing factor for electrons
    J_ion = np.zeros_like(a) # Focusing factor for ions
    J_neutral = np.zeros_like(a) # Focusing factor for neutral alkali (K)

    for i in range(len(a)):
        J_e[i] = calculate_focusing_factor_J(nu_ff_e[i], tau[i]) # Focusing factor for electrons
        J_ion[i] = calculate_focusing_factor_J(nu_ff_ion[i], tau[i]) # Focusing factor for ions
        J_neutral[i] = calculate_focusing_factor_J(0, tau[i]) # Focusing factor for neutrals

    # print("nu_ff_e/tau:")
    # print((nu_ff_e/tau))
    # print("")
    # print("J_e:")
    # print(J_e)
    # print("")
    # print("J_ion:")
    # print(J_ion)
    # print("")

    # Derivative of focusing factor wrt Z
    dJ_e_dZ = np.zeros_like(a) # Focusing factor for electrons
    dJ_ion_dZ = np.zeros_like(a) # Focusing factor for ions

    for i in range(len(a)):
        dJ_e_dZ[i] = -calculate_focusing_factor_derivative_dJ_dnu(nu_ff_e[i], tau[i]) # Focusing factor for electrons
        dJ_ion_dZ[i] = calculate_focusing_factor_derivative_dJ_dnu(nu_ff_ion[i], tau[i]) # Focusing factor for ions

    # print("dJ_e:")
    # print(dJ_e)
    # print("")
    # print("dJ_ion:")
    # print(dJ_ion)
    # print("")

    # Frequencies for different gas-phase species
    nu_e = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_e))**0.5 * s_electrons * J_e 
    nu_alk_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_plus))**0.5 * s_ions * J_ion
    nu_alk_0 = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_0))**0.5 * s_ions * J_neutral
    nu_m_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_m_plus))**0.5 * s_ions * J_ion
    nu_M_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_M_plus))**0.5 * s_ions * J_ion

    n_alk_0 = (n_alk_tot - ((1+(np.sum(nu_alk_plus)/nu_evap))*n_alk_plus))/(1+(np.sum(nu_alk_0)/nu_evap)) # initial concentration of neutral alkali (K) in cm^-3
    n_alk_cond = (1/nu_evap) * (nu_alk_plus*n_alk_plus + nu_alk_0*n_alk_0) # initial concentration of condensed alkali (K) on grains in cm^-3
    n_M_plus = n_e - np.sum(Z*n_gr) - n_alk_plus - n_m_plus # initial concentration of metal ion (Mg+) in cm^-3

    W_eff = calc_W_eff(Z) # effective work function in erg
    f_plus = calc_f_plus(Z) # fraction of alkali ions evaporated

    # Derivatives of frequencies for different gas-phase species wrt Z
    dnu_e_dZ = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_e))**0.5 * s_electrons * dJ_e_dZ 
    dnu_alk_plus_dZ = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_plus))**0.5 * s_ions * dJ_ion_dZ
    dnu_m_plus_dZ = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_m_plus))**0.5 * s_ions * dJ_ion_dZ
    dnu_M_plus_dZ = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_M_plus))**0.5 * s_ions * dJ_ion_dZ

    dn_alk_0_dZ = calc_dn_alk_0_dZ(n_alk_plus, dnu_alk_plus_dZ, nu_alk_0, nu_evap) # derivative of n_alk_0 wrt Z
    dn_alk_0_dn_alk_plus = -(np.sum(nu_alk_plus))/(nu_evap+np.sum(nu_alk_0)) # derivative of n_alk_0 wrt n_alk_plus

    dn_alk_cond_dZ = (1/nu_evap) * (dnu_alk_plus_dZ*n_alk_plus) # derivative of n_alk_cond wrt Z
    dn_alk_cond_dn_alk_plus = (1/nu_evap) * (nu_alk_plus + nu_alk_0*dn_alk_0_dn_alk_plus) # derivative of n_alk_cond wrt n_alk_plus

    dn_M_0_dZ = n_gr # derivative of n_M_0 wrt Z
    dn_M_0_dn_alk_plus = 1 # derivative of n_M_0 wrt n_alk_plus
    dn_M_0_dn_e = -1 # derivative of n_M_0 wrt n_e
    dn_M_0_dn_m_plus = 1 # derivative of n_M_0 wrt n_m_plus

    dn_M_plus_dZ = -n_gr # derivative of n_M_plus wrt Z
    dn_M_plus_dn_alk_plus = -1 # derivative of n_M_plus wrt n_alk_plus
    dn_M_plus_dn_e = 1 # derivative of n_M_plus wrt n_e
    dn_M_plus_dn_m_plus = -1 # derivative of n_M_plus wrt n_m_plus

    df_plus_dZ = calc_df_plus_dZ(f_plus,tau)
    dW_eff_dZ = (e**2)/a

    x_M = 3.67e-5
    x_H = 9.21e-1
    n_M_tot = (((2*x_M)/x_H) * n_H2)
    n_M_0 = n_M_tot - n_M_plus # initial concentration of neutral metal (Mg) in cm^-3

    R_alk_evap = n_alk_cond * nu_evap # rate of total alkali evaporation in cm^-3 s^-1
    R_therm = n_gr * 4*np.pi * a**2 * lambda_R * ((4*np.pi*m_e*((k_B*T)**2))/(h**3)) * np.exp(-W_eff/(k_B*T)) # rate of thermionic emission of electrons in cm^-3 s^-1
    dR_therm_dZ = -(1/(k_B*T)) * R_therm * dW_eff_dZ

    for i in range(N_gr):

        J[i, N_gr] = nu_alk_plus[i] + nu_M_plus[i]*dn_M_plus_dn_alk_plus - f_plus[i]*dn_alk_cond_dn_alk_plus[i]*nu_evap
        J[i, N_gr+1] = -nu_e[i] + nu_M_plus[i]*dn_M_plus_dn_e
        J[i, N_gr+2] = nu_m_plus[i] + nu_M_plus[i]*dn_M_plus_dn_m_plus

        J[N_gr, i] = f_plus[i]*dn_alk_cond_dZ[i]*nu_evap + df_plus_dZ[i]*R_alk_evap[i] - n_alk_plus*dnu_alk_plus_dZ[i] + k_2*n_H2*dn_alk_0_dZ[i]
        J[N_gr+1, i] = -gamma*n_e*dn_M_plus_dZ[i] - n_e*dnu_e_dZ[i] + dR_therm_dZ[i] + k_2*n_H2*dn_alk_0_dZ[i]
        J[N_gr+2, i] = -beta*n_m_plus*dn_M_0_dZ[i] - n_m_plus*dnu_m_plus_dZ[i]

        for j in range(N_gr):
            if i == j:
                J[i, j] = dnu_alk_plus_dZ[i]*n_alk_plus + dnu_m_plus_dZ[i]*n_m_plus + dnu_M_plus_dZ[i]*n_M_plus + dn_M_plus_dZ[i]*nu_M_plus[i] - dnu_e_dZ[i]*n_e + dR_therm_dZ[i] - f_plus[i]*dn_alk_cond_dZ[i]*nu_evap - df_plus_dZ[i]*R_alk_evap[i]
    
    # n_alk_plus
    J[N_gr, N_gr] = -k_minus_2*n_H2*n_e - gamma*n_e - np.sum(nu_alk_plus) + k_2*n_H2*dn_alk_0_dn_alk_plus + np.sum(f_plus*dn_alk_cond_dn_alk_plus*nu_evap)
    J[N_gr+1, N_gr] = -k_minus_2*n_H2*n_e - gamma*n_e + k_2*n_H2*dn_alk_0_dn_alk_plus - gamma*n_e*dn_M_plus_dn_alk_plus
    J[N_gr+2, N_gr] = -beta*n_m_plus*dn_M_0_dn_alk_plus

    # n_e
    J[N_gr, N_gr+1] = -k_minus_2*n_H2*n_alk_plus - gamma*n_alk_plus
    J[N_gr+1, N_gr+1] = -gamma*n_M_plus - gamma*n_e*dn_M_plus_dn_e - np.sum(nu_e) - alpha*n_m_plus - k_minus_2*n_H2*n_alk_plus - gamma*n_alk_plus
    J[N_gr+2, N_gr+1] = -beta*n_m_plus*dn_M_0_dn_e - alpha*n_m_plus

    # n_m_plus
    J[N_gr+1, N_gr+2] = -alpha*n_e - gamma*n_e*dn_M_plus_dn_m_plus
    J[N_gr+2, N_gr+2] = -beta*n_M_0 - beta*n_m_plus*dn_M_0_dn_m_plus - alpha*n_e - np.sum(nu_m_plus)

    return J



#%%

def solve_single_root(x_current, x_prev, i, tol=1e-7):

    def func_to_solve(x_i):
        # Create x_local with x_j^{(k)} for j < i, x_j^{(k-1)} for j > i
        x_local = x_current.copy()
        x_local[i+1:] = x_prev[i+1:]  # Use previous iteration for j > i
        x_local[i] = x_i
        F = calculate_F(x_local)
        return F[i]
    
    def fprime_func(x_i):
        x_local = x_current.copy()
        x_local[i+1:] = x_prev[i+1:] 
        x_local[i] = x_i
        J = calculate_Jacobian(x_local)
        diag = J[i, i]
        return diag

    result = root_scalar(func_to_solve, method='newton', x0=x_current[i], fprime=fprime_func, xtol=tol)

    if not result.converged:
        print('----------------')
        print(f"Warning: root_scalar did not converge for x[{i}]")
        print(f"[i={i}] Newton failed:")
        print(f"  x0 = {x_current[i]}")
        print(f"  F_i = {func_to_solve(x_current[i])}")
        print(f"  J_ii = {fprime_func(x_current[i])}")
        print('-- -- -- -- -- -- -- --')
        print(f"[i={i}] Newton failed at x0 = {x_current[i]}, trying secant.")
        print('----------------')
        try:
            result = root_scalar(func_to_solve, method='secant', x0=x_current[i], x1=x_current[i]*0.999, xtol=tol)
        except Exception as e:
            print()
            print(f"Secant method failed too: {e}")
            print()
            return x_current[i]
    
    return result.root

residual_history = []
component_history = [[] for _ in range(len(x))] 
omegas = []

def nSOR(x, iteration=1000, omega_start=1e-1, omega_end=1.2, tol_F = 1e-6):

    N = len(x)
    omega = omega_start
    prev_residual = np.inf

    for iter in range(iteration):

        x_old = x.copy()

        for i in range(N):
            x_hat = solve_single_root(x, x_old, i)
            x[i] = x_old[i] + (omega * (x_hat - x_old[i]))

        residual_norm = sp.linalg.norm(calculate_F(x))
        omegas.append(omega)

        if iter % 100 == 0 or iter == iteration - 1:
            print()
            print(f"Iteration {iter}")
            print()
            J = calculate_Jacobian(x)
            jac_svd = sp.linalg.svd(J, compute_uv=False)
            print("Initial Jacobian SVD:")
            print(jac_svd)
            print()
            print("Concentrations:")
            print(x)
            print()

        for j in range(N):
            component_history[j].append(x[j])

        residual_history.append(residual_norm)

        if residual_norm < tol_F:
            print(f"Converged: residual = {residual_norm:.2e}")
            return x
        
        # Determine update step based on current omega range
        if omega < 1e-4:
            delta = 1e-5
            drop = 5e-6
        elif omega < 1e-3:
            delta = 1e-4
            drop = 5e-5
        elif omega < 1e-2:
            delta = 1e-3
            drop = 5e-4
        elif omega < 1e-1:
            delta = 1e-2
            drop = 5e-3
        elif omega < 1.0:
            delta = 1e-1
            drop = 5e-2
        else:
            delta = 1e-1
            drop = 5e-2

        # Update omega based on residual improvement
        if residual_norm < prev_residual:
            omega += delta
        else:
            omega = max(omega-drop, 1e-12)  # avoid omega = 0

        # Clamp to maximum allowed value
        omega = min(omega, omega_end)

        # Update previous residual
        prev_residual = residual_norm

        # print(f"Iteration {iter + 1}, x values:")
        # print(x)

    print("Warning: nSOR did not converge within iteration limits.")
    return x

#%%

print()
print("Initial function:")
function = calculate_F(x)
print(function)
print()
print("Initial Jacobian:")
jacobian = calculate_Jacobian(x)
print(jacobian)
print()
jac_svd = sp.linalg.svd(jacobian, compute_uv=False)
print("Initial Jacobian SVD:")
print(jac_svd)
print()
print("Initial concentrations:")
print(x)
print()

x_1 = x.copy()

nSOR_ = nSOR(x)

print()
print("Initial concentrations:")
print(x_1)
print()
print("nSOR result:")
print(nSOR_)
print()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(residual_history, label='Residual Norm')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('||F(x)||')
plt.title('Residual Norm Convergence')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(omegas, label='Relaxation Factor (ω)')
plt.xlabel('Iteration')
plt.ylabel('ω')
plt.title('Relaxation Factor Over Iterations')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 5))

# for i, comp_history in enumerate(component_history):
#     plt.plot(comp_history, label=f"x[{i}]")
#     plt.xlabel('Iteration')
#     plt.ylabel('Concentration')
#     plt.title('Concentration of Species Over Iterations')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#%%

x = nSOR_
max_iter = 1000
tolerance = 1e-7
iter = 0
residual_history_2 = []

for iteration in range(max_iter):

    print("--------------------------------------------------")
    print(f"Iteration {iteration + 1}/{max_iter}")
    print("")

    F = calculate_F(x)
    print(f"Iteration {iteration + 1}, F values:")
    print(F)
    print("")
    
    J = calculate_Jacobian(x)
    print(f"Iteration {iteration + 1}, Jacobian:")
    print(J)
    print("")
    
    det = sp.linalg.det(J)
    # print(f"Determinant: {det}")
    
    # if abs(det) < 1e-12:
    #     print("Warning: Matrix is close to singular and might not be invertible.")
    #     break 

    J_inv = sp.linalg.inv(J)
    delta_x = -J_inv @ F
    print("Change in x:")
    print(delta_x)
    print("")
    
    delta = 100
    alpha_const = 0.1
    beta_const = 1e-14
    norm_delta_x = sp.linalg.norm(delta_x)
    if norm_delta_x >= delta:
        print("Adjusting step size due to large delta_x.")
        print("")
        delta_x = (alpha_const*sp.linalg.solve(J, -F)) - (2*beta_const*(J.T @ F))
        print("Beta constant bit")
        print(-(2*beta_const*(J.T @ F)))
        print("")
        print("Adjusted change in x:")
        print(delta_x)
        print("")

    
    print("Current concentrations:")
    print(x)
    print("")
    
    x += delta_x
    print("Updated concentrations:")
    print(x)
    print("")

    norm_F = sp.linalg.norm(F)
    residual_history_2.append(norm_F)
    if norm_F < tolerance:
        print(f"Convergence achieved after {iteration + 1} iterations.")
        print("")
        break
    
    if iteration == max_iter - 1:
        print("Maximum iterations reached without convergence.")
        print("")


# Plotting the residual history
plt.figure(figsize=(10, 5))
plt.plot(residual_history_2, label='Residual Norm', linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('||F(x)||')
plt.title('Residual Norm Convergence')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
