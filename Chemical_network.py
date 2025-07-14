#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

def calculate_nu_ff(Z,q,e):
    return Z * (e/q)

def calculate_tau(a,k_B,T,q):
    return a * k_B * T / q**2

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

def calculate_focusing_factor_derivative_dJ_dnu(nu, tau):
    # Not wrt Z!!
    if nu < -1:
        term1 = (-1 / tau) * (1 + np.sqrt(2 / (tau - 2 * nu)))
        term2 = (1 - nu / tau) * (np.sqrt(2 / ((tau - 2 * nu) ** 3.0)))
        return term1 + term2

    elif nu > 1:
        A = (4 * tau + 3 * nu)
        sqrt_A = np.sqrt(A)
        sqrt_nu = np.sqrt(nu)
        denom_inner = 1 + 1 / sqrt_nu
        exp_term = np.exp(-(nu / denom_inner) / tau)

        term1 = -3 * A ** (-1.5) * (1 + 1 / sqrt_A) * exp_term

        numerator = (0.5 * nu ** -0.5 + nu ** -0.5 + 1)
        denom = (1 + nu ** -0.5) ** 2
        term2 = (1 / tau) * (numerator / denom) * exp_term

        return term1 - (1 + 1 / sqrt_A) ** 2 * term2

    elif nu < 0:
        # Interpolation between nu = -1 and nu = 0
        left_val = (1 + np.sqrt(np.pi / (2 * tau)))
        right_val = ((1 - (-1) / tau) * (1 + np.sqrt(2 / (tau - 2 * (-1)))))
        return left_val - right_val

    else:
        # Interpolation between nu = 0 and nu = 1
        left_val = -(1 + np.sqrt(np.pi / (2 * tau)))
        sqrt_term = np.sqrt(4 * tau + 3 * 1)
        right_val = (1 + 1 / sqrt_term) ** 2 * np.exp(-(1. / (1 + 1 / np.sqrt(1.))) / tau)
        return left_val + right_val

def calc_W_eff(Z):
    phi = -(Z * e) / a  # electrostatic potential of the grain in V
    W_eff = W - (e * phi)  # effective work function in J
    return W_eff

def calc_f_plus(Z):
    phi = -(Z*e)/a # electrostatic potential of the grain in V
    W_eff = W - (e*phi) # effective work function in J
    g_plus = 1 # degeneracy factor for ions (Taken from Desch & Turner 2015)
    g_0 = 2 # degeneracy factor for neutrals (Taken from Desch & Turner 2015)
    f_plus = 1/(1+((g_0/g_plus)*np.exp(IP-W_eff/(k_B*T)))) # Fraction of alkali ions evaporated
    return f_plus

def calc_df_plus_dZ(Z):
    g_plus = 1
    g_0 = 2
    A = g_0 / g_plus
    B = 1 / (k_B * T)
    exponent = B * (IP - W - (Z * e**2) / a)
    exp_term = np.exp(exponent)
    numerator = A * e**2 * B * exp_term / a
    denominator = (1 + A * exp_term)**2
    return numerator / denominator


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

N_gr = 3
a = np.logspace(np.log10(a_min), np.log10(a_max), N_gr) # grain sizes in cm
n_gr = A * a**(-q)

# Initial heuristic charge distribution of grains
Z = np.full_like(a, -1)
# Z = -(a/1e-5)**0.1

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
nu_evap = 3.7e13 * np.exp(-E_a/(k_B*T)) # frequency of alk vibration resulting in evaporation in s^-1

# Mass of different gas-phase species (Used chatgpt for values)
m_e = 9.109e-28 # electron mass in g
m_alk_plus = 6.491e-23 # alkali ion (K+) mass in g
m_alk_0 = 6.491e-23 # neutral alkali (K) mass in g
m_m_plus = 4.814e-23 # molecular ion (HCO+) mass in g
m_M_plus = 4.037e-23 # metal ion (Mg+) mass in g

# Charge of different gas-phase species
q_e = -e # charge of electron in C
q_ion = e # charge of ion in C

# Initial concentrations of different species
n_alk_tot = 3.04e-7 * n_H2 # total concentration of alkali (K) in cm^-3
n_alk_plus = 1e-7 * n_H2 # initial concentration of alkali ion (K+) in cm^-3
n_e = 1e-7 * n_H2 # initial concentration of free electrons in cm^-3
n_m_plus = 1e-10 * n_H2 # initial concentration of molecular ion (HCO+) in cm^-3

x = np.concatenate((Z, np.array([n_alk_plus, n_e, n_m_plus])))

print("Initial concentrations:")
print(x)


#%%

# Modified Powell's Hybrid method
def calculate_F(x):

    Z = x[:N_gr] 
    n_alk_plus = x[N_gr]
    n_e = x[N_gr + 1]
    n_m_plus = x[N_gr + 2]
    
    F = np.zeros_like(x)

    nu_ff_e = calculate_nu_ff(Z, q_e, e) 
    nu_ff_ion = calculate_nu_ff(Z, q_ion, e)

    tau_e = calculate_tau(a, k_B, T, q_e)
    tau_ion = calculate_tau(a, k_B, T, q_ion)

    # Focusing factor and sticking coefficient of different gas-phase species
    J_e = np.zeros_like(a) # Focusing factor for electrons
    J_ion = np.zeros_like(a) # Focusing factor for ions

    for i in range(len(a)):
        J_e[i] = calculate_focusing_factor_J(nu_ff_e[i], tau_e[i]) # Focusing factor for electrons
        J_ion[i] = calculate_focusing_factor_J(nu_ff_ion[i], tau_ion[i]) # Focusing factor for ions

    J_neutral = np.ones_like(a) # Focusing factor for neutral alkali (K)
    s_electrons = 0.6 # sticking coefficient for electrons
    s_ions = 1 # sticking coefficient for ions and neutrals

    # Frequencies of different gas-phase species
    nu_e = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_e))**0.5 * s_electrons * J_e 
    nu_alk_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_plus))**0.5 * s_ions * J_ion
    nu_alk_0 = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_0))**0.5 * s_ions * J_neutral
    nu_m_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_m_plus))**0.5 * s_ions * J_ion
    nu_M_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_M_plus))**0.5 * s_ions * J_ion

    n_alk_0 = n_alk_tot - (1+(np.sum(nu_alk_plus)/nu_evap)*n_alk_plus)/(1+(np.sum(nu_alk_0))/nu_evap) # initial concentration of neutral alkali (K) in cm^-3
    n_alk_cond = (1/nu_evap) * (nu_alk_plus*n_alk_plus + nu_alk_0*n_alk_0) # initial concentration of condensed alkali (K) on grains in cm^-3
    n_M_plus = n_e - np.sum(Z*n_gr) - n_alk_plus - n_m_plus # initial concentration of metal ion (Mg+) in cm^-3

    W_eff = calc_W_eff(Z) # effective work function in J
    f_plus = calc_f_plus(Z) # fraction of alkali ions evaporated

    x_M = 3.67e-5
    x_H = 9.21e-1
    n_M_tot = (((2*x_M)/x_H) * n_H2)
    n_M_0 = n_M_tot - n_M_plus # initial concentration of neutral metal (Mg) in cm^-3

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
    R_therm = n_gr * 4*np.pi * a**2 * lambda_R * ((4*np.pi*m_e*((k_B*T)**2))/(h**3)) * np.exp(-W_eff/(k_B*T)) # rate of thermionic emission of electrons in cm^-3 s^-1

    for i in range(N_gr):
        F[i] = R_alk_plus_ads[i] + R_m_plus_ads[i] + R_M_plus_ads[i] - R_e_ads[i] + R_therm[i] - R_alk_plus_evap[i]

    F[N_gr] = - R_gas_3rec_alk_plus - R_gas_2rec_alk_plus + R_gas_collion + np.sum(R_alk_plus_evap) - np.sum(R_alk_plus_ads)

    F[N_gr + 1] = zeta * n_H2 - R_gas_2rec_M_plus - np.sum(R_e_ads) - R_dissrec - R_gas_2rec_alk_plus - R_gas_3rec_alk_plus + R_gas_collion + np.sum(R_therm)

    F[N_gr + 2] = zeta * n_H2 - R_ct - R_dissrec - np.sum(R_m_plus_ads)

    return F

def calculate_Jacobian(x):

    Z = x[:N_gr] 
    n_alk_plus = x[N_gr]
    n_e = x[N_gr + 1]
    n_m_plus = x[N_gr + 2]

    J = np.zeros((len(x), len(x)))

    nu_ff_e = calculate_nu_ff(Z, q_e, e) 
    nu_ff_ion = calculate_nu_ff(Z, q_ion, e)

    tau_e = calculate_tau(a, k_B, T, q_e)
    tau_ion = calculate_tau(a, k_B, T, q_ion)

    # Focusing factor for different gas-phase species
    J_e = np.zeros_like(a) # Focusing factor for electrons
    J_ion = np.zeros_like(a) # Focusing factor for ions

    for i in range(len(a)):
        J_e[i] = calculate_focusing_factor_J(nu_ff_e[i], tau_e[i]) # Focusing factor for electrons
        J_ion[i] = calculate_focusing_factor_J(nu_ff_ion[i], tau_ion[i]) # Focusing factor for ions

    # Derivative of focusing factor wrt Z
    dJ_e = np.zeros_like(a) # Focusing factor for electrons
    dJ_ion = np.zeros_like(a) # Focusing factor for ions

    for i in range(len(a)):
        dJ_e[i] = -calculate_focusing_factor_derivative_dJ_dnu(nu_ff_e[i], tau_e[i]) # Focusing factor for electrons
        dJ_ion[i] = calculate_focusing_factor_derivative_dJ_dnu(nu_ff_ion[i], tau_ion[i]) # Focusing factor for ions

    J_neutral = np.ones_like(a) # Focusing factor for neutral alkali (K)
    s_electrons = 0.6 # sticking coefficient for electrons
    s_ions = 1 # sticking coefficient for ions and neutrals

    # Frequencies for different gas-phase species
    nu_e = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_e))**0.5 * s_electrons * J_e 
    nu_alk_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_plus))**0.5 * s_ions * J_ion
    nu_alk_0 = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_0))**0.5 * s_ions * J_neutral
    nu_m_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_m_plus))**0.5 * s_ions * J_ion
    nu_M_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_M_plus))**0.5 * s_ions * J_ion

    n_alk_0 = n_alk_tot - (1+(np.sum(nu_alk_plus)/nu_evap)*n_alk_plus)/(1+(np.sum(nu_alk_0))/nu_evap) # initial concentration of neutral alkali (K) in cm^-3
    n_alk_cond = (1/nu_evap) * (nu_alk_plus*n_alk_plus + nu_alk_0*n_alk_0) # initial concentration of condensed alkali (K) on grains in cm^-3
    n_M_plus = n_e - np.sum(Z*n_gr) - n_alk_plus - n_m_plus # initial concentration of metal ion (Mg+) in cm^-3

    W_eff = calc_W_eff(Z) # effective work function in J
    f_plus = calc_f_plus(Z) # fraction of alkali ions evaporated

    # Derivatives of frequencies for different gas-phase species wrt Z
    dnu_e = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_e))**0.5 * s_electrons * dJ_e 
    dnu_alk_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_alk_plus))**0.5 * s_ions * dJ_ion
    dnu_m_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_m_plus))**0.5 * s_ions * dJ_ion
    dnu_M_plus = n_gr * np.pi * a**2 * ((8*k_B*T)/(np.pi*m_M_plus))**0.5 * s_ions * dJ_ion

    dn_alk_cond = (1/nu_evap) * (dnu_alk_plus*n_alk_plus) 
    df_plus_dZ = calc_df_plus_dZ(Z)

    x_M = 3.67e-5
    x_H = 9.21e-1
    n_M_tot = (((2*x_M)/x_H) * n_H2)
    n_M_0 = n_M_tot - n_M_plus # initial concentration of neutral metal (Mg) in cm^-3

    R_alk_evap = n_alk_cond * nu_evap # rate of total alkali evaporation in cm^-3 s^-1
    R_therm = n_gr * 4*np.pi * a**2 * lambda_R * ((4*np.pi*m_e*((k_B*T)**2))/(h**3)) * np.exp(-W_eff/(k_B*T)) # rate of thermionic emission of electrons in cm^-3 s^-1

    for i in range(N_gr):

        J[i, N_gr] = nu_alk_plus[i]
        J[i, N_gr+1] = -nu_e[i]
        J[i, N_gr+2] = nu_m_plus[i]

        J[N_gr, i] = f_plus[i]*dn_alk_cond[i]*nu_evap + df_plus_dZ[i]*R_alk_evap[i] - n_alk_plus*dnu_alk_plus[i]
        J[N_gr+1, i] = -n_e*dnu_e[i]
        J[N_gr+2, i] = -n_m_plus*dnu_m_plus[i]

        for j in range(N_gr):
            if i == j:
                J[i, j] = dnu_alk_plus[i]*n_alk_plus + dnu_m_plus[i]*n_m_plus + dnu_M_plus[i]*n_M_plus - dnu_e[i]*n_e + R_therm[i] - f_plus[i]*dn_alk_cond[i]*nu_evap - df_plus_dZ[i]*R_alk_evap[i]
    
    # n_alk_plus
    J[N_gr, N_gr] = -k_minus_2*n_H2*n_e - gamma*n_e
    J[N_gr+1, N_gr] = -k_minus_2*n_H2*n_e - gamma*n_e

    # n_e
    J[N_gr, N_gr+1] = -k_minus_2*n_H2*n_alk_plus - gamma*n_alk_plus
    J[N_gr+1, N_gr+1] = -gamma*n_M_plus - np.sum(nu_e) - alpha*n_m_plus - k_minus_2*n_H2*n_alk_plus - gamma*n_alk_plus
    J[N_gr+2, N_gr+1] = -alpha*n_m_plus

    # n_m_plus
    J[N_gr+1, N_gr+2] = -alpha*n_e
    J[N_gr+2, N_gr+2] = -beta*n_M_0 - alpha*n_e - np.sum(nu_m_plus)

    return J


#%%

max_iter = 10
tolerance = 1e-7

for iteration in range(max_iter):
    F = calculate_F(x)
    print(f"Iteration {iteration}, function values:")
    print(F)
    
    J = calculate_Jacobian(x)
    print(f"Iteration {iteration}, Jacobian:")
    print(J)
    
    det = np.linalg.det(J)
    print(f"Determinant: {det}")
    
    if abs(det) < 1e-12:
        print("Warning: Matrix is close to singular and might not be invertible.")
        break 

    J_inv = np.linalg.inv(J)
    print("Inverse of Jacobian:")
    print(J_inv)

    I = J_inv @ J
    print("Identity matrix:")
    print(I)

    delta_x = -J_inv @ F
    print("Change in x:")
    print(delta_x)
    
    delta = 1000
    alpha_const = 1e-12
    beta_const = 1e-15
    if abs(delta_x).max() >= delta:
        print("Adjusting step size due to large delta_x.")
        delta_x = -alpha_const * (J_inv @ F) - 2 * beta_const * (J.T @ F)
    
    print("Current concentrations:")
    print(x)
    
    x = x + delta_x
    print("Updated concentrations:")
    print(x)

    if abs(F).max() < tolerance:
        print(f"Convergence achieved after {iteration + 1} iterations.")
        break
else:
    print("Maximum iterations reached without convergence.")
