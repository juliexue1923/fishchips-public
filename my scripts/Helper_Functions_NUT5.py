
import numpy as np
from classy import Class

def Get_LCDM_Pks(h, omega_b, omega_cdm, tau_reio, A_s, n_s, P_k_max, k_per_decade, N_ur, N_ncdm, m_ncdm):
    LCDM = Class()
    
    # pass input parameters
    LCDM.set({'h':h,'omega_b':omega_b,'omega_cdm':omega_cdm,'tau_reio':tau_reio,'A_s':A_s,'n_s':n_s, \
                  'k_scalar_k_per_decade_for_pk':k_per_decade})
    LCDM.set({'N_ur':N_ur, 'N_ncdm':N_ncdm, 'm_ncdm':m_ncdm})
    LCDM.set({'output':'mPk','lensing':'no','P_k_max_1/Mpc':P_k_max})
    
    # run class
    LCDM.compute()
    
    # retrieve pk's at z=0
    h = LCDM.h()
    k_vec = np.logspace(-4,np.log10(P_k_max),100) # units of h/Mpc
    LCDM_Pk_vec = np.zeros(len(k_vec))  # units of (Mpc/h)**3
    for k in range(len(k_vec)):
        LCDM_Pk_vec[k]=LCDM.pk(k_vec[k]*h,0.) * h**3
    
    # output
    return k_vec, LCDM_Pk_vec


def Get_NU_Pks(h, omega_b, omega_cdm, tau_reio, A_s, n_s, P_k_max, k_per_decade, \
                                  f_nudm, u_ncdmdm_scale, N_ur, N_ncdm, m_ncdm, u_ncdmdm):
    NU = Class()
    
    # pass input parameters
    NU.set({'h':h, 'omega_b':omega_b, 'omega_cdm':omega_cdm, 'tau_reio':tau_reio, 'A_s':A_s, 'n_s':n_s, \
            'k_scalar_k_per_decade_for_pk':k_per_decade})
    NU.set({'N_ur':N_ur, 'N_ncdm':N_ncdm})
    NU.set({'output':'tCl,pCl,lCl,mPk,mTk','lensing':'yes','P_k_max_1/Mpc':P_k_max,'gauge': 'newtonian'})
    NU.set({'f_nudm':f_nudm, 'u_ncdmdm_scale':u_ncdmdm_scale})
    
    params1 = {}
    params1['m_ncdm']=m_ncdm
    NU.set(params1)  
    
    params2 = {}
    params2['u_ncdmdm']=u_ncdmdm
    NU.set(params2) 
    
    # run class
    NU.compute()
    
    # retrieve pk's at z=0
    h = NU.h()
    k_vec = np.logspace(-4,np.log10(P_k_max),100) # units of h/Mpc
    NU_Pk_vec = np.zeros(len(k_vec))  # units of (Mpc/h)**3
    for k in range(len(k_vec)):
        NU_Pk_vec[k]=NU.pk(k_vec[k]*h,0.) * h**3
    
    # output
    return k_vec, NU_Pk_vec


def Get_WDM_Tx(m_dmeff, omega_dmeff, k_vec, h):
    
    alpha =0.049*pow(omega_dmeff/0.25, 0.11) * pow(h/0.7,1.22) * pow(1.0 / m_dmeff, 1.11)
    Tf = pow(1 + pow(alpha * k_vec[:], 2 * 1.12), -5.0 / 1.12)
    WDM_Pk_vec = Tf * Tf
    
    # output
    return k_vec, WDM_Pk_vec


def Calc_K_Max(omega_b, omega_cdm, h):
    
    # define parameters
    M_sol = 1.989e30        # kg
    M_min = 5e8 * M_sol
    G = 6.6743e-11          # N.m^2/kg^2
    H = 67.27 / 3.086e19    # 1/s
    Omega_m = (omega_b + omega_cdm)/h/h
    
    # calculate k_max
    k_max = ( np.pi * 3.086e22 * (((H**2*Omega_m)/(2*G*M_min)) ** (1/3)) ) /h  # h/Mpc}
    
    #output
    return k_max

def Calc_MU_chi(sigma, m_chi):
    sigma_th = 0.665245854e-24
    MU = ((10**sigma)/sigma_th) * (m_chi/100)**-1
    
    return MU