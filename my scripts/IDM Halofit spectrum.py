import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#matplotlib.rcParams['text.usetex'] = True
#plt.rcParams['font.family'] = 'Times New Roman'

from numpy.fft import fft, ifft , rfft, irfft , fftfreq
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt
from classy import Class
from scipy.optimize import fsolve
from scipy.special import gamma
from scipy.special import hyp2f1
import sys,os
from time import time
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy import special
from scipy.special import factorial

#Starting CLASS

z_pk = 0
common_settings = {# fixed LambdaCDM parameters
                   'ln10^{10}A_s':3.0347,
                   'n_s':9.7914e-01,
                   'tau_reio':5.2431e-02,
                   'omega_b':0.022706,
                   'h':0.68262,
#                   '100*theta_s':1.0431,
                   'YHe':2.4798e-01,
#                   'N_eff':3.046,
                   'N_ur':2.0328,
                   'N_ncdm':1,
#                   'N_ncdm':0,
                   'm_ncdm':0.06,
                   # other output and precision parameters
                   'P_k_max_h/Mpc':100,
                   'z_pk':z_pk,
                   'output':'mPk,tCl,pCl,lCl',
                   'lensing':'yes'}
#                   'l_max_scalars':10000.}

M = Class()
M.set(common_settings)
#compute linear LCDM only
M.set({ 'non linear':'no',
        'omega_cdm': 1.1955e-01,
        'omega_dmeff': 0.,
      })
M.compute()

M1 = Class()
M1.set(common_settings)
#compute linear dmeff
M1.set({'non linear':'no',
        'omega_cdm':1e-15,
        'omega_dmeff': 1.1955e-01,
        'npow_dmeff': 0,
        'sigma_dmeff': 1e-27,
        'm_dmeff': 1e-2,
        'dmeff_target': 'baryons',
        'Vrel_dmeff': 0
      })
M1.compute()

M3 = Class()
M3.set(common_settings)
#compute Halofit dmeff
M3.set({'non linear':'Halofit',
        'omega_cdm':1e-15,
        'omega_dmeff': 1.1955e-01,
        'npow_dmeff': 0,
        'sigma_dmeff': 1e-27,
        'm_dmeff': 1e-2,
        'dmeff_target': 'baryons',
        'Vrel_dmeff': 0,
      })
M3.compute()

#Extracting and plotting spectra

bg_LCDM = M.get_background()
th_LCDM = M.get_thermodynamics()
cl_LCDM = M.lensed_cl()

bg_dmeff = M1.get_background()
th_dmeff = M1.get_thermodynamics()
cl_dmeff = M1.lensed_cl()

bg_dmeff_Halofit = M3.get_background()
th_dmeff_Halofit = M3.get_thermodynamics()
cl_dmeff_Halofit = M3.lensed_cl()

h = M.h()
h1 = M1.h()
print(h)
print(h1)

k = np.linspace(log(0.0001),log(100),1000)
k = np.exp(k)

twopi = 2.*math.pi
khvec = k*h

f,ax_list = plt.subplots(1,3,figsize=(9,3),constrained_layout=True)
ax = ax_list.ravel()

ax[0].set_xlabel('$\ell$')
#ax[0].set_yscale('log')
#ax[0].set_ylabel(r'$(C_\ell^{TT} - C_\ell^{TT, LCDM})/C_\ell^{TT, LCDM}$ [%]')
ax[0].set_ylabel(r'$(C_\ell^{TT, IDM} - C_\ell^{TT, LCDM})/C_\ell^{TT, LCDM} \ [\%]$')
ax[0].set_xlim([50,2500])
ax[0].set_ylim([0,0.2])

ax[1].set_xlabel('$\ell$')
#ax[1].set_yscale('log')
#ax[1].set_ylabel(r'$(C_\ell^{EE} - C_\ell^{EE, LCDM})/C_\ell^{EE, LCDM}$ [%]')
ax[1].set_ylabel(r'$(C_\ell^{EE, IDM} - C_\ell^{EE, LCDM})/C_\ell^{EE, LCDM} \ [\%]$')
ax[1].set_xlim([50,2500])
ax[1].set_ylim([0,0.2])

ax[2].set_xlabel('$k$')
#ax[2].set_xlabel('$k$')
#ax[2].set_yscale('log')
ax[2].set_xscale('log')
#ax[2].set_ylabel(r'$(C_\ell^{PP} - C_\ell^{PP, LCDM})/C_\ell^{PP, LCDM}$ [%]')
#ax[2].set_ylabel(r'$C_\ell^{PP, IDM}/C_\ell^{PP, LCDM}$')
ax[2].set_ylabel('$P/P_{CDM}$')
ax[2].set_xlim([1e-3,50])
#ax[2].set_xlim([1,2500])
#ax[2].set_ylim([0.8,1.05])

#ax[2].set_xscale('log')
#ax[2].set_xlabel('$k$')
#ax[2].set_yscale('log')
#ax[2].set_ylabel(r'$(P - P^{LCDM})/P^{LCDM}$ [%]')
#ax[2].set_ylabel('$P(k)_{IDM}/P(k)_{LCDM}$')
#ax[2].set_xlim([1e-3,0.5])
#ax[2].set_ylim([-20, 10])

power_spectrum_LCDM = np.linspace(log(0.0001),log(50),1000)
power_spectrum_dmeff = np.linspace(log(0.0001),log(50),1000)
power_spectrum_Halofit = np.linspace(log(0.0001),log(50),1000)

for i in range(len(k)):
    power_spectrum_LCDM[i] = M.pk_lin(k[i]*h,z_pk)*h**3
    power_spectrum_dmeff[i] = M1.pk_lin(k[i]*h,z_pk)*h**3
    power_spectrum_Halofit[i] = M3.pk(k[i]*h,z_pk)*h**3

np.savetxt('Halofit_spectrum.txt',np.column_stack([k,power_spectrum_Halofit]),fmt=['%1.10f','%1.40f'])

#Cleaning up

M.struct_cleanup()
M.empty()
M1.struct_cleanup()
M1.empty()
M3.struct_cleanup()
M3.empty()
