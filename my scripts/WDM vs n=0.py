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
#                    'P_k_max_1/Mpc':100.0,
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

M2 = Class()
M2.set(common_settings)
#compute EFT dmeff
M2.set({'non linear':'PT',
        'omega_cdm':1e-15,
        'omega_dmeff': 1.1955e-01,
        'npow_dmeff': 0,
        'sigma_dmeff': 1e-27,
        'm_dmeff': 1e-2,
        'dmeff_target': 'baryons',
        'Vrel_dmeff': 0,
        'IR resummation':'No',
        'Bias tracers':'Yes',
        'cb':'No',
        'RSD':'Yes',
        'AP':'Yes',
        'Omfid':'0.31',
      })
M2.compute()

#Extracting and plotting spectra

bg_LCDM = M.get_background()
th_LCDM = M.get_thermodynamics()
cl_LCDM = M.lensed_cl()

bg_dmeff = M1.get_background()
th_dmeff = M1.get_thermodynamics()
cl_dmeff = M1.lensed_cl()

bg_dmeff_EFT = M2.get_background()
th_dmeff_EFT = M2.get_thermodynamics()
cl_dmeff_EFT = M2.lensed_cl()

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
ax[2].set_ylim([0,1])

#ax[2].set_xscale('log')
#ax[2].set_xlabel('$k$')
#ax[2].set_yscale('log')
#ax[2].set_ylabel(r'$(P - P^{LCDM})/P^{LCDM}$ [%]')
#ax[2].set_ylabel('$P(k)_{IDM}/P(k)_{LCDM}$')
#ax[2].set_xlim([1e-3,0.5])
#ax[2].set_ylim([-20, 10])

power_spectrum_LCDM = np.linspace(log(0.0001),log(50),1000)
power_spectrum_dmeff = np.linspace(log(0.0001),log(50),1000)
Halofit_data = np.genfromtxt('/Users/adamhe/Research/Repositories/class_public-EFT-dmeff/Halofit_spectrum.txt')
power_spectrum_halofit = Halofit_data[:,1]

M2.initialize_output(khvec, z_pk, len(khvec))
power_spectrum_EFT_dmeff = M2.pk_mm_real(1)

for i in range(len(k)):
    power_spectrum_LCDM[i] = M.pk_lin(k[i]*h,z_pk)*h**3
    power_spectrum_dmeff[i] = M1.pk_lin(k[i]*h,z_pk)*h**3

l = np.array(cl_LCDM['ell'])
l = np.delete(l, 0)
l = np.delete(l, 0)

tt_LCDM = np.array(cl_LCDM['tt'])
tt_LCDM = np.delete(tt_LCDM, 0)
tt_LCDM = np.delete(tt_LCDM, 0)

ee_LCDM = np.array(cl_LCDM['ee'])
ee_LCDM = np.delete(ee_LCDM, 0)
ee_LCDM = np.delete(ee_LCDM, 0)

pp_LCDM = np.array(cl_LCDM['pp'])
pp_LCDM = np.delete(pp_LCDM, 0)
pp_LCDM = np.delete(pp_LCDM, 0)

tt_dmeff = np.array(cl_dmeff['tt'])
tt_dmeff = np.delete(tt_dmeff, 0)
tt_dmeff = np.delete(tt_dmeff, 0)

ee_dmeff = np.array(cl_dmeff['ee'])
ee_dmeff = np.delete(ee_dmeff, 0)
ee_dmeff = np.delete(ee_dmeff, 0)

pp_dmeff = np.array(cl_dmeff['pp'])
pp_dmeff = np.delete(pp_dmeff, 0)
pp_dmeff = np.delete(pp_dmeff, 0)

#WDM

omega_dmeff = 1.1955e-01
#m_dmeff = 0.5
m_dmeff = 3.1

alpha =0.049*pow(omega_dmeff/0.25, 0.11) * pow(h/0.7,1.22) * pow(1.0 / m_dmeff, 1.11)
Tf = pow(1 + pow(alpha * k, 2 * 1.12), -5.0 / 1.12)
WDM_Pk_vec = Tf * Tf

#Plotting spectra

#ax[0].plot(l, (tt_dmeff-tt_LCDM)/tt_LCDM * 100, color = 'green', label=r'$IDM$')
#ax[0].plot(l, (tt_dmeff_EFT-tt_dmeff)/tt_dmeff * 100, color = 'purple', label=r'$IDM + EFT$')
#ax[0].plot(l, (tt_dmeff_EFT-tt_LCDM_EFT)/tt_LCDM_EFT * 100, color = 'orange', label=r'$IDM + EFT$')
#ax[0].plot(l, tt_dmeff/tt_LCDM, color = 'purple', label=r'$IDM/LCDM$')
#ax[0].scatter(ACT_error_TT[:,0], (ACT_error_TT[:,1] - ACT_error_TT[:,2] - ACT_error_TT[:,0]*(ACT_error_TT[:,0]+1)*tt_LCDM_function(ACT_error_TT[:,0])/2/math.pi*1e7)/(ACT_error_TT[:,0]*(ACT_error_TT[:,0]+1)*tt_LCDM_function(ACT_error_TT[:,0])/2/math.pi*1e7))
#ax[0].scatter(ACT_error_TT[:,0], (ACT_error_TT[:,1] + ACT_error_TT[:,2] - ACT_error_TT[:,0]*(ACT_error_TT[:,0]+1)*tt_LCDM_function(ACT_error_TT[:,0])/2/math.pi*1e7)/(ACT_error_TT[:,0]*(ACT_error_TT[:,0]+1)*tt_LCDM_function(ACT_error_TT[:,0])/2/math.pi*1e7))

#ax[0].scatter(ACT_error_TT[:,0], (ACT_error_TT[:,1]/ACT_error_TT[:,0]/(ACT_error_TT[:,0]+1)*2*math.pi/(Tcmb**2) - ACT_error_TT[:,2]/ACT_error_TT[:,0]/(ACT_error_TT[:,0]+1)*2*math.pi/(Tcmb**2) - tt_LCDM_function(ACT_error_TT[:,0]))/tt_LCDM_function(ACT_error_TT[:,0]))
#ax[0].scatter(ACT_error_TT[:,0], (ACT_error_TT[:,1]/ACT_error_TT[:,0]/(ACT_error_TT[:,0]+1)*2*math.pi/(Tcmb**2) + ACT_error_TT[:,2]/ACT_error_TT[:,0]/(ACT_error_TT[:,0]+1)*2*math.pi/(Tcmb**2) - tt_LCDM_function(ACT_error_TT[:,0]))/tt_LCDM_function(ACT_error_TT[:,0]))

#ax[0].vlines(x=ACT_error_TT[:,0], ymin=(ACT_error_TT[:,1]/ACT_error_TT[:,0]/(ACT_error_TT[:,0]+1)*2*math.pi/(Tcmb**2) - ACT_error_TT[:,2]/ACT_error_TT[:,0]/(ACT_error_TT[:,0]+1)*2*math.pi/(Tcmb**2) - tt_LCDM_function(ACT_error_TT[:,0]))/tt_LCDM_function(ACT_error_TT[:,0]), ymax=(ACT_error_TT[:,1]/ACT_error_TT[:,0]/(ACT_error_TT[:,0]+1)*2*math.pi/(Tcmb**2) + ACT_error_TT[:,2]/ACT_error_TT[:,0]/(ACT_error_TT[:,0]+1)*2*math.pi/(Tcmb**2) - tt_LCDM_function(ACT_error_TT[:,0]))/tt_LCDM_function(ACT_error_TT[:,0]))

#ax[0].plot(l, (tt_dmeff_EFT - tt_LCDM_EFT)/tt_LCDM_EFT, color = 'orange', label=r'$IDM + EFT/LCDM + EFT$')
ax[0].plot(l, tt_dmeff*1e15, color = 'purple', label=r'$IDM$')
ax[0].plot(l, tt_LCDM*1e15, color = 'green', label=r'$LCDM$')
ax[0].legend(fontsize='8',ncol=1,loc='lower left')

#ax[1].scatter(ACT_error_EE[:,0], (ACT_error_EE[:,1]/ACT_error_EE[:,0]/(ACT_error_EE[:,0]+1)*2*math.pi/(Tcmb**2) - ACT_error_EE[:,2]/ACT_error_EE[:,0]/(ACT_error_EE[:,0]+1)*2*math.pi/(Tcmb**2) - ee_LCDM_function(ACT_error_EE[:,0]))/ee_LCDM_function(ACT_error_EE[:,0]))
#ax[1].scatter(ACT_error_EE[:,0], (ACT_error_EE[:,1]/ACT_error_EE[:,0]/(ACT_error_EE[:,0]+1)*2*math.pi/(Tcmb**2) + ACT_error_EE[:,2]/ACT_error_EE[:,0]/(ACT_error_EE[:,0]+1)*2*math.pi/(Tcmb**2) - ee_LCDM_function(ACT_error_EE[:,0]))/ee_LCDM_function(ACT_error_EE[:,0]))

#ax[1].vlines(x=ACT_error_EE[:,0], ymin=(ACT_error_EE[:,1]/ACT_error_EE[:,0]/(ACT_error_EE[:,0]+1)*2*math.pi/(Tcmb**2) - ACT_error_EE[:,2]/ACT_error_EE[:,0]/(ACT_error_EE[:,0]+1)*2*math.pi/(Tcmb**2) - ee_LCDM_function(ACT_error_EE[:,0]))/ee_LCDM_function(ACT_error_EE[:,0]), ymax=(ACT_error_EE[:,1]/ACT_error_EE[:,0]/(ACT_error_EE[:,0]+1)*2*math.pi/(Tcmb**2) + ACT_error_EE[:,2]/ACT_error_EE[:,0]/(ACT_error_EE[:,0]+1)*2*math.pi/(Tcmb**2) - ee_LCDM_function(ACT_error_EE[:,0]))/ee_LCDM_function(ACT_error_EE[:,0]))

#ax[1].plot(l, (ee_dmeff-ee_LCDM)/ee_LCDM * 100, color = 'green', label=r'$IDM$')
#ax[1].plot(l, (ee_dmeff_EFT-ee_dmeff)/ee_dmeff * 100, color = 'purple', label=r'$IDM + EFT$')
#ax[1].plot(l, (ee_dmeff_EFT-ee_LCDM_EFT)/ee_LCDM_EFT * 100, color = 'orange', label=r'$IDM + EFT$')
#ax[1].plot(l, ee_dmeff/ee_LCDM, color = 'purple', label=r'$IDM/LCDM$')
#ax[1].plot(l, (ee_dmeff_EFT - ee_LCDM_EFT)/ee_LCDM_EFT, color = 'orange', label=r'$IDM + EFT/LCDM + EFT$')
ax[1].plot(l, ee_dmeff*1e15, color = 'purple', label=r'$IDM$')
ax[1].plot(l, ee_LCDM*1e15, color = 'green', label=r'$LCDM$')
ax[1].legend(fontsize='8',ncol=1,loc='lower left')

#ax[2].plot(k, power_spectrum_dmeff/power_spectrum_LCDM, color = 'green', label=r'$IDM, n = 0$')
ax[2].plot(k, WDM_Pk_vec, color = 'blue', label=r'$WDM, m = 0.5 keV$')

ax[2].plot(k, power_spectrum_EFT_dmeff/power_spectrum_LCDM, color = 'green', label=r'$EFT$')
ax[2].plot(k, power_spectrum_halofit/power_spectrum_LCDM, color = 'blue', label=r'$Halofit$')

ax[2].legend(fontsize='8',ncol=1,loc='lower left')
#ax[2].plot(k, (power_spectrum_dmeff_EFT-power_spectrum_dmeff)/power_spectrum_dmeff * 100, color = 'purple', label=r'$IDM+ EFT$')
#ax[2].plot(k, pk_g0_m_dmeff, color = 'orange', label=r'$IDM + EFT$')
#ax[2].plot(k, pk_g0_m_lcdm, color = 'purplegreen', label=r'$LCDM$')
#ax[2].plot(k, power_spectrum_test, color = 'orange', label=r'$IDM + EFT$')
#ax[2].plot(l, pp_dmeff/pp_LCDM, color = 'orange', label=r'$IDM/LCDM$')
#ax[2].plot(k, power_spectrum_dmeff/power_spectrum_LCDM, color='purple', label = 'IDM')
#ax[2].plot(k, power_spectrum_LCDM/power_spectrum_LCDM, color='green', label='LCDM + EFT/LCDM + EFT')
#ax[2].plot(k, linear, color = 'orange', label=r'$IDM$')
#ax[2].plot(l, pp_LCDM_EFT, color = 'green', label=r'$LCDM + EFT$')
#ax[2].legend(fontsize='8',ncol=1,loc='lower left')

plt.show()

#Cleaning up

M.struct_cleanup()
M.empty()
M1.struct_cleanup()
M1.empty()
