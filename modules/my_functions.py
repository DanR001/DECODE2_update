import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from scipy.integrate import trapezoid as trapz, cumulative_trapezoid as cumtrapz
from scipy.optimize import curve_fit
from scipy.interpolate import *
from scipy.integrate import *
from scipy import special
import pandas as pd
from tqdm import tqdm
from time import time
import os
import sys

from colossus.lss import mass_function
from colossus.cosmology import cosmology

import warnings
warnings.filterwarnings("ignore")

cosmo = cosmology.setCosmology('planck18')
h=cosmo.H0/100.
h70=cosmo.H0/70.

c=3e8 #[m/s]
Msun=2e30 #[kg]
erg_to_J=1e-7
sec_to_yr=31536000
logLsun=np.log10(3.846e33) #[erg/s]
scatter_mhdotlog_mhlog=0.194



nz=50
z=np.linspace(0.01,5.01,nz)
logz=np.log10(z)
cosmic_time=np.flip(cosmo.age(z))
lb_time=cosmo.lookbackTime(z)
dzdt=1.02276e-10*h*(1.+z)*np.sqrt(1.-cosmo.Om0+cosmo.Om0*(1.+z)**3)
nms=350
mslog_min=5.
mslog_max=12.
mslog=np.linspace(mslog_min,mslog_max,nms)
ms=10.**mslog

nmhdot=1000
mhdotlog=np.linspace(0.5,17.,nmhdot)

sfrlog = np.arange(-6, 5, 0.1)
nsfr=sfrlog.size


nmh=1000
mhlog=np.linspace(5.,15.5,nmh)
mh=10.**mhlog
mhlog_min=11.; mhlog_min_cat=11.


nhsar=100
hsarlog=np.linspace(-5.,2.,nhsar)
hsar=10.**hsarlog


sbin=0.1
logsigma=np.arange(1., 3., sbin)
nsigma=logsigma.size


mbhdotlog=np.arange(-6, 2, 0.1)
nmbhdot=mbhdotlog.size



def nearest(vector,value):
    res=(np.abs(vector - value)).argmin()
    return res



def compute_dyn_friction_timescale(z):
    x=cosmo.Om(z)-1.
    Dvir=18.*np.pi**2.+82.*x-39.*x**2.
    return 1.628 / h / np.sqrt(Dvir/178.) * (cosmo.H0/cosmo.Hz(z))



def integrate_accretion_rates_across_time(z,log_acc_rate_cat):
    mass = np.flip( np.transpose( cumtrapz(np.flip(10.**np.transpose(log_acc_rate_cat)), np.flip(cosmo.age(z))*10.**9., axis=0, initial=0.) ) )
    return mass, np.log10(mass)



def mhdotlog_mhlog_relation(z,Z,mhlog):
    params = np.array([[  1.10320981,   1.04189297,   1.05466866,   1.01121661, 1.03165295,   0.99784522,   1.01528466,   0.98234759,
              1.00045836,   0.98121872,   0.99934336,   0.97855251, 0.99174069,   0.99147979,   0.97325701,   0.986431  ,
              0.99394462,   0.97616078,   0.98280507,   0.97281999, 0.97521068,   0.97400765,   0.97877229,   0.96974662,
              0.97969836,   0.98136448,   0.96579367,   0.97107004, 0.97137628,   0.96412803,   0.94359286,   0.95533298,
              0.95484576,   0.95249929,   0.93367736,   0.92886104, 0.92186463,   0.92433048,   0.92918776,   0.91255546,
              0.8868027 ,   0.88575913,   0.88388416,   0.87907199, 0.8759381 ,   0.88405567,   0.87797213,   0.87348   ,
              0.87810947,   0.87648979],
              [-11.67642727, -10.77454112, -10.87843318, -10.23739005, -10.42739899,  -9.92310931, -10.08444426,  -9.61259494,
             -9.78200797,  -9.47563309,  -9.65115942,  -9.34054795, -9.46072264,  -9.42756152,  -9.15990431,  -9.27898178,
             -9.33309742,  -9.08339908,  -9.13010439,  -8.97306045, -8.98114725,  -8.9410926 ,  -8.97140367,  -8.83409055,
             -8.91913663,  -8.91377223,  -8.71184186,  -8.74639581, -8.74285867,  -8.64603941,  -8.40088859,  -8.50008721,
             -8.48270876,  -8.44779874,  -8.22080449,  -8.15948178, -8.0692837 ,  -8.08691714,  -8.12882938,  -7.92934301,
             -7.6608463 ,  -7.63523838,  -7.60814981,  -7.5414963 , -7.4935035 ,  -7.56230647,  -7.50432552,  -7.43920168,
             -7.47667834,  -7.44697678]])
    m = interp1d(z,params[0,:],fill_value="extrapolate")(Z)
    q = interp1d(z,params[1,:],fill_value="extrapolate")(Z)
    return interp1d(mhlog, mhlog*m+q,fill_value="extrapolate")


def Mbh_sigma_relation(logsigma, z, velocity_dispersion):
    #logsigma in [km/s]
    #alpha=3.83; beta=8.21 #fit from Agnese's thesis
    if velocity_dispersion=="Ferrarese+2002":
        alpha=4.58; beta=8.22 #Merritt&Ferrarese+2001, sigma_c computed at r_e/8, with r_e the half light radius
        return alpha*(logsigma-np.log10(200))+beta
    if velocity_dispersion=="Marsden+2022":
        mm=np.loadtxt("Data/Marsden_2022/Mbh_sigma.txt")
        marsden=interp2d(np.array([0.1,1.,2.,3]), np.arange(2.0,2.441,0.02), mm)
        return np.array([marsden(z,logsigma[i])[0] for i in range(logsigma.size)])



def abundance_matching(z,nz,mhdotlog,nmhdot, phisfr,sfrlog,nsfr, dNdVdlogmhdot_active, sigma_am=0.2, delay=False):
    if delay:
        tau_dyn=np.array([compute_dyn_friction_timescale(Z) for Z in z])
        z_after_dyn=cosmo.age( cosmo.age(z)+tau_dyn, inverse=True)

    phisfr_cum=np.zeros((nsfr,nz))
    for iz in range(nz):
        phisfr_cum[:,iz]=trapz(phisfr[:,iz],sfrlog)-cumtrapz(phisfr[:,iz],sfrlog,initial=0.)
    phisfr_cum[phisfr_cum<0.]=10.**(-66.)

    if sigma_am==0.:
        dNdVdlogmhdot_active_cum=np.zeros((nz,nmhdot))
        for iz in range(nz):
            dNdVdlogmhdot_active_cum[iz,:]=trapz(dNdVdlogmhdot_active[iz,:],mhdotlog)-cumtrapz(dNdVdlogmhdot_active[iz,:],mhdotlog,initial=0.)
        dNdVdlogmhdot_active_cum[dNdVdlogmhdot_active_cum<0.]=10.**(-66.)
        sfrlog_am=np.zeros((nmhdot,nz))
    elif sigma_am>0.:
        dNdVdlogmhdot_active_cum=np.zeros((nz,nmhdot))
        integrand_2nd_part = np.array([ 0.5*special.erfc((mhdotlog[imhdot]-mhdotlog)/np.sqrt(2.)/sigma_am) for imhdot in range(nmhdot) ])
        for iz in range(nz):
            if delay:
                z_before_delay=interp1d(z_after_dyn,z,fill_value="extrapolate")(z[iz])
                dNdVdlogmhdot_active_cum[iz,:] = np.trapz(dNdVdlogmhdot_active[nearest(z,z_before_delay),:]*integrand_2nd_part, mhdotlog, axis=1)
            elif not delay:
                dNdVdlogmhdot_active_cum[iz,:] = np.trapz(dNdVdlogmhdot_active[iz,:]*integrand_2nd_part, mhdotlog, axis=1)
        dNdVdlogmhdot_active_cum[dNdVdlogmhdot_active_cum<0.]=10.**(-66.)
        sfrlog_am=np.zeros((nmhdot,nz))
    for iz in range(nz):
        sfrlog_am[:,iz] = interp1d(np.log10(np.flip(phisfr_cum[:,iz])),np.flip(sfrlog),fill_value="extrapolate")(np.log10(dNdVdlogmhdot_active_cum[iz,:]))
        if sfrlog_am[np.isfinite(sfrlog_am[:,iz]),iz].size >0:
            sfrlog_am[np.logical_not(np.isfinite(sfrlog_am[:,iz])),iz] = sfrlog_am[np.isfinite(sfrlog_am[:,iz]),iz][-1]
    return 10.**sfrlog_am, sfrlog_am



def smooth_sfr_har_relation(nz,sfrlog_am,nmhdot,mhdotlog):
    for iz in range(nz):
        dsfrlog_dmhdotlog=np.diff(sfrlog_am[:,iz]) / np.diff(mhdotlog)
        for imhdot in range(nmhdot-1):
            if dsfrlog_dmhdotlog[imhdot] > 0.:
                idx=imhdot
                break
        if idx<nmhdot:
            sfrlog_am[:idx,iz]=interp1d(mhdotlog[idx:], sfrlog_am[idx:,iz], fill_value="extrapolate")(mhdotlog[:idx])
    return sfrlog_am



def SMHM_double_pl(mhlog,N,logM,b,g):
    return np.log10( np.power(10, mhlog) * 2*N* np.power( (np.power(np.power(10,mhlog-logM), -b) + np.power(np.power(10,mhlog-logM), g)), -1) )



def sfr_catalogue(z,nz,nhalo,sfrlog_am,mhdotlog,dmhdtlog_arr,mhlog_arr,mhlog_crit, scatter_mhlog_crit=0.,delay=False):
    if delay:
        tau_dyn=np.array([compute_dyn_friction_timescale(Z) for Z in z])
        z_after_dyn=cosmo.age( cosmo.age(z)+tau_dyn, inverse=True)
    sfrlog_cat=np.zeros((nhalo,nz))
    jquench_cat=np.zeros((nhalo,nz),dtype=bool)
    for iz in range(nz):
        if delay:
            z_before_delay=interp1d(z_after_dyn,z,fill_value="extrapolate")(z[iz])
            sfrlog_cat[:,iz]=interp1d(mhdotlog,sfrlog_am[:,iz],bounds_error=False, fill_value='extrapolate')(dmhdtlog_arr[:,nearest(z,z_before_delay)]+9.)
        else:
            sfrlog_cat[:,iz]=interp1d(mhdotlog,sfrlog_am[:,iz],bounds_error=False, fill_value='extrapolate')(dmhdtlog_arr[:,iz]+9.)
        jquench_cat[:,iz]=mhlog_arr[:,iz]>mhlog_crit(z[iz]) + np.random.normal(0,scatter_mhlog_crit,mhlog_arr[:,iz].size)
        sfrlog_cat[jquench_cat[:,iz],iz]=-66.
    #correzione per riattivazioni
    for ihalo in range(nhalo):
        for iz in range(0,nz):
            if sfrlog_cat[ihalo,iz]==-66.:
                sfrlog_cat[ihalo,:iz]=-66.
    return sfrlog_cat



def SMHM_scatter_from_logms_cat(logmstar_integrated, mhlog_arr, iz, mhlog_smhm, Mstar_range, b, volume, hmf, mhlog, mhlog_min):
    mslog_smhm = np.array([ np.mean(logmstar_integrated[np.logical_and.reduce((logmstar_integrated[:,iz]>0., mhlog_arr[:,iz]>m-b/2., mhlog_arr[:,iz]<=m+b/2.)), iz]) for m in mhlog_smhm ])
    scatter_smhm = np.array([ np.std(logmstar_integrated[np.logical_and.reduce((logmstar_integrated[:,iz]>0., mhlog_arr[:,iz]>m-b/2., mhlog_arr[:,iz]<=m+b/2.)), iz]) for m in mhlog_smhm ])
    if mslog_smhm[np.isfinite(mslog_smhm)].size<2 or scatter_smhm[np.isfinite(scatter_smhm)].size<2:
        return None, None
    scatter = np.mean(scatter_smhm[np.isfinite(scatter_smhm)])

    try:
        popt,pcov=curve_fit(SMHM_double_pl, mhlog_smhm[np.isfinite(mslog_smhm)], mslog_smhm[np.isfinite(mslog_smhm)], p0 = [0.032,12.,2.,0.608])
        mask=np.isfinite(mslog_smhm)
        chi=np.sum( (mslog_smhm[mask]-SMHM_double_pl(mhlog_smhm[mask],*popt) )**2/mslog_smhm[mask].size )
        if chi<0.5:
            smhm = interp1d(mhlog_smhm, SMHM_double_pl(mhlog_smhm,*popt), fill_value="extrapolate")
            return smhm, scatter
        else:
            smhm = interp1d(mhlog_smhm[np.isfinite(mslog_smhm)], mslog_smhm[np.isfinite(mslog_smhm)], fill_value="extrapolate")
            return smhm, scatter
    except:
        smhm = interp1d(mhlog_smhm[np.isfinite(mslog_smhm)], mslog_smhm[np.isfinite(mslog_smhm)], fill_value="extrapolate")
        return smhm, scatter



def compute_subHMF(z_, logmh, HMF):
    #HMF in normal units, not in log
    a = 1./(1.+z_)
    C = np.power(10., -2.415 + 11.68*a - 28.88*a**2 + 29.33*a**3 - 10.56*a**4)
    logMcutoff = 10.94 + 8.34*a - 0.36*a**2 - 5.08*a**3 + 0.75*a**4
    return HMF * C * (logMcutoff - logmh)


def compute_objs_from_mass_function(logm, mf, volume, mask):
    #HMF in normal units, not in log
    MF=mf*volume
    cum_hmf_tot = np.cumsum(MF[mask])
    max_number = np.floor(np.trapz(MF[mask], logm[mask]))
    if (np.random.uniform(0,1) > np.trapz(MF[mask], logm[mask])-max_number): #Calculating number of halos to compute
        max_number += 1
    int_cum_phi = interp1d(cum_hmf_tot, logm[mask])
    range_numbers = np.random.uniform(np.min(cum_hmf_tot), np.max(cum_hmf_tot), int(max_number))
    return int_cum_phi(range_numbers)



def correct_sfrlog_quenched_Sats_missingGals(iz, mhlog_arr, sfrlog_cat, sats_mhlog_cat, delta_mhlog_cat):
    b=0.1; mhlog_smhm=np.arange(10,15.5,0.1)
    f_quench_mhlog = np.zeros(mhlog_smhm.size)
    for i in range(mhlog_smhm.size):
        mask=np.logical_and(mhlog_arr[:,iz]>=mhlog_smhm[i]-b/2., mhlog_arr[:,iz]<mhlog_smhm[i]+b/2.)
        mask_quench=np.logical_and(mask, sfrlog_cat[:,iz]==-66.)
        if mhlog_arr[mask,iz].size>0:
            f_quench_mhlog[i] = mhlog_arr[mask_quench,iz].size / mhlog_arr[mask,iz].size
        else:
            f_quench_mhlog[i] = np.nan
    sats_prob_quench = interp1d(mhlog_smhm, f_quench_mhlog, fill_value="extrapolate") (sats_mhlog_cat[iz])
    delta_prob_quench = interp1d(mhlog_smhm, f_quench_mhlog, fill_value="extrapolate") (delta_mhlog_cat[iz])
    sats_randn = np.random.uniform(0,1,sats_prob_quench.size)
    delta_randn = np.random.uniform(0,1,delta_prob_quench.size)
    mask_sats = np.logical_and.reduce((sats_prob_quench>=0., sats_prob_quench<=1., sats_randn<sats_prob_quench))
    mask_delta = np.logical_and.reduce((delta_prob_quench>=0., delta_prob_quench<=1., delta_randn<delta_prob_quench))
    return mask_sats, mask_delta
