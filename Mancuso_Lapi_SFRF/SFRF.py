import numpy as np
from scipy.special import erfc

def SFR_function_UV_IR_Mancuso_Lapi(z_sfr_grid=None):

    if z_sfr_grid==None:
        red_min=0.#redshift minimo della tua griglia
        red_max=100.#redshift minimo della tua griglia
        nred=100#punti nella griglia dei redshift
        sfrlog_min=-2.#sfrlog minima della tua griglia
        sfrlog_max=4.#sfrlog massima della tua griglia
        nsfr=100#punti nella griglia delle sfr
        #creo le griglie
        red=np.linspace(red_min,red_max,nred)
        sfrlog=np.linspace(sfrlog_min,sfrlog_max,nsfr)
    else:
        nred=z_sfr_grid[0]
        red=z_sfr_grid[1]
        nsfr=z_sfr_grid[2]
        sfrlog=z_sfr_grid[3]

    sfr=10.**sfrlog

    #calcolo le SFR functions
    phi0log=-2.13
    kphi1=-8.90
    kphi2=18.07
    kphi3=-11.77
    sfr0log=0.72
    ksfr1=8.56
    ksfr2=-10.08
    ksfr3=2.54
    alpha0=1.12
    kalpha1=3.73
    kalpha2=-7.80
    kalpha3=5.15
    #only UV
    phi0log_uv=-1.96
    kphi1_uv=-1.60
    kphi2_uv=4.22
    kphi3_uv=-5.23
    sfr0log_uv=0.01
    ksfr1_uv=2.85
    ksfr2_uv=0.43
    ksfr3_uv=-1.70
    alpha0_uv=1.11
    kalpha1_uv=2.85
    kalpha2_uv=-6.18
    kalpha3_uv=4.44

    phisfr_etg_lapi=np.zeros((nsfr,nred))
    phisfr_ltg_lapi=np.zeros((nsfr,nred))
    phisfr_lapi=np.zeros((nsfr,nred))
    for ired in range (len(red)):
        csi=np.log10(1.+red[ired])
        phiclog_uv=phi0log_uv+kphi1_uv*csi+kphi2_uv*csi**(2.)+kphi3_uv*csi**(3.)
        sfrclog_uv=sfr0log_uv+ksfr1_uv*csi+ksfr2_uv*csi**(2.)+ksfr3_uv*csi**(3.)
        alpha_uv=alpha0_uv+kalpha1_uv*csi+kalpha2_uv*csi**(2.)+kalpha3_uv*csi**(3.)
        if (red[ired] >= 1.1):
            phi_sfr_ltg=10.**(-33)
        else:
            phi_sfr_ltg=10.**(phiclog_uv)*(sfr/(10.**(sfrclog_uv)))**(1.-alpha_uv)*np.exp(-(sfr/(10.**(sfrclog_uv))))
        phiclog=phi0log+kphi1*csi+kphi2*csi**(2.)+kphi3*csi**(3.)
        sfrclog=sfr0log+ksfr1*csi+ksfr2*csi**(2.)+ksfr3*csi**(3.)
        alpha=alpha0+kalpha1*csi+kalpha2*csi**(2.)+kalpha3*csi**(3.)
        phi_sfr_ltg=10.**(phiclog_uv)*(sfr/(10.**(sfrclog_uv)))**(1.-alpha_uv)*np.exp(-(sfr/(10.**(sfrclog_uv))))*0.5*erfc((red[ired]-1.1)/0.5)
        phi_sfr_etg=10.**(phiclog)*(sfr/(10.**(sfrclog)))**(1.-alpha)*np.exp(-(sfr/(10.**(sfrclog))))-phi_sfr_ltg
        phi_sfr_tot=10.**(phiclog)*(sfr/(10.**(sfrclog)))**(1.-alpha)*np.exp(-(sfr/(10.**(sfrclog))))
        phisfr_etg_lapi[:,ired]=phi_sfr_etg
        j_etg_lapi=phisfr_etg_lapi[:,ired]<0.
        phisfr_etg_lapi[j_etg_lapi,ired]=10.**(-33)
        phisfr_ltg_lapi[:,ired]=phi_sfr_ltg
        j_ltg_lapi=phisfr_ltg_lapi[:,ired]<0.
        phisfr_ltg_lapi[j_ltg_lapi,ired]=10.**(-33)
        phisfr_lapi[:,ired]=phisfr_etg_lapi[:,ired]+phisfr_ltg_lapi[:,ired]

    if z_sfr_grid==None:
        return nred, red, nsfr, sfrlog, phisfr_lapi, phisfr_etg_lapi, phisfr_ltg_lapi
    else:
        return phisfr_lapi, phisfr_etg_lapi, phisfr_ltg_lapi
