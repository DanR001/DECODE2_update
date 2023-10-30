from modules.my_functions import *



def logLx_to_logMbh_acc_rate(logLx, eps, inverse=False):
    #logLx [erg/s]
    if inverse:
        return 10.**logLx * eps * c**2 * Msun / erg_to_J / sec_to_yr #[Msun/yr]
    elif not inverse:
        return 10.**logLx / eps / c**2 / Msun * erg_to_J * sec_to_yr #[Msun/yr]



def ed(z,logLx):
    p1_44=5.29; zb1_44=1.1; p2_44=-0.35; zb2=2.7; p3_44=-5.6
    alpha=0.18; logLxb=44.5; beta1=1.2; beta2=1.5
    p1=p1_44 + beta1 * (logLx - 44.)
    p2=p2_44 + beta2 * (logLx - 44.)
    p3=p3_44
    if logLx<=logLxb:
        zb1=zb1_44 * (10**logLx/10**logLxb)**alpha
    else:
        zb1=zb1_44
    if z<zb1:
        return (1+z)**p1
    elif zb1<=z<zb2:
        return (1+z)**p1 * ( (1+z) / (1+zb1) )**p2
    else:
        return (1+z)**p1 * ( (1+z) / (1+zb1) )**p2 * ( (1+z) / (1+zb2) )**p3



def Xray_L_function(nz, z, nLx, logLx, work="shen+2020"):
    Lxf=np.zeros((nLx,nz))
    if work=="miyaji+2015":
        for iz in range(nz):
            if z[iz]<0.2:#if 0.015<=z[iz]<0.2:
                zc=0.104; A44=1.2; Lxs=10.**43.7; gamma1=0.9; gamma2=2.53
            elif 0.2<=z[iz]<0.4:
                zc=0.296; A44=4.79; Lxs=10.**44.22; gamma1=1.05; gamma2=2.9
            elif 0.4<=z[iz]<0.6:
                zc=0.497; A44=7.64; Lxs=10.**43.98; gamma1=0.83; gamma2=2.7
            elif 0.6<=z[iz]<0.8:
                zc=0.697; A44=11.4; Lxs=10.**43.31; gamma1=0.41; gamma2=1.86
            elif 0.8<=z[iz]<1.:
                zc=0.897; A44=31.2; Lxs=10.**43.90; gamma1=0.40; gamma2=2.51
            elif 1.<=z[iz]<1.2:
                zc=1.098; A44=34.7; Lxs=10.**43.85; gamma1=0.19; gamma2=2.06
            elif 1.2<=z[iz]<1.6:
                zc=1.392; A44=44.2; Lxs=10.**44.43; gamma1=0.47; gamma2=2.82
            elif 1.6<=z[iz]<2.:
                zc=1.793; A44=36.8; Lxs=10.**44.57; gamma1=0.48; gamma2=2.73
            elif 2.<=z[iz]<2.4:
                zc=2.194; A44=42.7; Lxs=10.**44.50; gamma1=0.27; gamma2=2.84
            elif 2.4<=z[iz]<3.:
                zc=2.688; A44=32.8; Lxs=10.**44.63; gamma1=0.46; gamma2=3.12
            elif 3.<=z[iz]<5.8:
                zc=4.215; A44=4.37; Lxs=10.**44.94; gamma1=0.65; gamma2=4.38
            else:
                sys.exit("z > 5.8")
            evo_param=np.array([ ed(z[iz],logLx[iLx]) for iLx in range(nLx) ])
            Lxf[:,iz] = 1e-6*A44**z[iz] * evo_param * ( (10**44/Lxs)**gamma1 + (10**44/Lxs)**gamma2 ) / ((10**logLx/Lxs)**gamma1 + (10**logLx/Lxs)**gamma2)
    if work=="shen+2020":
        red_sample = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2., 3., 4., 5., 6.])
        gamma1_sample = np.array([0.812, 0.561, 0.599, 0.504, 0.484, 0.411, 0.424, 0.403, 0.26, 1.196])
        gamma2_sample = np.array([1.753, 2.108, 2.199, 2.423, 2.546, 2.487, 1.878, 1.988, 1.916, 2.349])
        logphiS_sample = np.array([-4.405, -4.151, -4.412, -4.53, -4.668, -4.679, -4.698, -5.244, -5.258, -8.019])
        logLs_sample = np.array([11.407, 11.65, 12.223, 12.622, 12.919, 13.011, 12.708, 12.73, 12.319, 13.709]) + logLsun
        logLxf_temp=np.zeros((nLx,red_sample.size))
        for iz in range(red_sample.size):
            logLxf_temp[:,iz] = logphiS_sample[iz] - np.log10 ( ( 10.**logLx/10.**logLs_sample[iz] )**gamma1_sample[iz] + ( 10.**logLx/10.**logLs_sample[iz] )**gamma2_sample[iz] )
        for iL in range(nLx):
            Lxf[iL,:] = 10.**interp1d(red_sample, logLxf_temp[iL,:], fill_value="extrapolate")(z)
    return Lxf



def compute_mbh_accrate_function(nz, z, nLx, logLx, Lxf, nmbhdot, mbhdotlog, eps, volume_mock, logLxmin, correct_active, logmstar_integrated, sfrlog_cat, volume):
    b=mbhdotlog[1]-mbhdotlog[0]
    phimbhdot=np.zeros((nmbhdot, nz))
    mbhdotlog_temp = np.arange(mbhdotlog[0]-b/2, mbhdotlog[-1]+b/2+0.1, b)
    #volume_mock = 550**3

    Mslog = np.arange(7,12.5,0.1)

    for iz in range(nz):
        logLx_mock = compute_objs_from_mass_function(logLx, Lxf[:,iz], volume_mock, mask=logLx>logLxmin)
        mbhdotlog_mock = np.log10(logLx_to_logMbh_acc_rate(logLx_mock,eps))
        phimbhdot[:,iz] = np.histogram(mbhdotlog_mock, bins=mbhdotlog_temp)[0] / b / volume_mock

        if correct_active:
            phiMs = np.histogram(logmstar_integrated[:,iz], bins=Mslog)[0] / 0.1 / volume
            logcumphiMs = np.log10( np.flip(cumtrapz(np.flip(phiMs), -np.flip(Mslog[:-1]))) )
            logcumphiMs = np.append(logcumphiMs, -66.)
            f_ac = np.histogram(logmstar_integrated[sfrlog_cat[:,iz]>-66.,iz], bins=Mslog)[0] / 0.1 / volume / phiMs
            logcumphiMbhdot = np.log10( np.flip(cumtrapz(np.flip(phimbhdot[:,iz]), -np.flip(mbhdotlog_temp[:-1]))) )
            logcumphiMbhdot = np.append(logcumphiMbhdot, -66.)
            mask = np.logical_and( np.isfinite(phiMs), np.isfinite(f_ac) )
            if f_ac[mask].size>2:
                f_ac_corrMbhdot = interp1d(logcumphiMs[mask], f_ac[mask], fill_value="extrapolate")(logcumphiMbhdot)
                phimbhdot[:,iz] = phimbhdot[:,iz]*f_ac_corrMbhdot
            phimbhdot[np.isnan(phimbhdot[:,iz]),iz]=0.

    return phimbhdot



def assign_bhar_aird(z,nz,nhalo,sfrlog_cat,logmstar_integrated):
    mbhdotlog_cat=np.zeros((nhalo,nz))
    red_sample=[0.3, 0.75, 1.25, 1.75, 2.25, 2.75]
    for iz in range(nz):
        Z=red_sample[nearest(red_sample,z[iz])]
        if Z==0.3 or Z==0.75:
            logms_sample=[8.75,9.25,9.75,10.25,10.75,11.25]
        elif Z==1.25:
            logms_sample=[9.25,9.75,10.25,10.75,11.25]
        elif Z==1.75:
            logms_sample=[9.75,10.25,10.75,11.25]
        elif Z==2.25 or Z==2.75:
            logms_sample=[10.25,10.75,11.25]
        p=[np.loadtxt("Data/Aird+2020/p_BHAR_SFR_z={:.2f}_Ms={:.2f}.txt".format(Z,logMS)) for logMS in logms_sample]
        cum_int=[ cumtrapz(10.**p[i][:,1], p[i][:,0], initial=0.) for i in range(len(logms_sample)) ]
        interps=[ interp1d(cum_int[i], p[i][:,0], fill_value="extrapolate") for i in range(len(logms_sample)) ]
        maxs=[np.max(cum_int[i]) for i in range(len(logms_sample)) ]
        mins=[np.min(cum_int[i]) for i in range(len(logms_sample)) ]
        indexes=[ nearest(logms_sample,logmstar_integrated[i,iz]) for i in range(nhalo) ]
        randn=np.array([ np.random.uniform(mins[indexes[i]],maxs[indexes[i]]) for i in range(nhalo) ])
        mbhdotlog_cat[:,iz]=np.array([float(interps[indexes[i]](randn[i])) for i in range(nhalo) ]) + sfrlog_cat[:,iz]
    return mbhdotlog_cat



def assign_bhar_yang(z,nz,nhalo,sfrlog_cat,logmstar_integrated,scatter):
    red_sample=np.array([0.5,0.8,1.,1.5,2.,3.,4.])
    mbhdotlog_cat=np.zeros((nhalo,nz))
    for iz in range(nz):
        Z=red_sample[nearest(red_sample,z[iz])]
        bhar_mstar_data=np.loadtxt("Data/Yang+2018/BHAR_Mstar_StarForming_z=%.1f.txt"%Z)
        bhar_mstar=interp1d(bhar_mstar_data[:,0], bhar_mstar_data[:,1], fill_value="extrapolate")
        mbhdotlog_cat[:,iz]=bhar_mstar(logmstar_integrated[:,iz])+np.random.normal(0,scatter,nhalo)
        mbhdotlog_cat[sfrlog_cat[:,iz]<-60.,iz]=-66.
    return mbhdotlog_cat



def assign_bhar_carraro(z,nz,nhalo,sfrlog_cat,logmstar_integrated,scatter):
    mbhdotlog_cat=np.zeros((nhalo,nz))
    for iz in range(nz):
        if z[iz]<0.65:
            m=1.02; q=-13.7
        elif 0.65<=z[iz]<1.3:
            m=0.87; q=-11.5
        elif 1.3<=z[iz]<2.25:
            m=1.18; q=-14.4
        elif z[iz]>=2.25:
            m=1.; q=-12.2
        mbhdotlog_cat[:,iz]=m*logmstar_integrated[:,iz]+q+np.random.normal(0,scatter,nhalo)
        mbhdotlog_cat[sfrlog_cat[:,iz]<-60.,iz]=-66.
    return mbhdotlog_cat



def compute_critical_quenching_BH_mass(nsigma,logsigma,nz,z,nhalo,logsigma_arr,velocity_dispersion):
    logMbhcrit=np.zeros((nsigma,nz))
    for iz in range(nz):
        logMbhcrit[:,iz]=Mbh_sigma_relation(logsigma, z[iz], velocity_dispersion)
    logMbhcrit_interpolators = [ interp1d(logsigma,logMbhcrit[:,iz],fill_value="extrapolate") for iz in range(nz) ]
    logMbhcrit_arr = np.zeros((nhalo,nz))
    for im in range(nhalo):
        logMbhcrit_arr[im,:] = np.array([ logMbhcrit_interpolators[iz](logsigma_arr[im,iz]) for iz in range(nz) ])
    return logMbhcrit_arr



def compute_sats_delta_mh_mhdot_forPhiTrue_abundancematching(nz,z,mhlog,hmf,hmf_cat,volume):
    delta_mhlog_cat = []
    delta_mhdotlog_cat = []
    sats_mhlog_cat = []
    sats_mhdotlog_cat = []
    for iz in range(nz):

        mhdotlog_mhlog = mhdotlog_mhlog_relation(z,z[iz],mhlog)
        scatter_mhdotlog_mhlog=0.2

        correction = compute_subHMF(z[iz], mhlog, hmf[:,iz])
        halo_masses = compute_objs_from_mass_function(mhlog, correction, volume, mask=mhlog>11)
        sats_mhlog_cat.append(halo_masses)
        sats_mhdotlog_cat.append( mhdotlog_mhlog(halo_masses) + np.random.normal(0,scatter_mhdotlog_mhlog,halo_masses.size) )

        delta_hmf = hmf[:,iz] - hmf_cat[:,iz]
        halo_masses = compute_objs_from_mass_function(mhlog, delta_hmf, volume, mask=np.logical_and(delta_hmf>0.,mhlog>11))
        delta_mhlog_cat.append(halo_masses)
        delta_mhdotlog_cat.append( mhdotlog_mhlog(halo_masses) + np.random.normal(0,scatter_mhdotlog_mhlog,halo_masses.size) )
    return delta_mhlog_cat, delta_mhdotlog_cat, sats_mhlog_cat, sats_mhdotlog_cat



def compute_phisfr_true(it,nz,z,volume,mhdotlog,sfrlog,sfrlog_am,mhlog_arr,sfrlog_cat,sats_mhlog_cat, delta_mhlog_cat, sats_mhdotlog_cat, delta_mhdotlog_cat):
    delta_sfrlog_cat = []
    sats_sfrlog_cat = []
    for iz in range(nz):
        sats_sfrlog_cat.append( interp1d(mhdotlog,sfrlog_am[:,iz],bounds_error=False, fill_value='extrapolate')(sats_mhdotlog_cat[iz]+9.) )
        delta_sfrlog_cat.append( interp1d(mhdotlog,sfrlog_am[:,iz],bounds_error=False, fill_value='extrapolate')(delta_mhdotlog_cat[iz]+9.) )

    if it > 0:
        mask_sats, mask_delta = correct_sfrlog_quenched_Sats_missingGals(iz, mhlog_arr, sfrlog_cat, sats_mhlog_cat, delta_mhlog_cat)
        sats_sfrlog_cat[iz][mask_sats]=-66.
        delta_sfrlog_cat[iz][mask_delta]=-66.

    phisfr_true=np.zeros((nsfr,nz))
    for iz in range(nz):
        phisfr_true[:,iz] = np.histogram( np.concatenate((sfrlog_cat[:,iz], sats_sfrlog_cat[iz], delta_sfrlog_cat[iz] )), bins=np.arange(sfrlog[0]-0.05, sfrlog[-1]+0.1, 0.1))[0] / 0.1 / volume

    return phisfr_true, delta_sfrlog_cat, sats_sfrlog_cat



def quench_blackhole_mask(nz, z, logS, logM, logsigma, logMcrit):
    diff=logMcrit-logM
    if diff[diff<0].size>0:
        idx=np.max(np.where(diff<0))
    else:
        idx=0
    mask=np.concatenate(( np.repeat(True,idx), np.repeat(False,nz-idx) ))
    return mask



def quench_blackholes_galaxies(nz,nhalo,masks,sfrlog_cat,mbhdotlog_cat):
    for iz in range(nz):
        if z[iz]<6.:
            for im in range(nhalo):
                if masks[im,iz]:
                    sfrlog_cat[im,iz]=-66.
                    mbhdotlog_cat[im,iz]=-66.
    return sfrlog_cat,mbhdotlog_cat



def assign_BHAR_cat(z,nz,nhalo,sfrlog_cat,logmstar_integrated,bhar_function):
    if bhar_function=="Aird+2019":
        return assign_bhar_aird(z,nz,nhalo,sfrlog_cat,logmstar_integrated)
    elif bhar_function=="Yang+2018":
        return assign_bhar_yang(z,nz,nhalo,sfrlog_cat,logmstar_integrated,0.3)
    elif bhar_function=="Carraro+2020":
        return assign_bhar_carraro(z,nz,nhalo,sfrlog_cat,logmstar_integrated,0.3)



def compute_BHAR_function_cat(nmbhdot,nz,mbhdotlog_cat,mbhdotlog,volume):
    phimbhdot_cat=np.zeros((nmbhdot,nz))
    for iz in range(nz):
        mbhdotlog_cat[np.logical_not(np.isfinite(mbhdotlog_cat[:,iz])),iz] = -66.
        phimbhdot_cat[:,iz] = np.histogram(mbhdotlog_cat[:,iz], bins=np.arange(mbhdotlog[0]-0.05, mbhdotlog[-1]+0.1, 0.1))[0] / 0.1 / volume
    return phimbhdot_cat
