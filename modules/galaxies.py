from modules.my_functions import *
from modules.compute_merger_rates_decode import *



def smf_active(z,nz,ms,nms, work="Weaver+2022"):
    #active
    if work=="Davidzon+2017":
        red_sample=np.array([0.35,0.65,0.95,1.3,1.75,2.25,2.75,3.25,3.75,5.])
        logm_star_sample=np.array([10.26,10.40,10.35,10.42,10.40,10.45,10.39,10.83,10.77,11.3])
        m_star_sample=10.**(logm_star_sample)
        alpha1_sample=np.array([-1.29,-1.32,-1.29,-1.21,-1.24,-1.5,-1.52,-1.78,-1.84,-2.12])
        phi1_sample=np.array([2.41,1.661,1.739,1.542,1.156,0.441,0.441,0.086,0.052,0.003])
        alpha2_sample=np.array([1.01,0.84,0.81,1.11,0.9,0.59,1.05])
        phi2_sample=np.array([1.3,0.86,0.95,0.49,0.46,0.38,0.13])
        smf_dav_ac=np.zeros((len(red_sample),nms))
        for iz in range(len(red_sample)):
            if(red_sample[iz]<3.):
                smf_dav_ac[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])+phi2_sample[iz]*(ms/m_star_sample[iz])**(alpha2_sample[iz]))
            else:
                smf_dav_ac[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])
    if work=="Weaver+2022":
        red_sample=np.array([0.35,0.65,0.95,1.3,1.75,2.25,2.75,3.25,4.,5.])
        logm_star_sample=np.array([10.73,10.83,10.91,10.93,10.9,10.84,10.86,10.96,10.57,10.38])
        m_star_sample=10.**(logm_star_sample)
        alpha1_sample=np.array([-1.41,-1.4,-1.36,-1.36,-1.47,-1.47,-1.57,-1.57,-1.57,1.57])
        phi1_sample=np.array([0.8,0.7,0.74,0.66,0.29,0.24,0.19,0.12,0.13,0.1])
        alpha2_sample=np.array([-0.02,-0.31,-0.28,0.41,-0.41,0.01,0.31,0.,0.,0.])
        phi2_sample=np.array([0.49,0.36,0.29,0.11,0.31,0.13,0.04,0.,0.,0.])
        smf_dav_ac=np.zeros((len(red_sample),nms))
        for iz in range(len(red_sample)):
            smf_dav_ac[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])+phi2_sample[iz]*(ms/m_star_sample[iz])**(alpha2_sample[iz]))

    # interpolazione
    smflog_ac=np.zeros((nz,nms))
    for ims in range(nms):
        smflog_ac[:,ims]=interp1d(red_sample,np.log10(smf_dav_ac[:,ims]),bounds_error=False, fill_value="extrapolate")(z)
    # correzione ad alto z
    for iz in range(nearest(z,5)+1,nz):
        for ims in range(nms):
            if smflog_ac[iz,ims]>smflog_ac[iz-1,ims]:
                smflog_ac[iz,ims]=smflog_ac[iz-1,ims]
    smflog_ac[np.isnan(smflog_ac)]=-66.
    return smflog_ac



def smf_passive(z,nz,ms,nms, work="Weaver+2022"):
    #passive
    if work=="Davidzon+2017":
        red_sample=np.array([0.35,0.65,0.95,1.3,1.75,2.25,2.75,3.25,3.75])
        logm_star_sample=np.array([10.83,10.83,10.75,10.56,10.54,10.69,10.24,10.10,10.10])
        m_star_sample=10.**(logm_star_sample)
        alpha1_sample=np.array([-1.3,-1.46,-0.07,0.53,0.93,0.17,1.15,1.15,1.15])
        phi1_sample=np.array([0.098,0.012,1.724,0.757,0.251,0.068,0.028,0.01,0.004])
        alpha2_sample=np.array([-0.39,-0.21])
        phi2_sample=np.array([1.58,1.44])
        smf_dav_pas=np.zeros((len(red_sample),nms))
        for iz in range(len(red_sample)):
            if(red_sample[iz]<0.8):
                smf_dav_pas[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])+phi2_sample[iz]*(ms/m_star_sample[iz])**(alpha2_sample[iz]))
            else:
                smf_dav_pas[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz]))

    if work=="Weaver+2022":
        red_sample=np.array([0.35,0.65,0.95,1.3,1.75,2.25,2.75,3.25,4.,5.])
        logm_star_sample=np.array([10.9,10.88,10.89,10.63,10.48,10.49,10.33,10.41,10.58,10.75])
        m_star_sample=10.**(logm_star_sample)
        alpha1_sample=np.array([-0.63,-0.47,-0.47,0.11,0.54,0.66,1.3,1.41,0.95,0.95])
        phi1_sample=np.array([90.92,94.15,95.88,69.9,35.36,11.48,5.65,2.13,1.35,0.29])*0.01
        alpha2_sample=np.array([-1.83,-2.02,-2.02,-2.02,0.,0.,0.,0.,0.,0.])
        phi2_sample=np.array([0.78,0.18,0.11,0.11,-2.02,0.,0.,0.,0.,0.,0.])*0.01
        smf_dav_pas=np.zeros((len(red_sample),nms))
        for iz in range(len(red_sample)):
            smf_dav_pas[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])+phi2_sample[iz]*(ms/m_star_sample[iz])**(alpha2_sample[iz]))

    smflog_pas=np.zeros((nz,nms))
    for ims in range(nms):
        smflog_pas[:,ims]=interp1d(red_sample,np.log10(smf_dav_pas[:,ims]),bounds_error=False, fill_value="extrapolate")(z)
    for iz in range(1,nz):
        for ims in range(nms):
            if smflog_pas[iz,ims]>smflog_pas[iz-1,ims]:
                smflog_pas[iz,ims]=smflog_pas[iz-1,ims]
    smflog_pas[np.isnan(smflog_pas)]=-66.
    smf_pas=10.**smflog_pas
    saved=smflog_pas[0,:]

    return smflog_pas



def sfr_main_sequence(z,nz,mslog,nms,work="popesso+2023"):
    sfrlog_ms=np.zeros((nz,nms))

    #==========================================main sequence speagle+2014
    if work=="speagle+2014":
        sigma_speagle=0.3
        for ired in range(nz):
            sfrlog_ms[ired,:]=(0.84-0.026*cosmo.age(z[ired]))*mslog-(6.51-0.11*cosmo.age(z[ired]))
    #==========================================

    #==========================================main sequence whitaker+2014
    if work=="whitaker+2014":
        red_sample2=np.array([0.75,1.25,1.75,2.25])
        a_sample2=np.array([-27.4,-26.03,-24.04,-19.99])
        b_sample2=np.array([5.02,4.62,4.17,3.44])
        c_sample2=np.array([-0.22,-0.19,-0.16,-0.13])
        a_wh=np.interp(z,red_sample2,a_sample2)
        b_wh=np.interp(z,red_sample2,b_sample2)
        c_wh=np.interp(z,red_sample2,c_sample2)
        for ired in range(nz):
            sfrlog_ms[ired,:]=a_wh[ired]+b_wh[ired]*mslog+c_wh[ired]*mslog*mslog
    #==========================================

    #==========================================main sequence ilbert+2015
    if work=="ilbert+2015":
        a_il=-1.02; beta_il=-0.201; b_il=3.09; sigma_il=0.22
        ssfrlog_il=np.zeros((nz,nms))
        for ired in range(nz):
            ssfrlog_il[ired,:]=a_il+beta_il*10.**(mslog-10.5)+b_il*np.log10(1.+z[ired])
            sfrlog_ms[ired,:]=ssfrlog_il[ired,:]+mslog-9.
    #==========================================

    #==========================================main sequence bethermin+12 (non so se e la stessa di sargent+12, sargent non e chiaro)
    if work=="bethermin+2012":
        ssfrlog_0_bet=-10.2
        beta_bet=-0.2; gamma_bet=3.; zevo=2.5
        ssfrlog_bet=np.zeros((nz,nms))
        for ired in range(nz):
            ssfrlog_bet[ired,:]=ssfrlog_0_bet+beta_bet*(mslog-11.)+gamma_bet*np.log10(1.+min(z[ired],zevo))
            sfrlog_ms[ired,:]=ssfrlog_bet[ired,:]+mslog
    #==========================================

    #==========================================main sequence leja+2015 (Pip's fit)
    if work=="leja+2015":
        for iz in range(nz):
            s0 = 0.6 + 1.22*z[iz] - 0.2*z[iz]*z[iz]
            M0 = np.power(10., 10.3 + 0.753*z[iz] - 0.15*z[iz]*z[iz])
            alpha = 1.3 - 0.1*z[iz]
            sfrlog_ms[iz,:] = s0 - np.log10(1. + np.power(10.**mslog/M0, -alpha))
    #==========================================

    #==========================================main sequence leja+2022
    if work=="leja+2022":
        a = -0.06707 + 0.3684*z - 0.1047*z*z
        b = 0.8552 - 0.101*z - 0.001816*z*z
        c = 0.2148 + 0.8137*z - 0.08052*z*z
        Mt = 10.29 - 0.1284*z + 0.1203*z*z
        for im in range(nms):
            sfrlog_ms[mslog[im]>Mt,im] = a[mslog[im]>Mt]*(mslog[im] - Mt[mslog[im]>Mt]) +c[mslog[im]>Mt]
            sfrlog_ms[mslog[im]<=Mt,im] = b[mslog[im]<=Mt]*(mslog[im] - Mt[mslog[im]<=Mt]) +c[mslog[im]<=Mt]
    #==========================================

    #==========================================main sequence popesso
    if work=="popesso+2023":
        for iz in range(nz):
            sfrlog_ms[iz,:]=(-27.58+0.26*cosmo.age(z[iz]))+(4.95-0.04*cosmo.age(z[iz]))*mslog-0.2*mslog*mslog
    #==========================================

    return sfrlog_ms



def sfr_function(z,nz,sfrlog,nsfr):
    """
    Sargent et al. 2012
    """
    red_sample = np.array([0., 0.5, 1.1, 1.5, 2.1])
    phisfr_data = [np.loadtxt("Data/Sargent_LF_SFRF_data/SFRfunction_z=%.1f.txt"%red) for red in red_sample]
    phisfr_temp = np.array( [ interp1d(phisfr_data[i][:,0], phisfr_data[i][:,1], fill_value="extrapolate")(sfrlog) for i in range(red_sample.size) ] )
    phisfr=np.zeros((nsfr,nz))
    for isfr in range(nsfr):
        phisfr[isfr,:] = interp1d(red_sample, phisfr_temp[:,isfr], fill_value="extrapolate")(z)
    return 10.**phisfr



def IRLF(z,nz,logLs, nLs):
    """
    Fujimoto et al. 2023
    """
    red_sample = np.array([0.8, 1.5, 2.5, 3.5, 5., 6.75])
    alpha_sample = np.array([0.94, 0.94, 0.93, 1.04, 0.94, 0.94])
    beta_sample = np.array([3.72, 3.72, 3.72, 3.72, 3.72, 3.72])
    logphiS_sample = np.array([-3.65, -3.38, -3.86, -4.49, -4.77, -5.26])
    logLs_sample = np.array([12.15, 12.21, 12.52, 12.73, 12.72, 12.71])
    logLF=np.zeros((nLs,nz))
    logLF_temp=np.zeros((nLs,red_sample.size))
    for iz in range(red_sample.size):
        logLF_temp[:,iz] = np.log10( 10.**logphiS_sample[iz] / ( ( 10.**logLs/10.**logLs_sample[iz] )**alpha_sample[iz] + ( 10.**logLs/10.**logLs_sample[iz] )**beta_sample[iz] ) )
    for iL in range(nLs):
        logLF[iL,:] = interp1d(red_sample, logLF_temp[iL,:], fill_value="extrapolate")(z)
    return 10.**logLF,logLF



def compute_SFR_function_from_LF(nz, z, nLs, logLs, LF, nsfr, sfrlog, volume_mock, logLsmin):
    b=sfrlog[1]-sfrlog[0]
    phisfr=np.zeros((nsfr, nz))
    sfrlog_temp = np.arange(sfrlog[0]-b/2, sfrlog[-1]+b/2+0.1, b)
    #volume_mock = 550**3
    for iz in range(nz):
        logLs_mock = compute_objs_from_mass_function(logLs, LF[:,iz], volume_mock, mask=logLs>logLsmin) #[Lsun]
        #sfrlog_mock = np.log10(3.88e-44) + logLs_mock + logLsun #Murphy+2011
        sfrlog_mock = np.log10(4.55e-44) + logLs_mock + logLsun #Kennicutt+1998
        phisfr[:,iz] = np.histogram(sfrlog_mock, bins=sfrlog_temp)[0] / b / volume_mock
    return phisfr



"""
def sigma_Mstar_relation(mstarlog):
    sigma=np.zeros(mstarlog.size)
    logsigmab=2.073; mblog=10.26
    sigma[mstarlog<=mblog]=logsigmab+0.403*(mstarlog[mstarlog<=mblog]-mblog)
    sigma[mstarlog>mblog]=logsigmab+0.293*(mstarlog[mstarlog>mblog]-mblog)
    return sigma
"""



def assign_velocity_dispersion(nz,nhalo,mhlog_arr,logmstar_integrated,velocity_dispersion):
    if velocity_dispersion=="Ferrarese+2002":
        logVc_arr=np.log10(2.8e-2 * (10**mhlog_arr*h)**0.316) #[km/s] #Formula
        logsigma_arr=1.14*logVc_arr-0.534 #[km/s] #Ferrarese+2002
        #https://ned.ipac.caltech.edu/level5/March02/Ferrarese/Fer5_2.html
        #https://ui.adsabs.harvard.edu/abs/2002ApJ...578...90F/abstract
        #sigma_c normalized to an aperture of size 1/8 the bulge effective radius
    if velocity_dispersion=="Marsden+2022":
        mm=np.loadtxt("Data/Marsden_2022/sigma_evo.txt")
        marsden=interp2d(np.arange(9.5,11.51,0.5), np.arange(0,4.01,0.2), mm)
        logsigma_arr=np.array([[marsden(logmstar_integrated[igal,iz],z[iz])[0] for iz in range(nz)] for igal in range(nhalo)])
        logsigma_arr+=np.random.normal(0.,0.05,logsigma_arr.shape)
    #logsigma_arr = np.transpose([ sigma_Mstar_relation(logmstar_integrated[:,iz]) + np.random.normal(0,0.01,nhalo) for iz in range(nz) ])
    #logsigma_arr[:,z>3] = 100
    return logsigma_arr



def add_mergers_Decode(z,nz,mslog,logmstar_integrated,nhalo,mhlog_arr,sfrlog_cat):
    logMR_cat = compute_merger_rates(z,nz,mslog,logmstar_integrated,nhalo,mhlog_arr)
    logMR_cat[:,z>3]=-65.
    msdotlog_cat = np.log10( (1.-0.44) * 10.**sfrlog_cat + 0.8 * 10.** logMR_cat)
    mstar_integrated = np.flip( np.transpose( cumtrapz(np.flip(10.**np.transpose(msdotlog_cat)), np.flip(cosmo.age(z))*10.**9., axis=0, initial=0.) ) )
    return mstar_integrated, np.log10(mstar_integrated), logMR_cat



def add_mergers_SatGen_tree(nz,nhalo,mhlog_arr,logmstar_integrated,mstar_integrated,tree):
    mbin=0.1; Mhlog=np.arange(10,15.5,mbin)
    popts=[]
    for iz in range(nz):
        mean_smhm = np.array([ np.mean(logmstar_integrated[np.logical_and.reduce((logmstar_integrated[:,iz]>0., mhlog_arr[:,iz]>m-mbin/2., mhlog_arr[:,iz]<=m+mbin/2.)),iz]) for m in Mhlog ])
        try:
            popt,pcov=curve_fit(SMHM_double_pl, Mhlog[np.isfinite(mean_smhm)], mean_smhm[np.isfinite(mean_smhm)], p0 = [0.032,12.,2.,0.608])
        except:
            pass
        popts.append(popt)
    tree["mstar"]=np.array([ np.array([SMHM_double_pl(tree["mhalo"][ihalo][i],*popts[nearest(z,tree["zinfall"][ihalo][i])]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    merger_history = np.array([[ np.sum(10**tree["mstar"][ihalo][np.logical_and(tree["z_merge"][ihalo]>Z,tree["order"][ihalo]<=1)]) for Z in z] for ihalo in range(nhalo)])
    mstar_integrated += merger_history * 0.8
    tree["mratio"]=np.array([ np.array([ 10.**tree["mstar"][ihalo][i]/mstar_integrated[ihalo,nearest(z,tree["z_merge"][ihalo][i])] for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    return mstar_integrated, np.log10(mstar_integrated), tree



def f_gas_limit_check(z, Mstar):
    Mstar = 10**Mstar
    alpha = 0.59 * (1.+z)**0.45
    Mgas_Mstar = 0.04 * np.power(Mstar / (4.5*10**11), -alpha)
    f_gas = Mgas_Mstar / (Mgas_Mstar + 1.)
    if f_gas > 0.5:
        return True
    else:
        return False
def add_disc_instability(M_disc, z1,z2):
    dt = (cosmo.lookbackTime(z1) - cosmo.lookbackTime(z2)) * 1e9 #[yr]
    Mdot = 25. * (M_disc / 10**11) * np.power((1.+z2)/3., 1.5)
    return Mdot * dt



def evolve_one_bulge_disc(nz,z, tree_mstar,tree_z_merge,tree_mratio, mratio_threshold, logmstar_integrated_ihalo, sfrlog_cat_ihalo, f_discregrowth, add_disc_inst):
    Mbulge=np.zeros(nz)+1e-66
    Mdisc=np.zeros(nz)+1e-66
    for i in range(nz-1):
        iz1=nz-i-1; iz2=nz-i-2
        majormergers=tree_mstar[np.logical_and.reduce((tree_z_merge>=z[iz2],tree_z_merge<z[iz1],tree_z_merge>=mratio_threshold))]
        if majormergers.size>0:
            Mdisc[iz2]=f_discregrowth*10.**logmstar_integrated_ihalo[iz2]+1e-66
            Mbulge[iz2]=(1.-f_discregrowth)*10.**logmstar_integrated_ihalo[iz2]
        else:
            Mdisc[iz2] = Mdisc[iz1] + (1.-0.44)* (10.**sfrlog_cat_ihalo[iz1]+10.**sfrlog_cat_ihalo[iz2])/2. * (cosmo.age(z[iz2])-cosmo.age(z[iz1]))*1e9
            minormergers=tree_mstar[np.logical_and.reduce((tree_z_merge>=z[iz2],tree_z_merge<z[iz1],tree_z_merge<mratio_threshold))]
            Mbulge[iz2] = Mbulge[iz1] + np.sum(10.**minormergers)
        if add_disc_inst and f_gas_limit_check(z[iz2], logmstar_integrated_ihalo[iz2]):
            Mbulge[iz2] += add_disc_instability(Mdisc[iz2], z[iz1], z[iz2])
            Mdisc[iz2] -= add_disc_instability(Mdisc[iz2], z[iz1], z[iz2])
    return [Mbulge,Mdisc]
def form_evolve_bulge_disc(nz,z,nhalo,mhlog_arr,logmstar_integrated,mstar_integrated,tree,sfrlog_cat,mratio_threshold, f_discregrowth, add_disc_inst):
    res = Parallel(n_jobs=-1)(
    delayed( evolve_one_bulge_disc )( nz,z,tree["mstar"][ihalo],tree["z_merge"][ihalo],tree["mratio"][ihalo],mratio_threshold,logmstar_integrated[ihalo,:],sfrlog_cat[ihalo,:],f_discregrowth,add_disc_inst ) for ihalo in range(nhalo))
    mbulge = np.array([ list(res[ihalo][0]) for ihalo in range(nhalo) ])
    mdisc = np.array([ list(res[ihalo][1]) for ihalo in range(nhalo) ])
    return mbulge,mdisc,np.log10(mbulge),np.log10(mdisc)



def assign_init_size(z, Z, logmstar, f_quenched, Mslog, galaxy_quenched=0):
    if not bool(galaxy_quenched):
        prob_quenched=interp1d(Mslog, f_quenched[:,nearest(z,Z)], fill_value="extrapolate")(logmstar)
        random_number=np.random.uniform(0,1)
        if random_number<prob_quenched:
            galaxy_quenched=True
        else:
            galaxy_quenched=False
    #Suess+2019 relation
    if not galaxy_quenched:
        return 0.05*(logmstar-1.2)
    elif galaxy_quenched and Z>1.5:
        return 0.3*(logmstar-10.5)
    elif galaxy_quenched and Z<=1.5:
        return 0.6*(logmstar-10.5)

def update_size(M1,M2,R1,R2,f_orb=0.,c=0.45):
    # Shankar+2014
    return (M1+M2)**2 / ( M1**2/R1 + M2**2/R2 + f_orb/c*M1*M2/(R1+R2) )

def evolve_one_galaxy_size(nz,z,tree_mstar,tree_z_merge,tree_mratio,logmstar_integrated_ihalo,sfrlog_cat_ihalo,Mbulge,Mdisc,f_quenched,Mslog):
    # This function computes the size evolution of one single galaxy
    Rtot=np.zeros(nz)+1e-66
    Rb=np.zeros(nz)+1e-66
    Rd=np.zeros(nz)+1e-66
    for i in range(nz-1):
        iz1=nz-i-1; iz2=nz-i-2
        mergers=tree_mstar[np.logical_and(tree_z_merge>=z[iz2],tree_z_merge<z[iz1])]
        Rd[iz2]=10.**assign_init_size(z,z[iz2],np.log10(Mdisc[iz2]),f_quenched,Mslog,sfrlog_cat_ihalo[iz2]<-60.)
        if mergers.size>0:
            Rsat=10.**assign_init_size(z,z[iz2],np.max(mergers),f_quenched,Mslog)
            Rb[iz2]=update_size(Mbulge[iz2],10.**np.max(mergers),Rb[iz1],Rsat)
        else:
            Rb[iz2]=Rb[iz1]
        Rtot[iz2]= (Rb[iz2]*Mbulge[iz2] + Rd[iz2]*Mdisc[iz2]) / (Mbulge[iz2]+Mdisc[iz2])
    return [Rtot,Rb,Rd]

def evolve_sizes(nz,z,nhalo,mhlog_arr,logmstar_integrated,mstar_integrated,tree,sfrlog_cat,mbulge,mdisc,f_quenched,Mslog):
    # This function computes the size evolution of the galaxies in the catalogue
    res = Parallel(n_jobs=-1)(
    delayed( evolve_one_galaxy_size )( nz,z,tree["mstar"][ihalo],tree["z_merge"][ihalo],tree["mratio"][ihalo],logmstar_integrated[ihalo,:],sfrlog_cat[ihalo,:],mbulge[ihalo,:],mdisc[ihalo,:],f_quenched,Mslog )
    for ihalo in range(nhalo))
    Rtotal = np.array([ list(res[ihalo][0]) for ihalo in range(nhalo) ])
    Rbulge = np.array([ list(res[ihalo][1]) for ihalo in range(nhalo) ])
    Rdisc = np.array([ list(res[ihalo][2]) for ihalo in range(nhalo) ])
    return Rtotal,Rbulge,Rdisc,np.log10(Rtotal),np.log10(Rbulge),np.log10(Rdisc)
