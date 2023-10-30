from modules.my_functions import *



def dark_matter_halo_catalogue(reduced_catalogue,nz):
    if reduced_catalogue:
        nhalo=9908
        mhlog_arr=np.loadtxt("Dark_matter_halo_catalogue/mhlog_arr_z8_>11_reduced.txt").reshape(nhalo,nz)
        dmhdtlog_arr=np.loadtxt("Dark_matter_halo_catalogue/dmhdtlog_arr_z8_>11_reduced.txt").reshape(nhalo,nz)
    else:
        nhalo=100000
        mhlog_arr=np.loadtxt("Dark_matter_halo_catalogue/mhlog_arr_z8_>11.txt").reshape(nhalo,nz)
        dmhdtlog_arr=np.loadtxt("Dark_matter_halo_catalogue/dmhdtlog_arr_z8_>11.txt").reshape(nhalo,nz)
    return nhalo, mhlog_arr, dmhdtlog_arr



def halo_mass_function(z,nz,nhalo,mhlog_arr,mhlog,nmh,mhlog_min,sigma_am=0.):
    mh=10.**mhlog
    hmflog=np.zeros((nmh,nz))
    for iz in range(nz):
        hmflog[:,iz] = np.log10(mass_function.massFunction(mh*h,z[iz],mdef='sp-apr-mn',model = 'diemer20',q_out = 'dndlnM')*np.log(10.)*h**3.)
        #hmflog[:,iz] = np.log10(hgm.halo_galaxies_massfunc(mhlog,z[iz])*mh*np.log(10.))
    #hmflog[:,99]=hmflog[:,98]
    hmf=10.**hmflog

    if sigma_am==0.:
        hmf_cum=np.zeros((nmh,nz))
        for iz in range(nz):
            hmf_cum[:,iz]=trapz(hmf[:,iz],mhlog)-cumtrapz(hmf[:,iz],mhlog,initial=0.)
        hmf_cum[hmf_cum<0.]=10.**(-66.)
    elif sigma_am>0.:
        hmf_cum=np.zeros((nmh,nz))
        for iz in range(nz):
            for imh in range(nmh):
                hmf_cum[imh,iz]=trapz(hmf[:,iz]*0.5*special.erfc((mhlog[imh]-mhlog)/np.sqrt(2.)/sigma_am),mhlog)
        hmf_cum[hmf_cum<0.]=10.**(-66.)

    #questa norm è il numero di aloni reali sopra mhlog_min ad ogni z
    norm=np.zeros((nz))
    for iz in range(nz):
        norm[iz]=np.trapz(hmf[mhlog>mhlog_min,iz],mhlog[mhlog>mhlog_min])
    norm=hmf_cum[nearest(mhlog,mhlog_min),:]

    volume = nhalo / trapz(hmf[nearest(mhlog,11):,0], mhlog[nearest(mhlog,11):]) #Mpc^-3
    cube_side = np.cbrt(volume) #Mpc
    print("Volume cube side: %f Mpc"%cube_side)

    hmf_cat = np.zeros((nmh,nz))
    for iz in range(nz):
        hmf_cat[:,iz] = interp1d(mhlog[:-1]+(mhlog[1]-mhlog[0])/2., np.histogram(mhlog_arr[:,iz],bins=mhlog)[0]/(mhlog[1]-mhlog[0])/volume, fill_value="extrapolate")(mhlog)

    return hmf, np.log10(hmf), hmf_cum, norm, volume, cube_side, hmf_cat



def hmf_catalogue(z,nz,mhlog_arr,hmflog,mhlog,mhlog_min_cat):
    jcut=np.logical_and(mhlog>mhlog_min_cat,mhlog<16.)
    #questa norm è il numero di aloni sopra mhlog_min_cat a z=0. Serve a normalizzare gli istogrammi a z=0, cioè trasformare
    #i numeri del catalogo in N/Mpc^3
    #norm_histo=hmf_cum2[nearest(mhlog,mhlog_min_cat),0]
    norm_histo=np.trapz(10.**hmflog[jcut,0],mhlog[jcut])
    nmh_cat=1000
    mhlog_cat_min=5.#chekkare di non escludere troppi aloni che andando indietro nel tempo possono diventare piccoli
    mhlog_cat=np.linspace(3.,15.,nmh_cat)
    mh_cat=10.**mhlog_cat
    bins_widths=np.append(np.diff(mhlog_cat),np.array(mhlog_cat[1]-mhlog_cat[0]))
    bins=np.append(mhlog_cat-bins_widths/2.,mhlog_cat[-1]+bins_widths[-1]/2.)
    dpdmhlog_cat=np.zeros((nmh_cat,nz))
    dNdmhlog_cat=np.zeros((nmh_cat,nz))
    for iz in range(nz):
        dpdmhlog_cat[:,iz],bins=np.histogram(mhlog_arr[:,iz],bins=bins,density=True)
    dNdmhlog_cat=dpdmhlog_cat*norm_histo
    return mhlog_cat,dNdmhlog_cat, norm_histo



def hsar_function(z,nz,hsarlog,nhsar,hsarlog_cat,mhlog_arr,mhlog,nhalo,mhlog_min,hmf,mhlog_cat,dNdmhlog_cat,norm,norm_histo):
    bins_widths=np.append(np.diff(hsarlog),np.array(hsarlog[1]-hsarlog[0]))
    bins=np.append(hsarlog-bins_widths/2.,hsarlog[-1]+bins_widths[-1]/2.)
    dpdhsarlog=np.zeros((nhsar,nz))
    dpdhsarlog_temp=np.zeros((nhsar,nz))
    dpdhsarlog_norm=np.zeros((nz))
    dNdhsarlog=np.zeros((nhsar,nz))
    dNdhsarlog_cat=np.zeros((nhsar,nz))
    peso=np.zeros((nhalo,nz))
    peso_interp=np.zeros((nhalo,nz))
    jsoglia=np.zeros((nz,len(mhlog_arr[:,0])),dtype=bool)
    for iz in range(nz):
        jsoglia[iz,:]=mhlog_arr[:,iz]>mhlog_min
        peso[:,iz]=np.interp(mhlog_arr[:,iz],mhlog,hmf[:,iz])/np.interp(mhlog_arr[:,iz],mhlog_cat,dNdmhlog_cat[:,iz])
        #peso_interp[:,iz]=np.interp(mhlog_arr[:,iz],mhlog,peso[:,iz])
        dpdhsarlog_temp[:,iz],bins=np.histogram(hsarlog_cat[jsoglia[iz,:],iz],bins=bins,weights=peso[jsoglia[iz,:],iz])
        dpdhsarlog_norm[iz]=np.trapz(dpdhsarlog_temp[:,iz],hsarlog)
        dpdhsarlog[:,iz]=dpdhsarlog_temp[:,iz]/dpdhsarlog_norm[iz]
        dNdhsarlog_cat[:,iz]=dpdhsarlog[:,iz]*norm_histo
        dNdhsarlog[:,iz]=dpdhsarlog[:,iz]*norm[iz]
    return dNdhsarlog,jsoglia,bins,dpdhsarlog



def hsar_ac_function(z,nz,hsarlog,nhsar,hsarlog_cat,mhlog_arr,mhlog,nhalo,mhlog_min,hmf,mhlog_cat,dNdmhlog_cat,norm,norm_histo, mhlog_crit):
    bins_widths=np.append(np.diff(hsarlog),np.array(hsarlog[1]-hsarlog[0]))
    bins=np.append(hsarlog-bins_widths/2.,hsarlog[-1]+bins_widths[-1]/2.)
    dpdhsarlog=np.zeros((nhsar,nz))
    dpdhsarlog_temp=np.zeros((nhsar,nz))
    dpdhsarlog_norm=np.zeros((nz))
    dNdhsarlog=np.zeros((nhsar,nz))
    dNdhsarlog_cat=np.zeros((nhsar,nz))
    peso=np.zeros((nhalo,nz))
    peso_interp=np.zeros((nhalo,nz))
    jsoglia=np.zeros((nz,len(mhlog_arr[:,0])),dtype=bool)
    for iz in range(nz):
        jsoglia[iz,:]=np.logical_and(mhlog_arr[:,iz]>mhlog_min, mhlog_arr[:,iz]<mhlog_crit(z[iz]))
        peso[:,iz]=np.interp(mhlog_arr[:,iz],mhlog,hmf[:,iz])/np.interp(mhlog_arr[:,iz],mhlog_cat,dNdmhlog_cat[:,iz])
        #peso_interp[:,iz]=np.interp(mhlog_arr[:,iz],mhlog,peso[:,iz])
        dpdhsarlog_temp[:,iz],bins=np.histogram(hsarlog_cat[jsoglia[iz,:],iz],bins=bins,weights=peso[jsoglia[iz,:],iz])
        dpdhsarlog_norm[iz]=np.trapz(dpdhsarlog_temp[:,iz],hsarlog)
        dpdhsarlog[:,iz]=dpdhsarlog_temp[:,iz]/dpdhsarlog_norm[iz]
        dNdhsarlog_cat[:,iz]=dpdhsarlog[:,iz]*norm_histo
        dNdhsarlog[:,iz]=dpdhsarlog[:,iz]*norm[iz]
    return dNdhsarlog,jsoglia,bins,dpdhsarlog



def har_distrib_active(dpdlogmhdot,z,nz,mhlog,nmh,mhlog_crit):
    dpdlogmhdot_active=np.copy(dpdlogmhdot)
    for iz in range(nz):
        dpdlogmhdot_active[iz,np.where(mhlog>=mhlog_crit(z[iz])),:]=0.
    return dpdlogmhdot_active



def har_distribution(mhdotlog,nmhdot,hsarlog,nhsar,dpdhsarlog,z,nz,mhlog,nmh):
    dpdlogmhdot=np.zeros((nz,nmh,nmhdot))
    for iz in range(nz):
        for imh in range(nmh):
            dpdlogmhdot[iz,imh,:]=interp1d(hsarlog,dpdhsarlog[:,iz],bounds_error=False,fill_value='extrapolate')(mhdotlog-mhlog[imh])
    return dpdlogmhdot



def halo_mass_crit_quench_givenparams(z_crit, m, mhlogcrit0, halo_quenching):
    if not halo_quenching:
        mhlogcrit0=16.
    z=np.arange(0,10,0.05)
    mhlogcrit=np.zeros(z.size)
    mhlogcrit[z<z_crit]=mhlogcrit0
    mhlogcrit[z>=z_crit] = m * (z[z>=z_crit]-z_crit) + mhlogcrit0
    return interp1d(z,mhlogcrit)



def har_distrib_ac_pas(dpdlogmhdot,z,nz,mhlog,nmh,mhlog_crit):
    dpdlogmhdot_active=np.copy(dpdlogmhdot)
    for iz in range(nz):
        dpdlogmhdot_active[iz,np.where(mhlog>=mhlog_crit(z[iz])),:]=0.
    dpdlogmhdot_passive=dpdlogmhdot-dpdlogmhdot_active
    return dpdlogmhdot_active, dpdlogmhdot_passive



def har_function(z,nz,mhlog,nmh,nmhdot, hmf, dpdlogmhdot_active, dpdlogmhdot_passive, dpdlogmhdot):
    dNdVdlogmhdot_active=np.zeros((nz,nmhdot))
    dNdVdlogmhdot_passive=np.zeros((nz,nmhdot))
    dNdVdlogmhdot=np.zeros((nz,nmhdot))
    jmassa=mhlog>5.
    for iz in range(nz):
        dNdVdlogmhdot_active[iz,:]=np.trapz(hmf[jmassa,iz][:,None]*dpdlogmhdot_active[iz,jmassa,:], mhlog[jmassa], axis=0)
        dNdVdlogmhdot_passive[iz,:]=np.trapz(hmf[jmassa,iz][:,None]*dpdlogmhdot_passive[iz,jmassa,:],mhlog[jmassa], axis=0)
        dNdVdlogmhdot[iz,:]=np.trapz(hmf[jmassa,iz][:,None]*dpdlogmhdot[iz,jmassa,:],mhlog[jmassa], axis=0)
    return dNdVdlogmhdot_active, dNdVdlogmhdot_passive, dNdVdlogmhdot



def har_function_active(z,nz,mhlog,nmh,nmhdot, hmf, dpdlogmhdot_active):
    dNdVdlogmhdot_active=np.zeros((nz,nmhdot))
    jmassa=mhlog>5.
    for iz in range(nz):
        dNdVdlogmhdot_active[iz,:]=np.trapz(hmf[jmassa,iz][:,None]*dpdlogmhdot_active[iz,jmassa,:], mhlog[jmassa], axis=0)
    return dNdVdlogmhdot_active



def read_SatGen_merger_tree(reduced_catalogue,nz,z,nhalo,mhlog_arr):
    if reduced_catalogue:
        tree=np.load("Dark_matter_halo_catalogue/SatGen_merger_trees/SatGen_tree_reduced.npy", allow_pickle=True)
    else:
        tree=np.load("Dark_matter_halo_catalogue/SatGen_merger_trees/SatGen_tree.npy", allow_pickle=True)
    tree={"mhalo":tree[0,:], "zinfall":tree[1,:], "order":tree[2,:]}

    #x=np.array([ np.array([ cosmo.Om(tree["zinfall"][ihalo][i]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ]) -1.
    #tree["Dvir"]=18.*np.pi**2.+82.*x-39.*x**2.
    #HZ=np.array([ np.array([ cosmo.Hz(tree["zinfall"][ihalo][i]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    #sqrtDvir178=np.array([ np.array([ np.sqrt(tree["Dvir"][ihalo][i]/178.) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    #tree["tau_dyn"]=1.628 / h / sqrtDvir178 * (cosmo.H0/HZ)
    tree["tau_dyn"]=np.array([ np.array([ compute_dyn_friction_timescale(tree["zinfall"][ihalo][i]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Dynamical friction timescales computed.")
    tree["mhaloratio"]=np.array([ np.array([ 10.**(tree["mhalo"][ihalo][i]-mhlog_arr[ihalo,nearest(z,tree["zinfall"][ihalo][i])]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Mass ratios computed.")
    tree["fudge"]=0.00035035*tree["mhaloratio"]+0.65
    print("Fudge factor computed.")
    tree["orb_circ"]=np.array([ np.array([ np.random.normal(0.5,0.23) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["orb_energy"]=np.array([ np.array([ np.power(tree["orb_circ"][ihalo][i],2.17)/(1.-np.sqrt(1.-tree["orb_circ"][ihalo][i]**2.)) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["tau_merge"]=np.array([ np.array([ tree["tau_dyn"][ihalo][i]*0.9*tree["mhaloratio"][ihalo][i] / np.log(1.+tree["mhaloratio"][ihalo][i]) * np.exp(0.6*tree["orb_circ"][ihalo][i]) * np.power(tree["orb_energy"][ihalo][i],0.1) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Merging timescales computed")
    tree["age_merge"] = np.array([ np.array([ cosmo.age(tree["zinfall"][ihalo][i])+tree["tau_merge"][ihalo][i] for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["z_merge"] = np.array([ np.array([ -1. for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    age_today=cosmo.age(0.)
    for ihalo in range(nhalo):
        tree["z_merge"][ihalo][tree["age_merge"][ihalo]<age_today] = cosmo.age(tree["age_merge"][ihalo][tree["age_merge"][ihalo]<age_today], inverse=True)
    print("Redshift at merging computed.")

    return tree



def update_hsar_function(z,nz,hsarlog,nhsar,hsarlog_cat,mhlog_arr,mhlog,nhalo,mhlog_min,hmf,mhlog_cat,dNdmhlog_cat,norm,norm_histo,sfrlog_cat):
    bins_widths=np.append(np.diff(hsarlog),np.array(hsarlog[1]-hsarlog[0]))
    bins=np.append(hsarlog-bins_widths/2.,hsarlog[-1]+bins_widths[-1]/2.)
    dpdhsarlog=np.zeros((nhsar,nz))
    dpdhsarlog_temp=np.zeros((nhsar,nz))
    dpdhsarlog_norm=np.zeros((nz))
    dNdhsarlog=np.zeros((nhsar,nz))
    dNdhsarlog_cat=np.zeros((nhsar,nz))
    peso=np.zeros((nhalo,nz))
    peso_interp=np.zeros((nhalo,nz))
    jsoglia=np.zeros((nz,len(mhlog_arr[:,0])),dtype=bool)
    for iz in range(nz):
        jsoglia[iz,:]=np.logical_and(mhlog_arr[:,iz]>mhlog_min, sfrlog_cat[:,iz]>-65.)
        peso[:,iz]=np.interp(mhlog_arr[:,iz],mhlog,hmf[:,iz])/np.interp(mhlog_arr[:,iz],mhlog_cat,dNdmhlog_cat[:,iz])
        dpdhsarlog_temp[:,iz],bins=np.histogram(hsarlog_cat[jsoglia[iz,:],iz],bins=bins,weights=peso[jsoglia[iz,:],iz])
        dpdhsarlog_norm[iz]=np.trapz(dpdhsarlog_temp[:,iz],hsarlog)
        dpdhsarlog[:,iz]=dpdhsarlog_temp[:,iz]/dpdhsarlog_norm[iz]
        dNdhsarlog_cat[:,iz]=dpdhsarlog[:,iz]*norm_histo
        dNdhsarlog[:,iz]=dpdhsarlog[:,iz]*norm[iz]
    return dNdhsarlog,jsoglia,bins,dpdhsarlog
def update_SFR_HAR_abundance_matching(z,nz,hsarlog_cat,dmhdtlog_arr,mhlog_arr,mhlog,nhalo,mhlog_min,hmf,mhlog_cat,dNdmhlog_cat,norm,norm_histo,sfrlog_cat,phisfrLF,sigma_sfr):

    dNdhsarlog,jsoglia,bins,dpdhsarlog=update_hsar_function(z,nz,hsarlog,nhsar,hsarlog_cat,mhlog_arr,mhlog,nhalo,mhlog_min,hmf,mhlog_cat,dNdmhlog_cat,norm,norm_histo,sfrlog_cat)

    dpdlogmhdot = har_distribution(mhdotlog, nmhdot, hsarlog, nhsar, dpdhsarlog, z,nz,mhlog,nmh)

    dpdlogmhdot_active=np.copy(dpdlogmhdot)
    dpdlogmhdot_passive=dpdlogmhdot-dpdlogmhdot_active

    new_dNdVdlogmhdot_active, new_dNdVdlogmhdot_passive, new_dNdVdlogmhdot = har_function(z,nz,mhlog,nmh,nmhdot, hmf, dpdlogmhdot_active, dpdlogmhdot_passive, dpdlogmhdot)

    sfr_am, sfrlog_am = abundance_matching(z,nz,mhdotlog,nmhdot, phisfrLF,sfrlog,nsfr, new_dNdVdlogmhdot_active, sigma_sfr)
    for iz in range(nz):
        sfrlog_cat[sfrlog_cat[:,iz]>-65.,iz]=interp1d(mhdotlog,sfrlog_am[:,iz],bounds_error=False, fill_value='extrapolate')(dmhdtlog_arr[sfrlog_cat[:,iz]>-65.,iz]+9.)
    sfrlog_cat += np.random.normal(0,sigma_sfr,(nhalo,nz))

    mstar_integrated, logmstar_integrated = integrate_accretion_rates_across_time(z, sfrlog_cat+np.log10(1.-0.44))

    return new_dNdVdlogmhdot_active, sfr_am, sfrlog_am, sfrlog_cat, mstar_integrated, logmstar_integrated
