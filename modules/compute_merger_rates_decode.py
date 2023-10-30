import numpy as np
from scipy.interpolate import interp1d
from modules.my_functions import *
from tqdm import tqdm


_z_=np.arange(0.,5.01,0.2)
_nz_=_z_.size
_mhlog_=np.arange(11.,15.6,0.2)
_nmh_=_mhlog_.size
cosmic_time=cosmo.age(_z_)



def read_decode_merger_tree(file_idx):
    IDpar = np.loadtxt("Dark_matter_halo_catalogue/Decode_merger_trees/halo_merger_tree_decode%d/data/output_parents.txt"%file_idx, usecols=0)
    Mpar = np.loadtxt("Dark_matter_halo_catalogue/Decode_merger_trees/halo_merger_tree_decode%d/data/output_parents.txt"%file_idx, usecols=1)
    IDsub = np.loadtxt("Dark_matter_halo_catalogue/Decode_merger_trees/halo_merger_tree_decode%d/data/output_mergers.txt"%file_idx, usecols=0)
    Msub = np.loadtxt("Dark_matter_halo_catalogue/Decode_merger_trees/halo_merger_tree_decode%d/data/output_mergers.txt"%file_idx, usecols=4)
    z_inf = np.loadtxt("Dark_matter_halo_catalogue/Decode_merger_trees/halo_merger_tree_decode%d/data/output_mergers.txt"%file_idx, usecols=5)
    tau_m = np.loadtxt("Dark_matter_halo_catalogue/Decode_merger_trees/halo_merger_tree_decode%d/data/output_mergers.txt"%file_idx, usecols=6)
    return IDpar, Mpar, IDsub, Msub, z_inf, tau_m


def compute_merger_rates_grid(z,nz,mslog,logmstar_integrated,mhlog_arr):

    sigmalogMR = np.zeros((_nz_,_nmh_))
    logMR = np.zeros((_nz_,_nmh_))

    smhm=[]; scatter=[]
    for iz in range(nz):
        b=0.1; mhlog_smhm=np.arange(10,15.5,b)
        mslog_smhm = np.array([ np.mean(logmstar_integrated[np.logical_and.reduce((logmstar_integrated[:,iz]>8., mhlog_arr[:,iz]>m-b/2., mhlog_arr[:,iz]<=m+b/2.)),iz]) for m in mhlog_smhm ])
        scatter_smhm = np.array([ np.std(logmstar_integrated[np.logical_and.reduce((logmstar_integrated[:,iz]>8., mhlog_arr[:,iz]>m-b/2., mhlog_arr[:,iz]<=m+b/2.)),iz]) for m in mhlog_smhm ])
        if z[iz]<=3. and mhlog_smhm[np.isfinite(mslog_smhm)].size>3:
            popt,pcov=curve_fit(SMHM_double_pl, mhlog_smhm[np.isfinite(mslog_smhm)], mslog_smhm[np.isfinite(mslog_smhm)], p0 = [0.032,12.,2.,0.608])
            smhm.append( interp1d(mhlog_smhm, SMHM_double_pl(mhlog_smhm,*popt), fill_value="extrapolate") )
        else:
            try:
                smhm.append( smhm[iz-1] )
            except:
                print(iz)
                print(logmstar_integrated[:,iz])
        scatter.append( np.mean(scatter_smhm[np.isfinite(scatter_smhm)]) )

    for imh in range(_nmh_):

        #print("ciao 1")
        if imh<5:
            #IDpar, Mpar, IDsub, Msub, z_inf, tau_m = read_decode_merger_tree(1)
            logMR[:,imh]=-66.
            continue
        if imh==5:
            IDpar, Mpar, IDsub, Msub, z_inf, tau_m = read_decode_merger_tree(2)
        if imh==10:
            IDpar, Mpar, IDsub, Msub, z_inf, tau_m = read_decode_merger_tree(3)
        if imh==13:
            IDpar, Mpar, IDsub, Msub, z_inf, tau_m = read_decode_merger_tree(4)
        if imh==16:
            IDpar, Mpar, IDsub, Msub, z_inf, tau_m = read_decode_merger_tree(5)
        if imh==18:
            IDpar, Mpar, IDsub, Msub, z_inf, tau_m = read_decode_merger_tree(6)

        #print("ciao 2")
        Mask = np.logical_and( _mhlog_[imh]-0.05<Mpar, Mpar<=_mhlog_[imh]+0.05 )
        #if imh<5:
        #    Mask = np.logical_and( _mhlog_[5]-0.05<Mpar, Mpar<=_mhlog_[5]+0.05 )
        IDthispar = IDpar[Mask].copy()
        Nhaloes=IDthispar.size

        #print("ciao 3")
        Nhaloesmax=200
        if Nhaloes>Nhaloesmax:
            prob=float(Nhaloesmax)/float(Nhaloes)
            p=np.random.uniform(0.,1.,Mask.size)
            Mask[np.logical_and(Mask==True,p>prob)]=False
            IDthispar = IDpar[Mask].copy()
            Nhaloes=IDthispar.size

        #print("ciao 4")
        mask = [ID in IDthispar for ID in IDsub]
        IDthissub = IDsub[mask].copy()
        Msub_ = Msub[mask].copy(); z_inf_ = z_inf[mask].copy(); tau_m_ = tau_m[mask].copy()
        age_at_merge = cosmo.age(z_inf_) + tau_m_
        z_at_merge = np.zeros(age_at_merge.size)
        z_at_merge[age_at_merge>cosmo.age(0.)] = -1.
        z_at_merge[age_at_merge<=cosmo.age(0.)] = cosmo.age(age_at_merge[age_at_merge<=cosmo.age(0.)], inverse=True)

        indexes=[nearest(z,z_inf_[i]) for i in range(z_inf_.size)]
        Msat = np.array( [ smhm[indexes[i]](Msub_[i]) + np.random.normal(0., scatter[indexes[i]]) for i in range(z_inf_.size) ] )
        #Msat = np.array( [ SMHM_matrix_v2(Msub_[i], smhm_matrix, z_array, z_inf_[i], 0.) for i in range(z_inf_.size) ] )

        Mmer = np.array( [ np.sum(10.**Msat[z_at_merge>_z_[iz]][np.isfinite(Msat[z_at_merge>_z_[iz]])]) / Nhaloes for iz in range(_nz_) ] )

        logMR[:-1,imh] = np.log10( (Mmer[:-1] - Mmer[1:]) * 1e-9 / ( cosmic_time[:-1] - cosmic_time[1:] ) )
        for iz in range(_nz_-1):
            x = [ np.log10( ( np.sum( 10.**Msat[ np.logical_and.reduce((IDthissub==ID, z_at_merge>_z_[iz], np.isfinite(Msat))) ] ) - np.sum( 10.**Msat[ np.logical_and.reduce((IDthissub==ID, z_at_merge>_z_[iz+1], np.isfinite(Msat))) ] ) ) * 1e-9 / ( cosmic_time[iz] - cosmic_time[iz+1] ) ) for ID in IDthispar ]
            sigmalogMR[iz,imh] = np.std( np.isfinite(x) )

    return logMR, sigmalogMR



def compute_merger_rates(z,nz,mslog,logmstar_integrated,nhalo,mhlog_arr):

    logMR, sigmalogMR = compute_merger_rates_grid(z,nz,mslog,logmstar_integrated,mhlog_arr)
    sigmalogMR = np.mean( [ np.mean(sigmalogMR[np.isfinite(sigmalogMR[:,imh]),imh]) for imh in range(_nmh_) ] )
    #print("Merger rates grid computed")
    if sigmalogMR>0.3:
        sigmalogMR=0.3

    logMR_cat=np.array([np.repeat(-66.,nz) for i in range(nhalo)])
    logMR_interp=[]
    for imh in range(_nmh_):
        logMR_interp.append( interp1d(_z_[np.isfinite(logMR[:,imh])], logMR[np.isfinite(logMR[:,imh]),imh]) )
    for imh in range(nhalo):
        for iz in range(nz):
            try:
                logMR_cat[imh,iz] = logMR_interp[nearest(_mhlog_,mhlog_arr[imh,0])](z[iz]) + np.random.normal(0,sigmalogMR)
            except:
                pass

    return logMR_cat
