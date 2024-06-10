import sys,os,time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.interpolate import interp1d,CubicSpline
from scipy.integrate import trapz,cumtrapz
import joblib
from joblib import Parallel, delayed


import contextlib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()



satgenPath  = './SatGen/'
outPath = './modules/newModules/newData/FullCatalogues/'

sys.path.insert(1,satgenPath)
from FirstOrderOnly_TreeGen_Sub import *

np.random.seed(1234)
cosmo = cosmology.setCosmology('planck18')

def get_HMF(z,Mh):
    HMF = np.array([np.log10(mass_function.massFunction(10**Mh*cosmo.h,z[iz],mdef='sp-apr-mn',model='diemer20',q_out='dndlnM')*np.log(10.)*cosmo.h**3.) for iz in range(z.size)])
    return HMF


'''#############################################################################
                        Halo Catalogue Parameters
#############################################################################'''

'''
Test Case :
    1    z value : z0 = 0
    10^4 halos
    Halo Masses Between 11 and 16.5
    Mass resolution 1e-8
    Mass cut 10^9 Msun
'''

z0s = np.arange(0.01,6.51,0.5) ; nz0 = z0s.size
Mmin = 11.0 ; Mmax = 16.5 ; dM = 0.01  # Later put this inside the loop so that the mass limits can change with the initial z...
Mmins = np.linspace(11,7,nz0)
Mcuts = np.linspace(9,6,nz0)
nhalo = 10000

# Redshift array to map the output MAH onto if SatGen is inhomogeneous or to match the z values from diffmah catalogues
zmin = 0.01 ; zmax = 7.01 ; dz = 0.05 ; zarr = np.arange(zmin,zmax+dz,dz) ; nz = zarr.size
Use_zarr = False

ti = time.time()

for z0,iz in zip(z0s,range(nz0)):
    print(
    '''\n
    Starting z0={0}
    Time elapsed so far {1}s
    '''.format(round(z0,3),round(time.time()-ti))
    )
    outDir = outPath+'Nhalo_{0}e{1}/z{2:02.2f}/'.format(round(nhalo/10**int(np.log10(nhalo))),int(np.log10(nhalo)),z0).replace('.','_')
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    '''#############################################################################
                                Sample HMF
    #############################################################################'''

    print('Sampling HMF...')
    Marr = np.arange(Mmins[iz],Mmax+dM,dM)
    HMF = 10**get_HMF(np.atleast_1d(zmin),Marr)[0]
    CMF = cumtrapz(HMF,Marr,initial=0)/trapz(HMF,Marr)
    Mh0 = interp1d(CMF,Marr)(np.random.uniform(0,1,nhalo))


    '''#############################################################################
                            Generate Merger Trees with SatGen
    #############################################################################'''

    def get_tree(z0,M0,ihalo,zarr,Mres=1e-5,lgMresCut=10,MapToZarr=False,SaveFullTree=False):
        iter = 0
        while True:
            try:
                MAH,MT = genTree(z0,M0,res=Mres,lgMresCut=lgMresCut,SaveFullTree=SaveFullTree)
                break
            except:
                iter += 1
                if iter == 5:
                    print('ERROR: failed on ',ihalo)
                    MT = np.empty((2,1)) ; MT.fill(np.NaN) # return nans so ihalo still maps
                    MAH = np.empty((1,zarr.size));MAH.fill(np.NaN) # return nans so ihalo still maps
                    break
        return MT,MAH

    #results = []
    #[results.append(get_tree(z0,Mh0[ihalo],ihalo,zarr,Mres=1e-8,lgMresCut=7,MapToZarr=False)) for ihalo in tqdm(range(nhalo))]

    with tqdm_joblib(tqdm(desc='Running SatGen for z0={0} of {1} values, Nhalo={2}'.format(z0,nz0,nhalo),total=nhalo)) as ProgBar:
        results = Parallel(n_jobs=-1)(
        delayed(get_tree)(z0,Mh0[ihalo],ihalo,zarr,Mres=1e-8,lgMresCut=Mcuts[iz],MapToZarr=False) for ihalo in range(nhalo)
        )

    print('get_tree DONE : ',time.time()-ti)
    Nsubs = np.array([len(results[ihalo][0][0]) for ihalo in range(nhalo)])

    SatsInfall = np.empty((2,nhalo,max(Nsubs))) ; SatsInfall.fill(np.NaN) # Same as the previous merger tree files but without the order as by construction they are all 1
    for ihalo in range(nhalo):
        SatsInfall[0,ihalo,:Nsubs[ihalo]] = results[ihalo][0][0] # Infall mass of first order subhalos of central halo ihalo
        SatsInfall[1,ihalo,:Nsubs[ihalo]] = results[ihalo][0][1] # Infall redshift of first order subhalos of central halo ihalo

    zs = np.array([ results[ihalo][1][0] for ihalo in range(nhalo) ])
    if np.any(np.diff(zs,axis=0) > 0) or Use_zarr:
        print('Different z values - Remapping...')
        MAHs = np.array([interp1d(results[ihalo][1][0],results[ihalo][1][1],bounds_error=False,fill_value=np.NaN)(zarr) for ihalo in tqdm(range(nhalo))])
        MAHs = np.append(np.atleast_2d(zarr),MAHs,axis=0)
    else:
        print('Same z values - Combining...')
        MAHs = np.array([ results[ihalo][1][1] for ihalo in range(nhalo) ]) # Mass accretion histories of all centals at z=z0
        MAHs = np.append(np.atleast_2d(results[0][1][0]),MAHs,axis=0) # Include z values as the first row for ease

    print('Saving...')
    np.save(outDir+'SGMT.npy',SatsInfall)
    np.save(outDir+'SGMAHs.npy',MAHs)

    print('Calculating HARs...')
    HARs = []
    ztmp = cosmo.lookbackTime(np.arange(0,13.5,0.1),inverse=True)
    for MAH in MAHs[1:]:
        MAHtmp = interp1d(MAHs[0],10**MAH,fill_value='extrapolate')(ztmp)
        HARtmp = np.diff(MAHtmp)/np.diff(ztmp)
        HARs.append(interp1d((ztmp[1:]+ztmp[:-1])/2,HARtmp,fill_value='extrapolate')(MAHs[0]))
    HARs = np.array(HARs)
    HARs = np.append(np.atleast_2d(MAHs[0]),HARs,axis=0)

    print('Saving...')
    np.save(outDir+'SGHARs.npy',HARs)

print(
'''
############################################################################

Finished... z0 = {0}, Time Taken {1} s

############################################################################
'''.format(round(z0,3),round(time.time()-ti,4))
)
