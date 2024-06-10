import sys
import os
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import sys
from os import path

satgenPath  = './SatGen/'

sys.path.insert(1,satgenPath)
from TreeGen_Sub import *
import config as cfg
import cosmo as co
import init
from profiles import NFW
import aux

startTime = time.time()

def genTree(z0,lgM0,Ntree=1,optype='zzli',conctype='zhao',res=1e-3,lgMresCut=10,SaveFullTree=False):

    cfg.psi_res = res
    lgMres = lgM0 + np.log10(cfg.psi_res) # psi_{res} = 10^-5 by default
    lgMres = np.max([lgMres, lgMresCut])

    np.random.seed() # [important!] reseed the random number generator
    cfg.M0 = 10.**lgM0
    cfg.z0 = z0
    cfg.Mres = 10.**lgMres
    cfg.Mmin = 0.04*cfg.Mres

    # zmax and Nmax have been altered within config.py as it was hit and miss as to whether the  code recognised it properly

    k = 0               # the level, k, of the branch being considered
    ik = 0              # how many level-k branches have been finished
    Nk = 1              # total number of level-k branches
    Nbranch = 1         # total number of branches in the current tree

    Mak = [cfg.M0]      # accretion masses of level-k branches
    zak = [cfg.z0]
    idk = [0]           # branch ids of level-k branches
    ipk = [-1]          # parent ids of level-k branches (-1: no parent)

    Mak_tmp = []
    zak_tmp = []
    idk_tmp = []
    ipk_tmp = []

    mass = np.zeros((cfg.Nmax,cfg.Nz)) - 99.
    order = np.zeros((cfg.Nmax,cfg.Nz),np.int8) - 99
    ParentID = np.zeros((cfg.Nmax,cfg.Nz),np.int16) - 99

    VirialRadius = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    concentration = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.

    coordinates = np.zeros((cfg.Nmax,cfg.Nz,6),np.float32)

    while True:
        M = [Mak[ik]]   # mass history of current branch in fine timestep
        z = [zak[ik]]   # the redshifts of the mass history
        cfg.M0 = Mak[ik]# descendent mass
        cfg.z0 = zak[ik]# descendent redshift
        id = idk[ik]    # branch id
        ip = ipk[ik]    # parent id

        while cfg.M0>cfg.Mmin:

            if cfg.M0>=cfg.Mres: zleaf = cfg.z0 # update leaf redshift

            co.UpdateGlobalVariables(**cfg.cosmo)
            M1,M2,Np = co.DrawProgenitors(**cfg.cosmo)

            # update descendent halo mass and descendent redshift
            cfg.M0 = M1
            cfg.z0 = cfg.zW_interp(cfg.W0+cfg.dW)
            if cfg.z0>cfg.zmax: break

            if Np>1 and cfg.M0>=cfg.Mres: # register next-level branches

                Nbranch += 1
                Mak_tmp.append(M2)
                zak_tmp.append(cfg.z0)
                idk_tmp.append(Nbranch)
                ipk_tmp.append(id)

            # record the mass history at the original time resolution
            M.append(cfg.M0)
            z.append(cfg.z0)

        # Now that a branch is fully grown, do some book-keeping

        # convert mass-history list to array
        M = np.array(M)
        z = np.array(z)

        # downsample the fine-step mass history, M(z), onto the
        # coarser output timesteps, cfg.zsample
        Msample,zsample = aux.downsample(M,z,cfg.zsample)
        iz = aux.FindClosestIndices(cfg.zsample,zsample)
        if(isinstance(iz,np.int64)):
            iz = np.array([iz]) # avoids error in loop below
            zsample = np.array([zsample])
            Msample = np.array([Msample])
        izleaf = aux.FindNearestIndex(cfg.zsample,zleaf)
        # Note: zsample[j] is same as cfg.zsample[iz[j]]

        if k == 0:
            break # If computed the first order for the central then break

    #MAH = np.array([zsample,Msample])
    ''' Msample is the MAH of the central and zsample is the corresponding redshifts which are a subset of cfg.zsample '''
    ''' We use cfg.zsample such that they are the same for all halos at this init z and therefore do not need remapping '''
    MAH = np.empty((2,cfg.zsample.size));MAH.fill(-99.)
    MAH[0]=cfg.zsample ; MAH[1,np.isin(cfg.zsample,zsample)] = np.log10(Msample)
    ''' Mak_tmp is the infall mass and zak_tmp is the infall redshift '''
    MT = np.array([np.log10(np.array(Mak_tmp)),np.array(zak_tmp)])
    return MAH,MT
