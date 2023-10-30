import numpy as np
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology("planck18")
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("SatGen/")
from TreeGen_Sub import *

nz=50
z=np.linspace(0.01,5.01,nz)
#nhalo=100000
#mhlog_arr=np.loadtxt("../Method2/mhlog_arr_z8_>11.txt").reshape(nhalo,nz)
nhalo=9908
mhlog_arr=np.loadtxt("mhlog_arr_z8_>11_reduced.txt").reshape(nhalo,nz)

output_file="SatGen_merger_trees/SatGen_tree_reduced.npy"

#create merger tree using Satgen
subhaloes=[]
infall_zs=[]
orders=[]
merger_histories=[]
for ihalo in tqdm(range(nhalo)):
    #red, mass, order=loop(z,mhlog_arr[ihalo,:])
    try:
        red, mass, order=loop(z,mhlog_arr[ihalo,:])
    except:
        subhaloes.append(np.zeros(10)-99.)
        infall_zs.append(np.zeros(10))
        orders.append(np.zeros(10)+10)
        merger_histories.append(np.zeros(z.size)-99.)
    mass=np.log10(mass)
    subhaloes.append(np.array([]))
    infall_zs.append(np.array([]))
    orders.append(np.array([]))
    for i in range(mass[:,0].size-1):
        if mass[i+1,np.isfinite(mass[i+1,:])].size>0:
            idx=np.nanargmax(mass[i+1,:])
            subhaloes[ihalo]=np.append(subhaloes[ihalo], mass[i+1,idx])
            infall_zs[ihalo]=np.append(infall_zs[ihalo], red[idx])
            orders[ihalo]=np.append(orders[ihalo], order[i+1,idx])
    merger_histories.append( np.array([ np.log10(np.sum(10**subhaloes[ihalo][np.logical_and(infall_zs[ihalo]>Z,orders[ihalo]<=1)]) ) for Z in z]) )

#np.save(output_file, [np.array(subhaloes), np.array(infall_zs), np.array(orders), merger_histories])
