#! /usr/bin/env python 


import numpy as np
import h5py

def load_dmet_iter_npy(fname = './dmet_iter.npy'):
    Mu, last_dmu, vcor_param, GRhoEmb, basis = np.load(fname)
    return Mu, last_dmu, vcor_param, GRhoEmb, basis

def load_dmet_npy(fname = './dmet.npy'):
    Mu, last_dmu, vcor_param, basis, GRhoEmb,\
            GRhoImp, EnergyImp, nelecImp = np.load(fname)
    return Mu, last_dmu, vcor_param, basis, GRhoEmb, \
            GRhoImp, EnergyImp, nelecImp

def get_order_param_3band(GRhoImp, AFM_idx = [0, 3, 9, 6]):
    limp = GRhoImp.shape[0] / 2

    AFM_idx_flatten = np.array(AFM_idx).flatten()
    rdm_a = GRhoImp[:limp,:limp][np.ix_(AFM_idx_flatten, AFM_idx_flatten)]
    rdm_b = np.eye(len(AFM_idx_flatten)) - GRhoImp[limp:,limp:][np.ix_(AFM_idx_flatten, AFM_idx_flatten)]
    rdm_ab = GRhoImp[:limp, limp:][np.ix_(AFM_idx_flatten, AFM_idx_flatten)]


    m0 = 0.5*(rdm_a[0,0]-rdm_b[0,0])
    m3 = 0.5*(rdm_a[3,3]-rdm_b[3,3])
    m1 = 0.5*(rdm_a[1,1]-rdm_b[1,1])
    m2 = 0.5*(rdm_a[2,2]-rdm_b[2,2])
    afm = 0.25*(m0+m3-m1-m2)

    s = 0.5**0.5
    d01 = s*(rdm_ab[0,1]+rdm_ab[1,0])
    d23 = s*(rdm_ab[2,3]+rdm_ab[3,2])
    d02 = s*(rdm_ab[0,2]+rdm_ab[2,0])
    d13 = s*(rdm_ab[1,3]+rdm_ab[3,1])
    dwv = 0.25*(d01+d23-d02-d13)

    return afm,dwv



if __name__ == '__main__':

    import sys
    import numpy as np
    if len(sys.argv) > 1 :
        fname = sys.argv[1]
    else:
        fname = './dmet.npy'

    Mu, last_dmu, vcor_param, basis, GRhoEmb, \
        GRhoImp, EnergyImp, nelecImp = load_dmet_npy(fname)

    np.set_printoptions(3, linewidth =1000, suppress = True)

    print get_order_param_3band(GRhoImp)

