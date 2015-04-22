import numpy as np
import numpy.linalg as la
import itertools as it
from libdmet.utils.misc import mdot
import libdmet.utils.logger as log

def transform_trans_inv(basis, lattice, H):
    ncells = lattice.ncells
    nbasis = basis.shape[2]
    res = np.zeros((nbasis, nbasis))
    for i, j in it.product(range(ncells), repeat = 2):
        res += mdot(basis[i].T, H[lattice.substract(j,i)], basis[j])
    return res

def transform_local(basis, lattice, H):
    # assume H is (nscsites, nscsites)
    ncells = lattice.ncells
    nbasis = basis.shape[2]
    res = np.zeros((nbasis, nbasis))
    for i in range(ncells):
        res += mdot(basis[i].T, H, basis[i])
    return res

def transform_imp(basis, lattice, H):
    return mdot(basis[0].T, H, basis[0])

def transform_imp_env(basis, lattice, H):
    ncells = lattice.ncells
    nbasis = basis.shape[2]
    res = np.zeros((nbasis, nbasis))
    for i in range(ncells):
        # this is the proper way to do it. equivalently, we do symmetrization 0.5 * (res+res.T)
        #res += 0.5 * mdot(basis[0].T, H[i], basis[i])
        #res += 0.5 * mdot(basis[i].T, H[lattice.substract(0,i)], basis[0])
        res += mdot(basis[0].T, H[i], basis[i])
    res = 0.5 * (res + res.T)
    return res

def transform_4idx(vijkl, ip, jq, kr, ls):
    return np.swapaxes(np.tensordot(np.swapaxes(np.tensordot(np.swapaxes(np.tensordot(jq, \
        np.swapaxes(np.tensordot(ip, vijkl, axes = (0,0)),0,1), \
        axes = (0,0)),0,1), ls, axes = (3,0)),2,3), kr, axes = (3,0)),2,3)
