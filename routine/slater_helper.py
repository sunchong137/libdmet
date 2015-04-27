import numpy as np
import numpy.linalg as la
import itertools as it
from libdmet.utils.misc import mdot, find
import libdmet.utils.logger as log

def transform_trans_inv(basis, lattice, H, symmetric = True):
    ncells = lattice.ncells
    nbasis = basis.shape[2]
    res = np.zeros((nbasis, nbasis))
    if symmetric:
        for i in range(ncells):
            res += mdot(basis[i].T, H[0], basis[i])
        for i, j in it.combinations(range(ncells), 2):
            temp = mdot(basis[i].T, H[lattice.substract(j,i)], basis[j])
            res += temp + temp.T
    else:
        for i, j in it.product(range(ncells), repeat = 2):
            res += mdot(basis[i].T, H[lattice.substract(j,i)], basis[j])
    return res

def transform_trans_inv_sparse(basis, lattice, H, symmetric = True, thr = 1e-7):
    ncells = lattice.ncells
    nbasis = basis.shape[2]
    res = np.zeros((nbasis, nbasis))
    mask_basis = find(True, map(lambda basis: la.norm(basis) > thr, basis))
    mask_H = find(True, map(lambda basis: la.norm(basis) > thr, H))
    if symmetric:
        for i in mask_basis:
            res += mdot(basis[i].T, H[0], basis[i])
        for i, j in it.combinations(mask_basis, 2):
            Hidx = lattice.substract(j,i)
            if Hidx in mask_H:
                temp = mdot(basis[i].T, H[Hidx], basis[j])
                res += temp + temp.T
    else:
        for i, j in it.product(mask_basis, repeat = 2):
            Hidx = lattice.substract(j,i)
            if Hidx in mask_H:
                res += mdot(basis[i].T, H[Hidx], basis[j])
    return res

def transform_local(basis, lattice, H):
    # assume H is (nscsites, nscsites)
    ncells = lattice.ncells
    nbasis = basis.shape[2]
    res = np.zeros((nbasis, nbasis))
    for i in range(ncells):
        res += mdot(basis[i].T, H, basis[i])
    return res

def transform_local_sparse(basis, lattice, H, thr = 1e-7):
    # assume H is (nscsites, nscsites)
    ncells = lattice.ncells
    nbasis = basis.shape[2]
    res = np.zeros((nbasis, nbasis))
    mask_basis = find(True, map(lambda basis: la.norm(basis) > thr, basis))
    for i in mask_basis:
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
