# only implement non-scf routines
# restricted/unrestricted
# Slater/BCS
# thermal occupation

import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log
from libdmet.utils.misc import mdot
from scipy.optimize import minimize

# thermal occupation
# input in kspace
# return density matrices in kspace, chemical potential


def linearFit():
    pass

def DiagRHF(Fock, vcor):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((ncells, nscsites))
    ev = np.empty((ncells, nscsites, nscsites))
    for i in range(ncells):
        ew[i], ev[i] = la.eigh(Fock[i] + vcor(i, True)[0])
    return ew, ev

def DiagUHF(Fock, vcor):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((2, ncells, nscsites))
    ev = np.empty((2, ncells, nscsites, nscsites))
    for i in range(ncells):
        ew[0][i], ev[0][i] = la.eigh(Fock[i] + vcor(i, True)[0])
        ew[1][i], ev[1][i] = la.eigh(Fock[i] + vcor(i, True)[1])
    return ew, ev

def DiagBdG(Fock, vcor, mu):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((ncells, nscsites * 2))
    ev = np.empty((ncells, nscsites * 2, nscsites * 2))
    temp = np.empty(nscsites * 2, nscsites * 2)
    
    for i in range(ncells):
        temp[:nscsites, :nscsites] = Fock[0][i] + vcor(i, True)[0] - mu * np.eye(nscsites)
        temp[nscsites:, nscsites:] = -H1[1][i] - vcor(i, True)[1] + mu * np.eye(nscsites)
        temp[:nscsites, nscsites:] = vcor(i, True)[2]
        temp[nscsites:, :nscsites] = vcor(i, True)[2].T
        ew[i], ev[i] = la.eigh(temp[i])
    return ew, ev

def fermi(e, mu, beta):
    return 1./(1.+np.exp(beta * (e-mu)))


def HF(Fock, vcor, occ, restricted, mu0 = 0., beta = np.inf):
    log.eassert(beta >= 0, "beta cannot be negative")
    if restricted:
        ew, ev = DiagRHF(Fock, vcor)
        ew = ew[np.newaxis,:]
        ev = ev[np.newaxis,:]
    else:
        ew, ev = DiagUHF(Fock, vcor)
    nelec = ew.size * occ
    if beta < np.inf:
        # finite temperature occupation, n is continuous
        mu = minimize(lambda x: (np.sum(fermi(ew, x, beta)) - nelec)**2, mu0, tol = 5e-6)
        ewocc = fermi(ew, x, beta)
    else:
        res = 1e-6
        nelec = int(nelec)
        log.warning("T=0, nelec is rounded to integer nelec = %d", nelec)
        # we prefer not to change mu
        if np.sum(ew < mu0-res) <= nelec and np.sum(ew <= mu0 + res) >= nelec:
            mu = mu0
        else:
            ew_sorted = np.sort(np.ravel(a))
            # otherwise choose between homo and lumo
            mu = 0.5 * (ew_sorted[nelec-1] + ew_sorted[nelec])
        
        ewocc = 1. * (ew < mu0-res) 
        nremain_elec = nelec - np.sum(ewocc)
        if nremain_elec > 0:
            # fractional occupation
            remain_orb = np.logical_and(ew <= mu + res, ew >= mu - res)
            ewocc += (float(nremain_elec) / np.sum(remain_orb)) * remain_orb

    rho = np.empty_like(ev)
    for i, j in range(rho.shape[0], rho.shape[1]): # spin and kpoints
        rho[i,j] = mdot(ew[i,j], np.diag(ewocc[i,j]), ew[i,j].T)

    if rho.shape[0] == 1:
        return rho.reshape(rho.shape[1:]), mu
    else:
        return rho, mu
