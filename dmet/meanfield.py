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
    ev = np.empty((ncells, nscsites, nscsites), dtype = complex)
    for i in range(ncells):
        ew[i], ev[i] = la.eigh(Fock[i] + vcor(i, True)[0])
    return ew, ev

def DiagUHF(Fock, vcor):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((2, ncells, nscsites))
    ev = np.empty((2, ncells, nscsites, nscsites), dtype = complex)
    for i in range(ncells):
        ew[0][i], ev[0][i] = la.eigh(Fock[i] + vcor(i, True)[0])
        ew[1][i], ev[1][i] = la.eigh(Fock[i] + vcor(i, True)[1])
    return ew, ev

def DiagBdG(Fock, vcor, mu):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((ncells, nscsites * 2))
    ev = np.empty((ncells, nscsites * 2, nscsites * 2), dtype = complex)
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

def HF(lattice, vcor, occ, restricted, mu0 = 0., beta = np.inf, ires = False):
    # ires: more results, eg. total energy, gap, ew, ev
    log.eassert(beta >= 0, "beta cannot be negative")
    Fock = lattice.getFock(kspace = True)
    if restricted:
        ew, ev = DiagRHF(Fock, vcor)
        ew = ew[np.newaxis,:]
        ev = ev[np.newaxis,:]
    else:
        ew, ev = DiagUHF(Fock, vcor)
    nelec = ew.size * occ # rhf: per spin  uhf: total nelec

    ew_sorted = np.sort(np.ravel(ew))
    if beta < np.inf:
        # finite temperature occupation, n is continuous
        opt = minimize(lambda x: (np.sum(fermi(ew, x, beta)) - nelec)**2, mu0, tol = 5e-6)
        mu = opt.x
        ewocc = fermi(ew, mu, beta)
    else:
        thr_degenerate = 1e-6
        log.warning("T=0, nelec is rounded to integer nelec = %d (original %.2f)", int(nelec), nelec)
        nelec = int(nelec)
        # we prefer not to change mu
        if np.sum(ew < mu0-thr_degenerate) <= nelec and np.sum(ew <= mu0 + thr_degenerate) >= nelec:
            mu = mu0
        else:
            # otherwise choose between homo and lumo
            mu = 0.5 * (ew_sorted[nelec-1] + ew_sorted[nelec])

        ewocc = 1. * (ew < mu0-thr_degenerate)
        nremain_elec = nelec - np.sum(ewocc)
        if nremain_elec > 0:
            # fractional occupation
            remain_orb = np.logical_and(ew <= mu + thr_degenerate, ew >= mu - thr_degenerate)
            ewocc += (float(nremain_elec) / np.sum(remain_orb)) * remain_orb

    rho = np.empty_like(ev)
    rhoT = np.empty_like(rho)
    for i in range(rho.shape[0]):  # spin
        for j in range(rho.shape[1]): # kpoints
            rho[i,j] = mdot(ev[i,j], np.diag(ewocc[i,j]), ev[i,j].T.conj())
        rhoT[i] = lattice.FFTtoT(rho[i])

    # obtain rho in real space
    if np.allclose(rhoT.imag, 0.):
        rhoT = rhoT.real

    FockT, H1T = lattice.getFock(kspace = False), lattice.getH1(kspace = False)
    if vcor.islocal():
        vcorT = vcor(0, kspace = False)
    else:
        vcorT = np.asarray(map(lambda i: vcor(i, kspace = False), range(lattice.ncells)))
    if rhoT.shape[0] == 1:
        rhoT = rhoT[0]
        E = np.sum((FockT+H1T)*rhoT)
        if vcor.islocal():
            E += np.sum(vcorT[0] * rhoT[0])
        else:
            E += np.sum(vcorT[:,0,:,:] * rhoT)
    else:
        E = 0.5 * np.sum((FockT+H1T)*(rhoT[0]+rhoT[1]))
        if vcor.islocal():
            E += 0.5 * np.sum(vcorT[0] * rhoT[0][0] + vcorT[1] * rhoT[1][0])
        else:
            E += 0.5 * np.sum(vcorT[:,0,:,:] * rhoT[0] + vcorT[:,1,:,:] * rhoT[1])

    if ires:
        homo, lumo = filter(lambda x: x < mu, ew_sorted)[-1], filter(lambda x: x > mu, ew_sorted)[0]
        res = {"gap": lumo - homo, "e": ew, "coef": ev}
        return rhoT, mu, E, res
    else:
        return rhoT, mu, E
