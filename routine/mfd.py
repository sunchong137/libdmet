# only implement non-scf routines
# restricted/unrestricted
# Slater/BCS
# thermal occupation

import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log
from libdmet.utils.misc import mdot
from libdmet.routine.bcs_helper import extractRdm
from scipy.optimize import minimize

# thermal occupation
# input in kspace
# return density matrices in kspace, chemical potential


def DiagRHF(Fock, vcor):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((ncells, nscsites))
    ev = np.empty((ncells, nscsites, nscsites), dtype = complex)
    if vcor is None:
        for i in range(ncells):
            ew[i], ev[i] = la.eigh(Fock[i])
    else:
        for i in range(ncells):
            ew[i], ev[i] = la.eigh(Fock[i] + vcor.get(i, True)[0])
    return ew, ev

def DiagUHF(Fock, vcor):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((2, ncells, nscsites))
    ev = np.empty((2, ncells, nscsites, nscsites), dtype = complex)
    if vcor is None:
        for i in range(ncells):
            ew[0][i], ev[0][i] = la.eigh(Fock[i])
            ew[1][i], ev[1][i] = la.eigh(Fock[i])
    else:
        for i in range(ncells):
            ew[0][i], ev[0][i] = la.eigh(Fock[i] + vcor.get(i, True)[0])
            ew[1][i], ev[1][i] = la.eigh(Fock[i] + vcor.get(i, True)[1])
    return ew, ev

def DiagBdG(Fock, vcor, mu):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((ncells, nscsites * 2))
    ev = np.empty((ncells, nscsites * 2, nscsites * 2), dtype = complex)
    temp = np.empty((nscsites * 2, nscsites * 2), dtype = complex)

    for i in range(ncells):
        temp[:nscsites, :nscsites] = Fock[i] + vcor.get(i, True)[0] - mu * np.eye(nscsites)
        temp[nscsites:, nscsites:] = -Fock[i] - vcor.get(i, True)[1] + mu * np.eye(nscsites)
        temp[:nscsites, nscsites:] = vcor.get(i, True)[2]
        temp[nscsites:, :nscsites] = vcor.get(i, True)[2].T
        ew[i], ev[i] = la.eigh(temp)
    return ew, ev

def DiagBdGsymm(Fock, vcor, mu, lattice):
    ncells = Fock.shape[0]
    nscsites = Fock.shape[1]
    ew = np.empty((ncells, nscsites * 2))
    ev = np.empty((ncells, nscsites * 2, nscsites * 2), dtype = complex)
    temp = np.empty((nscsites * 2, nscsites * 2), dtype = complex)

    computed = set()
    for i in range(ncells):
        neg_i = lattice.cell_pos2idx(-lattice.cell_idx2pos(i))
        if neg_i in computed:
            ew[i], ev[i] = ew[neg_i], ev[neg_i].conj()
        else:
            temp[:nscsites, :nscsites] = Fock[i] + vcor.get(i, True)[0] - mu * np.eye(nscsites)
            temp[nscsites:, nscsites:] = -Fock[i] - vcor.get(i, True)[1] + mu * np.eye(nscsites)
            temp[:nscsites, nscsites:] = vcor.get(i, True)[2]
            temp[nscsites:, :nscsites] = vcor.get(i, True)[2].T
            ew[i], ev[i] = la.eigh(temp)
            computed.add(i)
    return ew, ev

def HF(lattice, vcor, occ, restricted, mu0 = 0., beta = np.inf, ires = False):
    # ires: more results, eg. total energy, gap, ew, ev
    log.eassert(beta >= 0, "beta cannot be negative")
    Fock = lattice.getFock(kspace = True)
    if restricted:
        log.info("restricted Hartree-Fock")
        ew, ev = DiagRHF(Fock, vcor)
        ew = ew[np.newaxis,:]
        ev = ev[np.newaxis,:]
    else:
        log.info("unrestricted Hartree-Fock")
        ew, ev = DiagUHF(Fock, vcor)
    nelec = ew.size * occ # rhf: per spin  uhf: total nelec
    ew_sorted = np.sort(np.ravel(ew))
    ewocc, mu, nerr = assignocc(ew, nelec, beta, mu0)
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
        vcorT = vcor.get(0, kspace = False)
    else:
        vcorT = np.asarray(map(lambda i: vcor.get(i, kspace = False), range(lattice.ncells)))
    if rhoT.shape[0] == 1:
        E = np.sum((FockT+H1T)*rhoT[0])
        if vcor.islocal():
            E += np.sum(vcorT[0] * rhoT[0][0])
        else:
            E += np.sum(vcorT[:,0,:,:] * rhoT[0])
    else:
        E = 0.5 * np.sum((FockT+H1T)*(rhoT[0]+rhoT[1]))
        if vcor.islocal():
            E += 0.5 * np.sum(vcorT[0] * rhoT[0][0] + vcorT[1] * rhoT[1][0])
        else:
            E += 0.5 * np.sum(vcorT[:,0] * rhoT[0] + vcorT[:,1] * rhoT[1])
    if ires:
        homo, lumo = filter(lambda x: x <= mu, ew_sorted)[-1], filter(lambda x: x >= mu, ew_sorted)[0]
        res = {"gap": lumo - homo, "e": ew, "coef": ev, "nerr": nerr}
        return rhoT, mu, E, res
    else:
        return rhoT, mu, E

def HFB(lattice, vcor, restricted, mu = 0., beta = np.inf, ires = False):
    log.eassert(beta >= 0, "beta cannot be negative")
    Fock = lattice.getFock(kspace = True)
    if restricted:
        log.error("restricted Hartree-Fock-Bogoliubov not implemented")
    else:
        log.debug(1, "unrestricted Hartree-Fock-Bogoliubov")
        # use inversion symmetry F(k) = F(-k)^*
        #ew, ev = DiagBdGsymm(Fock, vcor, mu, lattice)
        ew, ev = DiagBdG(Fock, vcor, mu)

    ewocc = 1 * (ew < 0.)
    #nocc = np.sum(ewocc)
    # FIXME should it be a warning or an error?
    #log.check(nocc*2 == ew.size, \
    #        "number of negative and positive modes are not equal," \
    #        "the difference is %d, this means total spin on lattice is nonzero", \
    #        nocc*2 - ew.size)

    GRho = np.empty_like(ev) # k-space
    GRhoT = np.empty_like(GRho) # real space
    for i in range(GRho.shape[0]): # kpoints
        nocc = np.sum(ewocc[i])
        GRho[i] = np.dot(ev[i,:,:nocc], ev[i,:,:nocc].T.conj())
    GRhoT = lattice.FFTtoT(GRho)

    if np.allclose(GRhoT.imag, 0.):
        GRhoT = GRhoT.real

    FockT, H1T = lattice.getFock(kspace = False), lattice.getH1(kspace = False)
    if vcor.islocal():
        vcorT = vcor.get(0, kspace = False)
    else:
        vcorT = np.asarray(map(lambda i: vcor.get(i, kspace = False), range(lattice.ncells)))

    rhoTA, rhoTB, kappaTBA = np.swapaxes(np.asarray(map(extractRdm, GRhoT)), 0, 1)
    for c in range(1, rhoTB.shape[0]):
        rhoTB[c] -= np.eye(rhoTB.shape[1])

    n = np.trace(rhoTA[0]) + np.trace(rhoTB[0])
    E = 0.5 * np.sum((FockT+H1T) * (rhoTA + rhoTB))
    if vcor.islocal():
        E += 0.5 * np.sum(vcorT[0] * rhoTA[0] + vcorT[1] * rhoTB[0] + \
                2 * vcorT[2] * kappaTBA[0])
    else:
        E += 0.5 * np.sum(vcorT[:,0] * rhoTA + vcorT[:,1] * rhoTB + \
                2 * vcorT[:,2] * kappaTBA)

    if ires:
        res = {"gap": np.min(abs(ew)), "e": ew, "coef": ev}
        return GRhoT, n, E, res
    else:
        return GRhoT, n, E

def fermi(e, mu, beta):
    return 1./(1.+np.exp(beta * (e-mu)))

def assignocc(ew, nelec, beta, mu0):
    ew_sorted = np.sort(np.ravel(ew))
    if beta < np.inf:
        log.info("thermal occupation T=%10.5f", 1./beta)
        opt = minimize(lambda x: (np.sum(fermi(ew,x,beta)) - nelec)**2, mu0, tol = 5e-6)
        mu = opt.x
        nerr = abs(np.sum(fermi(ew, mu, beta)) - nelec)
        ewocc = fermi(ew, mu, beta)
    else:
        thr_degenerate = 1e-6
        if (nelec - int(nelec)) > 1e-5:
            log.warning("T=0, nelec is rounded to integer nelec = %d (original %.2f)", int(nelec), nelec)
        nelec = int(nelec)
        # we prefer not to change mu
        if np.sum(ew < mu0-thr_degenerate) <= nelec and np.sum(ew <= mu0 + thr_degenerate) >= nelec:
            mu = mu0
        else:
            # otherwise choose between homo and lumo
            mu = 0.5 * (ew_sorted[nelec-1] + ew_sorted[nelec])

        ewocc = 1. * (ew < mu - thr_degenerate)
        nremain_elec = nelec - np.sum(ewocc)
        if nremain_elec > 0:
            # fractional occupation
            remain_orb = np.logical_and(ew <= mu + thr_degenerate, ew >= mu - thr_degenerate)
            nremain_orb = np.sum(remain_orb)
            log.warning("degenerate HOMO-LUMO, assign fractional occupation\n"
                "%d electrons assigned to %d orbitals", nremain_elec, nremain_orb)
            ewocc += (float(nremain_elec) / nremain_orb) * remain_orb
        nerr = 0.
    return ewocc, mu, nerr

