import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log
from slater_helper import *
from tempfile import NamedTemporaryFile
from libdmet.system import integral
from fit import minimize
from mfd import assignocc, HF
from math import sqrt
from copy import deepcopy

TmpDir = "/tmp"

def MatSqrt(M):
    log.eassert(la.norm(M - M.T.conj()) < 1e-10, "matrix must be symmetric")
    ew, ev = la.eigh(M)
    log.eassert((ew >= 0).all(), "matrix must be positive definite")
    log.check(ew[0] > 1e-10, "small eigenvalue for rho_imp,"
        "cut-off is recommended\nthe first 5 eigenvalues are %s", ew[:5])
    ewsq = np.sqrt(ew).real
    return np.dot(ev, np.diag(ewsq))

def normalizeBasis(b):
    # simple array
    ovlp = np.dot(b.T, b)
    log.debug(1, "basis overlap is\n%s", ovlp)
    n = np.diag(1./np.sqrt(np.diag(ovlp)).real)
    return np.dot(b, n)

def normalizeBasis1(b):
    # array in blocks
    ovlp = np.tensordot(b, b, axes = ((0,1), (0,1)))
    log.debug(1, "basis overlap is\n%s", ovlp)
    log.debug(0, "basis norm is\n%s", np.diag(ovlp))
    norms = np.diag(1./np.sqrt(np.diag(ovlp)))
    return np.tensordot(b, norms, axes = (2,0))

def embBasis(lattice, rho, local = True, **kwargs):
    if local:
        return __embBasis_proj(lattice, rho, **kwargs)
    else:
        return __embBasis_phsymm(lattice, rho, *kwargs)

def __embBasis_proj(lattice, rho, **kwargs):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    spin = rho.shape[0]
    basis = np.zeros((spin, ncells, nscsites, nscsites * 2))
    for s in range(spin):
        A = MatSqrt(rho[s,0])
        #B1 = np.swapaxes(rho[s,1:], 0, 1).reshape((nscsites, -1))
        #B1 = np.dot(la.inv(A), B1).T
        #B1 = normalizeBasis(B1)
        B = np.swapaxes(np.tensordot(la.inv(A), rho[s], axes = (1,1)), 0, 1)[1:]
        B = np.swapaxes(B, 1, 2)
        B = normalizeBasis1(B)
        basis[s, 0, :, :nscsites] = np.eye(nscsites)
        basis[s, 1:, :, nscsites:] = B
    return basis

def __embBasis_phsymm(lattice, rho, **kwargs):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    spin = rho.shape[0]
    basis = np.empty((spin, ncells, nscsites, nscsites * 2))
    for s in range(spin):
        # particle
        A1 = MatSqrt(rho[s,0])
        AB1 = np.swapaxes(np.tensordot(la.inv(A1), rho[s], axes = (1,1)), 0, 1)
        AB1 = np.swapaxes(AB1, 1, 2)
        AB1 = normalizeBasis1(AB1)
        # hole
        rho_h = -rho[s]
        rho_h[0] += np.eye(nscsites)
        A2 = MatSqrt(rho_h[0])
        AB2 = np.swapaxes(np.tensordot(la.inv(A2), rho_h, axes = (1,1)), 0, 1)
        AB2 = np.swapaxes(AB2, 1, 2)
        AB2 = normalizeBasis1(AB2)
        basis[s,:,:,:nscsites] = AB1
        basis[s,:,:,nscsites:] = AB2
    return basis

def embHam(lattice, basis, vcor, local = True, **kwargs):
    log.info("One-body part")
    Int1e, Int1e_energy = __embHam1e(lattice, basis, vcor, **kwargs)
    log.info("Two-body part")
    Int2e = __embHam2e(lattice, basis, vcor, local, **kwargs)

    spin = basis.shape[0]
    nbasis = basis.shape[3]
    return integral.Integral(nbasis, spin == 1, False, 0, {"cd": Int1e}, {"ccdd": Int2e}), {"cd": Int1e_energy}

def __embHam1e(lattice, basis, vcor, **kwargs):
    log.eassert(vcor.islocal(), "nonlocal correlation potential cannot be treated in this routine")
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[3]
    latFock = lattice.getFock(kspace = False)
    latH1 = lattice.getH1(kspace = False)
    ImpJK = lattice.getImpJK()
    spin = basis.shape[0]
    H1 = np.empty((spin, nbasis, nbasis))
    H1energy = np.empty((spin, nbasis, nbasis))

    for s in range(spin):
        log.debug(0, "Spin Component %d of %d", s, spin)
        # Fock part first
        log.debug(1, "transform Fock")
        H1[s] = transform_trans_inv_sparse(basis[s], lattice, latFock)
        # then add Vcor only in environment
        # need to substract impurity contribution
        log.debug(1, "transform Vcor")
        H1[s] += transform_local(basis[s], lattice, vcor.get()[s])

        if not "fitting" in kwargs or not kwargs["fitting"]:
            # for fitting purpose, we need H1 with vcor on impurity
            H1[s] -= transform_imp(basis[s], lattice, vcor.get()[s])

        # substract impurity Fock if necessary
        # i.e. rho_kl[2(ij||kl)-(il||jk)] where i,j,k,l are all impurity
        if ImpJK is not None:
            log.debug(1, "transform impurity JK")
            H1[s] -= transform_imp(basis[s], lattice, ImpJK)

        log.debug(1, "transform native H1")
        H1energy[s] = transform_imp_env(basis[s], lattice, latH1)

    return H1, H1energy

def __embHam2e(lattice, basis, vcor, local, **kwargs):
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[3]
    spin = basis.shape[0]

    if "mmap" in kwargs.keys() and kwargs["mmap"]:
        log.debug(0, "Use memory map for 2-electron integral")
        H2 = np.memmap(NamedTemporaryFile(dir = TmpDir), \
            dtype = float, mode = 'w+', shape = (spin*(spin+1)/2, nbasis, nbasis, nbasis, nbasis))
    else:
        H2 = np.zeros((spin*(spin+1)/2, nbasis, nbasis, nbasis, nbasis))

    log.info("H2 memory allocated size = %d MB", H2.size*8. / 1024 / 1024)

    if local:
        for s in range(spin):
            log.eassert(la.norm(basis[s,0,:,:nscsites] - np.eye(nscsites)) < 1e-10, \
                "the embedding basis is not local")
        for i in range(H2.shape[0]):
            H2[i, :nscsites, :nscsites, :nscsites, :nscsites] = lattice.getH2()
    else:
        for idx, (s1, s2) in enumerate(it.combinations_with_replacement(range(spin), 2)):
            # notice the order aa,ab,bb
            H2[idx] = transform_4idx(lattice.getH2(), basis[s1][0], basis[s1][0], \
                basis[s2][0], basis[s2][0])
    return H2

def FitVcorEmb(rho, lattice, basis, vcor, beta, MaxIter = 300, **kwargs):
    spin = basis.shape[0]
    nbasis = basis.shape[3]
    nscsites = lattice.supercell.nsites
    nelec = nscsites * spin

    embH1 = np.empty((spin, nbasis, nbasis))
    for s in range(spin):
        embH1[s] = transform_trans_inv_sparse(basis[s], lattice, lattice.getFock(kspace = False))

    ew = np.empty((spin, nbasis))
    ev = np.empty((spin, nbasis, nbasis))

    def errfunc(param):
        vcor.update(param)
        for s in range(spin):
            embHeff = embH1[s] + transform_local(basis[s], lattice, vcor.get()[s])
            ew[s], ev[s] = la.eigh(embHeff)
        ewocc, _, _ = assignocc(ew, nelec, beta, 0.)
        rho1 = np.empty_like(rho)
        for s in range(spin):
            rho1[s] = mdot(ev[s], np.diag(ewocc[s]), ev[s].T)

        return la.norm(rho - rho1) / sqrt(spin)

    param, err = minimize(errfunc, vcor.param, MaxIter, **kwargs)
    vcor.update(param)
    return vcor, err

def FitVcorFull(rho, lattice, basis, vcor, beta, filling, MaxIter = 20, **kwargs):
    spin = basis.shape[0]
    nbasis = basis.shape[3]
    rho1 = np.empty((spin, nbasis, nbasis))

    def errfunc(param):
        vcor.update(param)
        verbose = log.verbose
        log.verbose = "RESULT"
        rhoT, _, _ = HF(lattice, vcor, filling, spin == 1, mu0 = 0., beta = beta)
        log.verbose = verbose
        for s in range(spin):
            rho1[s] = transform_trans_inv_sparse(basis[s], lattice, rhoT[s], thr = 1e-6)
        return la.norm(rho - rho1) / sqrt(spin)

    param, err = minimize(errfunc, vcor.param, MaxIter, **kwargs)
    vcor.update(param)
    return vcor, err

def FitVcorTwoStep(rho, lattice, basis, vcor, beta, filling, MaxIter1 = 300, MaxIter2 = 20):
    vcor_new = deepcopy(vcor)
    log.result("Using two-step vcor fitting")
    if MaxIter1 > 0:
        log.info("Impurity model stage  max %d steps", MaxIter1)
        vcor_new, err = FitVcorEmb(rho, lattice, basis, vcor_new, beta, \
            MaxIter = MaxIter1, serial = True)
        log.info("residue = %20.12f", err)
    if MaxIter2 > 0:
        log.info("Full lattice stage  max %d steps", MaxIter2)
        vcor_new, err = FitVcorFull(rho, lattice, basis, vcor_new, beta, \
            filling, MaxIter = MaxIter2)
    log.result("residue = %20.12f", err)
    return vcor_new, err

def transformResults(rhoEmb, E, basis, ImpHam, H1e):
    spin = rhoEmb.shape[0]
    nscsites = basis.shape[2]
    rhoImp = np.empty((spin, nscsites, nscsites))
    nelec = 0
    for s in range(spin):
        rhoImp[s] = mdot(basis[s,0], rhoEmb[s], basis[s,0].T)
        nelec += np.trace(rhoImp[s])
    nelec *= (2./spin)

    if E is not None:
        Veff = ImpHam.H1["cd"] - H1e["cd"]
        Efrag = E - np.sum(Veff * rhoEmb) / spin * 2
    else:
        Efrag = None
    return rhoImp, Efrag, nelec
