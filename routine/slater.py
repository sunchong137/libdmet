import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log
from slater_helper import *
from tempfile import TemporaryFile
from libdmet.system import integral

tmp = "/tmp"

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
    rhobig = lattice.expand(rho[0])
    basis = np.zeros((spin, ncells, nscsites, nscsites * 2))
    for s in range(spin):
        A = MatSqrt(rho[s,0])
        B1 = np.swapaxes(rho[s,1:], 0, 1).reshape((nscsites, -1))
        B1 = np.dot(la.inv(A), B1).T
        B1 = normalizeBasis(B1)
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

    nspin = basis.shape[0]
    nbasis = basis.shape[3]
    if nspin == 1:
        return integral.Integral(nbasis, True, False, 0, {"cd": Int1e}, {"ccdd": Int2e}), {"cd": Int1e_energy}
    else:
        if "mmap" in kwargs.keys() and kwargs["mmap"]:
            H2 = {
                "ccddA": np.memmap(NamedTemporaryFile(dir = TmpDir), \
                    dtype = float, mode = 'w+', shape = (nbasis, nbasis, nbasis, nbasis)),
                "ccddB": np.memmap(NamedTemporaryFile(dir = TmpDir), \
                    dtype = float, mode = 'w+', shape = (nbasis, nbasis, nbasis, nbasis)),
                "ccddAB": np.memmap(NamedTemporaryFile(dir = TmpDir), \
                    dtype = float, mode = 'w+', shape = (nbasis, nbasis, nbasis, nbasis)),
            }
        else:
            H2 = {
                "ccddA": np.empty((nbasis, nbasis, nbasis, nbasis)),
                "ccddB": np.empty((nbasis, nbasis, nbasis, nbasis)),
                "ccddAB": np.empty((nbasis, nbasis, nbasis, nbasis)),
            }
        H2["ccddA"][:] = Int2e[0][:]
        H2["ccddAB"][:] = Int2e[1][:]
        H2["ccddB"][:] = Int2e[2][:]

        return integral.Integral(nbasis, False, False, 0, {"cdA": Int1e[0], "cdB": Int1e[1]}, H2),\
            {"cdA": Int1e_energy[0], "cdB": Int1e_energy[1]}

def __embHam1e(lattice, basis, vcor, **kwargs):
    log.eassert(vcor.islocal(), "nonlocal correlation potential cannot be treated in this routine")
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    latFock = lattice.getFock(kspace = False)
    latH1 = lattice.getH1(kspace = False)
    ImpJK = lattice.getImpJK()
    spin = basis.shape[0]
    H1 = np.empty((spin, 2*nscsites, 2*nscsites))
    H1energy = np.empty((spin, 2*nscsites, 2*nscsites))

    for s in range(spin):
        log.debug(0, "Spin Component %d of %d", s, spin)
        # Fock part first
        log.debug(1, "transform Fock")
        H1[s] = transform_trans_inv(basis[s], lattice, latFock)

        # then add Vcor only in environment
        # need to substract impurity contribution
        log.debug(1, "transform Vcor")
        H1[s] += transform_local(basis[s], lattice, vcor()[s])
        H1[s] -= transform_imp(basis[s], lattice, vcor()[s])

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
        H2 = np.memmap(NamedTemporaryFile(dir = TmpDir), \
            dtype = float, mode = 'w+', shape = (spin*(spin+1)/2, nbasis, nbasis, nbasis, nbasis))
    else:
        H2 = np.zeros((spin*(spin+1)/2, nbasis, nbasis, nbasis, nbasis))

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
