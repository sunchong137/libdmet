import numpy as np
import numpy.linalg as la
import libdmet.utils.logger as log
from libdmet.system import integral
from bcs_helper import *
from slater import MatSqrt, orthonormalizeBasis

def embBasis(lattice, GRho, local = True, **kwargs):
    if local:
        return __embBasis_proj(lattice, GRho, **kwargs)
    else:
        return __embBasis_phsymm(lattice, GRho, **kwargs)

def __embBasis_proj(lattice, GRho, **kwargs):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    # spins give an additional factor of 2
    basis = np.zeros((2, ncells, nscsites*2, nscsites*2))
    # A is square root of impurity part
    A = MatSqrt(GRho[0])
    B = np.swapaxes(np.tensordot(la.inv(A), GRho[1:], axes = (1,1)), 0, 1)
    B = np.swapaxes(B, 1, 2)
    B = orthonormalizeBasis(B)
    basis[0, 0, :nscsites, :nscsites] = np.eye(nscsites)
    basis[1, 0, :nscsites, :nscsites] = np.eye(nscsites)
    # FIXME cut B to gain the largest particle property?
    w = np.diag(np.tensordot(B[:,:nscsites], B[:,:nscsites], axes = ((0,1),(0,1))))
    order = np.argsort(w)[::-1]
    w1 = np.sort(w)[::-1]
    orderA, orderB = order[:nscsites], order[nscsites:]
    wA, wB = w1[:nscsites], 1. - w1[nscsites:]
    log.debug(0, "particle character:\nspin A max %.2f min %.2f mean %.2f"
            "\nspin B max %.2f min %.2f mean %.2f", np.max(wA), np.min(wA), \
            np.average(wA), np.max(wB), np.min(wB), np.average(wB))
    basis[0, 1:, :, nscsites:] = B[:,:,orderA]
    basis[1, 1:, :nscsites, nscsites:], basis[1, 1:, nscsites:, nscsites:] = \
            B[:, nscsites:, orderB], B[:, :nscsites, orderB]
    return basis

def __embBasis_phsymm(lattice, GRho, **kwargs):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    basis = np.empty((2, ncells, nscsites*2, nscsites*2))
    A1 = MatSqrt(GRho[0])
    AB1 = np.swapaxes(np.tensordot(la.inv(A1), GRho, axes = (1,1)), 0, 1)
    AB1 = np.swapaxes(AB1, 1, 2)
    AB1 = orthonormalizeBasis(AB1)
    basis[0] = AB1
    GRho_h = -GRho
    GRho_h[0] += np.eye(nscsites*2)
    A2 = MatSqrt(GRho_h[0])
    AB2 = np.swapaxes(np.tensordot(la.inv(A2), GRho_h, axes = (1,1)), 0, 1)
    AB2 = np.swapaxes(AB2, 1, 2)
    AB2 = orthonormalizeBasis(AB2)
    basis[1, :, :nscsites], basis[1, :, nscsites:] = \
            AB2[:, nscsites:], AB2[:, nscsites:]
    return basis

def embHam(lattice, basis, vcor, local = True, **kwargs):
    log.info("One-body part")
    (Int1e, H0_from1e), (Int1e_energy, H0_energy_from1e) = \
            __embHam1e(lattice, basis, vcor, **kwargs)
    log.info("Two-body part")
    Int2e, Int1e_from2e, H0_from2e = __embHam2e(lattice, basis, vcor, local, **kwargs)

    nbasis = basis.shape[3]
    Int1e["cd"] += Int1e_from2e["cd"]
    Int1e["cc"] += Int1e_from2e["cc"]
    H0 = H0_from1e + H0_from2e
    Int1e_energy["cd"] += Int1e_from2e["cd"]
    Int1e_energy["cc"] += Int1e_from2e["cc"]
    H0_energy = H0_energy_from1e + H0_from2e
    return integral.Integral(nbasis, True, False, H0, Int1e, Int2e), \
            (Int1e_energy, H0_energy)


def __embHam1e(lattice, basis, vcor, **kwargs):
    log.eassert(vcor.islocal(), "nonlocal correlation potential cannot be treated in this routine")
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[3]
    latFock = lattice.getFock(kspace = False)
    latH1 = lattice.getH1(kspace = False)
    ImpJK = lattice.getImpJK()
    spin = 2
    H0 = 0.
    H1 = {"cd": np.empty((2, nbasis, nbasis)), "cc": np.empty((nbasis, nbasis))}
    H0energy = 0.
    H1energy = {"cd": np.empty((2, nbasis, nbasis)), "cc": np.empty((nbasis, nbasis))}

    # Fock part first
    log.debug(1, "transform Fock")
    H1["cd"], H1["cc"], H0 = transform_trans_inv_sparse(basis, lattice, latFock)
    # then add Vcor, only in environment
    # add it everywhere then subtract impurity part
    log.debug(1, "transform Vcor")
    tempCD, tempCC, tempH0 = transform_local(basis, lattice, vcor.get())
    H1["cd"] += tempCD
    H1["cc"] += tempCC
    H0 += tempH0

    if not "fitting" in kwargs or not kwargs["fitting"]:
        # for fitting purpose, we need H1 with vcor on impurity
        tempCD, tempCC, tempH0 = transform_imp(basis, lattice, vcor.get())
        H1["cd"] -= tempCD
        H1["cc"] -= tempCC
        H0 -= tempH0

    # subtract impurity Fock if necessary
    # i.e. rho_kl[2(ij||kl)-(il||jk)] where i,j,k,l are all impurity
    if ImpJK is not None:
        log.debug(1, "transform impurity JK")
        tempCD, tempCC, tempH0 = transform_imp(basis, lattice, ImpJK)
        H1["cd"] -= tempCD
        H1["cc"] -= tempCC
        H0 -= tempH0

    log.debug(1, "transform native H1")
    H1energy["cd"], H1energy["cc"], H0energy = transform_imp_env(basis, lattice, latH1)

    return (H1, H0), (H1energy, H0energy)

def __embHam2e(lattice, basis, vcor, local, **kwargs):
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[3]

    if "mmap" in kwargs.keys() and kwargs["mmap"]:
        log.debug(0, "Use memory map for 2-electron integral")
        ccdd = np.memmap(NamedTemporaryFile(dir = TmpDir), dtype = float, \
                mode = 'w+', shape = (3, nbasis, nbasis, nbasis, nbasis))
        if local:
            cccd = cccc = None
        else:
            cccd = np.memmap(NamedTemporaryFile(dir = TmpDir), dtype = float, \
                mode = 'w+', shape = (2, nbasis, nbasis, nbasis, nbasis))
            cccc = np.memmap(NamedTemporaryFile(dir = TmpDir), dtype = float, \
                mode = 'w+', shape = (nbasis, nbasis, nbasis, nbasis))
    else:
        ccdd = np.zeros((3, nbasis, nbasis, nbasis, nbasis))
        if local:
            cccd = cccc = None
        else:
            cccd = np.zeros((2, nbasis, nbasis, nbasis, nbasis))
            cccc = np.zeros((nbasis, nbasis, nbasis, nbasis))

    log.info("H2 memory allocated size = %d MB", ccdd.size * 2 * 8. / 1024 / 1024)
    
    if local:
        for s in range(2):
            log.eassert(la.norm(basis[s,0,:nscsites,:nscsites] - np.eye(nscsites)) \
                    < 1e-10, "the embedding basis is not local")
        for i in range(ccdd.shape[0]):
            ccdd[i, :nscsites, :nscsites, :nscsites, :nscsites] = lattice.getH2()
        cd = np.zeros((2, nbasis, nbasis))
        cc = np.zeros((nbasis, nbasis))
        H0 = 0.
    else:
        log.error("Int2e for nonlocal embedding basis not implemented yet")
    return {"ccdd": ccdd, "cccd": cccd, "cccc": cccc}, {"cd": cd, "cc": cc}, H0
