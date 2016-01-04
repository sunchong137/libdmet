import numpy as np
import numpy.linalg as la
from copy import deepcopy
from math import sqrt
import itertools as it
import libdmet.utils.logger as log
from libdmet.system import integral
from bcs_helper import *
from slater import MatSqrt, orthonormalizeBasis
from mfd import assignocc, HFB
from fit import minimize
from libdmet.utils.misc import mdot, find

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
            AB2[:, nscsites:], AB2[:, :nscsites]
    return basis

def embHam(lattice, basis, vcor, mu, local = True, **kwargs):
    log.info("One-body part")
    (Int1e, H0_from1e), (Int1e_energy, H0_energy_from1e) = \
            __embHam1e(lattice, basis, vcor, mu, **kwargs)
    log.info("Two-body part")
    Int2e, Int1e_from2e, H0_from2e = \
            __embHam2e(lattice, basis, vcor, local, **kwargs)
    nbasis = basis.shape[-1]
    Int1e["cd"] += Int1e_from2e["cd"]
    Int1e["cc"] += Int1e_from2e["cc"]
    H0 = H0_from1e + H0_from2e
    Int1e_energy["cd"] += Int1e_from2e["cd"]
    Int1e_energy["cc"] += Int1e_from2e["cc"]
    H0_energy = H0_energy_from1e + H0_from2e
    return integral.Integral(nbasis, False, True, H0, Int1e, Int2e), \
            (Int1e_energy, H0_energy)

def __embHam1e(lattice, basis, vcor, mu, **kwargs):
    log.eassert(vcor.islocal(), "nonlocal correlation potential cannot be treated in this routine")
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[-1]
    latFock = lattice.getFock(kspace = False)
    latH1 = lattice.getH1(kspace = False)
    ImpJK = lattice.getImpJK()
    spin = 2
    H0 = 0.
    H1 = {"cd": np.empty((2, nbasis, nbasis)), "cc": np.empty((1, nbasis, nbasis))}
    H0energy = 0.
    H1energy = {"cd": np.empty((2, nbasis, nbasis)), "cc": np.empty((1, nbasis, nbasis))}

    # Fock part first
    log.debug(1, "transform Fock")
    H1["cd"], H1["cc"][0], H0 = transform_trans_inv_sparse(basis, lattice, latFock)
    # then add Vcor, only in environment; and -mu*I in impurity and environment
    # add it everywhere then subtract impurity part
    log.debug(1, "transform Vcor")
    v = deepcopy(vcor.get())
    v[0] -= mu * np.eye(nscsites)
    v[1] -= mu * np.eye(nscsites)
    tempCD, tempCC, tempH0 = transform_local(basis, lattice, v)
    H1["cd"] += tempCD
    H1["cc"][0] += tempCC
    H0 += tempH0

    if not "fitting" in kwargs or not kwargs["fitting"]:
        # for fitting purpose, we need H1 with vcor on impurity
        tempCD, tempCC, tempH0 = transform_imp(basis, lattice, vcor.get())
        H1["cd"] -= tempCD
        H1["cc"][0] -= tempCC
        H0 -= tempH0

    # subtract impurity Fock if necessary
    # i.e. rho_kl[2(ij||kl)-(il||jk)] where i,j,k,l are all impurity
    if ImpJK is not None:
        log.debug(1, "transform impurity JK")
        tempCD, tempCC, tempH0 = transform_imp(basis, lattice, ImpJK)
        H1["cd"] -= tempCD
        H1["cc"][0] -= tempCC
        H0 -= tempH0

    log.debug(1, "transform native H1")
    H1energy["cd"], H1energy["cc"][0], H0energy = transform_imp_env(basis, lattice, latH1)
    return (H1, H0), (H1energy, H0energy)

def __embHam2e(lattice, basis, vcor, local, **kwargs):
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[-1]

    if "mmap" in kwargs.keys() and kwargs["mmap"]:
        log.debug(0, "Use memory map for 2-electron integral")
        ccdd = np.memmap(NamedTemporaryFile(dir = TmpDir), dtype = float, \
                mode = 'w+', shape = (3, nbasis, nbasis, nbasis, nbasis))
        cccd = np.memmap(NamedTemporaryFile(dir = TmpDir), dtype = float, \
            mode = 'w+', shape = (2, nbasis, nbasis, nbasis, nbasis))
        cccc = np.memmap(NamedTemporaryFile(dir = TmpDir), dtype = float, \
            mode = 'w+', shape = (1, nbasis, nbasis, nbasis, nbasis))
    else:
        ccdd = np.zeros((3, nbasis, nbasis, nbasis, nbasis))
        cccd = np.zeros((2, nbasis, nbasis, nbasis, nbasis))
        cccc = np.zeros((1, nbasis, nbasis, nbasis, nbasis))

    log.info("H2 memory allocated size = %d MB", ccdd.size * 2 * 8. / 1024 / 1024)
    
    if local:
        for s in range(2):
            log.eassert(la.norm(basis[s,0,:nscsites,:nscsites] - np.eye(nscsites)) \
                    < 1e-10, "the embedding basis is not local")
        for i in range(ccdd.shape[0]):
            ccdd[i, :nscsites, :nscsites, :nscsites, :nscsites] = lattice.getH2()
        cd = np.zeros((2, nbasis, nbasis))
        cc = np.zeros((1, nbasis, nbasis))
        H0 = 0.
    else:
        from libdmet.integral.integral_nonlocal_emb import transform
        VA, VB, UA, UB = separate_basis(basis)
        H0, cd, cc, ccdd, cccd, cccc = \
                transform(VA[0], VB[0], UA[0], UB[0], lattice.getH2())
        # FIXME the definition of UA and UB
    return {"ccdd": ccdd, "cccd": cccd, "cccc": cccc}, {"cd": cd, "cc": cc}, H0

def foldRho(GRho, Lat, basis, thr = 1e-7):
    ncells = Lat.ncells
    nscsites = Lat.supercell.nsites
    nbasis = basis.shape[-1]
    basisCanonical = np.empty((ncells, nscsites*2, nbasis*2))
    basisCanonical[:,:,:nbasis] = basis[0] # (VA, UB)^T
    basisCanonical[:,:nscsites,nbasis:] = basis[1, :, nscsites:] # UA
    basisCanonical[:,nscsites:,nbasis:] = basis[1, :, :nscsites] # VB
    res = np.zeros((nbasis*2, nbasis*2))
    mask_basis = set(find(True, map(lambda a: la.norm(a) > thr, basisCanonical)))
    mask_GRho = set(find(True, map(lambda a: la.norm(a) > thr, GRho)))
    if len(mask_GRho) < len(mask_basis):
        for i, Hidx in enumerate(mask_GRho):
            for i in mask_basis:
                j = Lat.add(i, Hidx)
                if j in mask_basis:
                    res += mdot(basisCanonical[i].T, GRho[Hidx], basisCanonical[j])
    else:
        for i, j in it.product(mask_basis, repeat = 2):
            Hidx = Lat.subtract(j, i)
            if Hidx in mask_GRho:
                res += mdot(basisCanonical[i].T, GRho[Hidx], basisCanonical[j])
    return res

def addDiag(v, scalar):
    rep = v.get()
    nscsites = rep.shape[1]
    rep[0] += np.eye(nscsites) * scalar
    rep[1] += np.eye(nscsites) * scalar
    v.assign(rep)
    return v

def FitVcorEmb(GRho, lattice, basis, vcor, mu, MaxIter = 300, **kwargs):
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[-1]
    (embHA, embHB), embD, _ = transform_trans_inv_sparse(basis, lattice, \
            lattice.getFock(kspace = False))

    embH = np.empty((nbasis*2, nbasis*2))
    embH[:nbasis, :nbasis] = embHA
    embH[nbasis:, nbasis:] = -embHB
    embH[:nbasis, nbasis:] = embD
    embH[nbasis:, :nbasis] = embD.T

    # now compute dV/dparam (will be used in gradient)
    dV_dparam = np.empty((vcor.length(), nbasis*2, nbasis*2))
    for ip in range(vcor.length()):
        (dA_dV, dB_dV), dD_dV, _ = \
                transform_local(basis, lattice, vcor.gradient()[ip])
        dV_dparam[ip, :nbasis, :nbasis] = dA_dV
        dV_dparam[ip, nbasis:, nbasis:] = -dB_dV
        dV_dparam[ip, :nbasis, nbasis:] = dD_dV
        dV_dparam[ip, nbasis:, :nbasis] = dD_dV.T

    vcor_zero = deepcopy(vcor)
    vcor_zero.update(np.zeros(vcor_zero.length()))
    v0 = vcor_zero.get()
    v0[0] -= mu * np.eye(nscsites)
    v0[1] -= mu * np.eye(nscsites)
    (A0, B0), D0, _ = \
            transform_local(basis, lattice, v0)

    def Vemb_param(param):
        V = np.tensordot(param, dV_dparam, axes = (0, 0))
        V[:nbasis, :nbasis] += A0
        V[nbasis:, nbasis:] -= B0
        V[:nbasis, nbasis:] += D0
        V[nbasis:, :nbasis] += D0.T
        return V

    def errfunc(param):
        vcor.update(param)
        embHeff = embH + Vemb_param(param)
        ew, ev = la.eigh(embHeff)
        occ = 1 * (ew < 0.)
        GRho1 = mdot(ev, np.diag(occ), ev.T)
        return la.norm(GRho - GRho1) / sqrt(2.)

    def gradfunc(param):
        vcor.update(param)
        embHeff = embH + Vemb_param(param)
        ew, ev = la.eigh(embHeff)
        nocc = 1 * (ew < 0.)
        GRho1 = mdot(ev, np.diag(nocc), ev.T)
        val = la.norm(GRho - GRho1)
        ewocc, ewvirt = ew[:nbasis], ew[nbasis:]
        evocc, evvirt = ev[:, :nbasis], ev[:, nbasis:]
        # dGRho_ij / dV_ij, where V corresponds to terms in the
        # embedding generalized density matrix
        #c_jln = np.einsum("jn,ln->jln", evocc, evocc)
        #c_ikm = np.einsum("im,km->ikm", evvirt, evvirt)
        #e_mn = 1. / (-ewvirt.reshape((-1,1)) + ewocc)
        #dGRho_dV = np.swapaxes(np.tensordot(np.tensordot(c_ikm, e_mn, \
        #        axes = (2,0)), c_jln, axes = (2,2)), 1, 2)
        #dGRho_dV += np.swapaxes(np.swapaxes(dGRho_dV, 0, 1), 2, 3)
        #dnorm_dV = np.tensordot(GRho1 - GRho, dGRho_dV, \
        #        axes = ((0,1), (0,1))) / val / sqrt(2.)
        e_mn = 1. / (-ewvirt.reshape((-1,1)) + ewocc)
        temp_mn = mdot(evvirt.T, GRho1 - GRho, evocc) * e_mn / val / sqrt(2.)
        dnorm_dV = mdot(evvirt, temp_mn, evocc.T)
        dnorm_dV += dnorm_dV.T
        return np.tensordot(dV_dparam, dnorm_dV, axes = ((1,2), (0,1)))

    err_begin = errfunc(vcor.param)
    log.info("Using analytic gradient")
    param, err_end = minimize(errfunc, vcor.param, MaxIter, gradfunc, **kwargs)
    return vcor, err_begin, err_end

def FitVcorFull(GRho, lattice, basis, vcor, mu, MaxIter, **kwargs):
    nbasis = basis.shape[-1]

    def errfunc(param):
        vcor.update(param)
        verbose = log.verbose
        log.verbose = "RESULT"
        GRhoT, _, _ = HFB(lattice, vcor, False, mu = mu, beta = np.inf)
        log.verbose = verbose
        GRho1 = foldRho(GRhoT, lattice, basis)
        return la.norm(GRho - GRho1) / sqrt(2.)

    err_begin = errfunc(vcor.param)
    param, err_end = minimize(errfunc, vcor.param, MaxIter, **kwargs)
    vcor.update(param)
    return vcor, err_begin, err_end

def FitVcorTwoStep(GRho, lattice, basis, vcor, mu, MaxIter1 = 300, MaxIter2 = 0):
    vcor_new = deepcopy(vcor)
    log.result("Using two-step vcor fitting")
    if MaxIter1 > 0:
        log.info("Impurity model stage  max %d steps", MaxIter1)
        vcor_new, err_begin, err_end = FitVcorEmb(GRho, lattice, basis, vcor_new, \
            mu, MaxIter = MaxIter1, serial = True)
        log.result("residue (begin) = %20.12f", err_begin)
        log.info("residue (end)   = %20.12f", err_end)
    if MaxIter2 > 0:
        log.info("Full lattice stage  max %d steps", MaxIter2)
        vcor_new, _, err_end = FitVcorFull(GRho, lattice, basis, vcor_new, \
                mu, MaxIter = MaxIter2)
    log.result("residue (end)   = %20.12f", err_end)
    return vcor_new, err_begin

def transformResults(GRhoEmb, E, lattice, basis, ImpHam, H_energy, dmu):
    VA, VB, UA, UB = separate_basis(basis)
    nscsites = basis.shape[-2] / 2
    nbasis = basis.shape[-1]
    R = np.empty((nscsites*2, nbasis*2))
    R[:nscsites, :nbasis] = VA[0]
    R[nscsites:, :nbasis] = UB[0]
    R[:nscsites, nbasis:] = UA[0]
    R[nscsites:, nbasis:] = VB[0]
    GRhoImp = mdot(R, GRhoEmb, R.T)
    occs = np.diag(GRhoImp)
    nelec = np.sum(occs[:nscsites]) - np.sum(occs[nscsites:]) + nscsites
    if E is not None:
        # FIXME energy expression is definitely wrong with mu built in the
        # Hamiltonian
        H1energy, H0energy = H_energy
        rhoA, rhoB, kappaBA = extractRdm(GRhoEmb)

        tempCD, tempCC, tempH0 = transform_imp(basis, lattice, dmu * np.eye(nscsites))

        CDeff = ImpHam.H1["cd"] - H1energy["cd"] - tempCD
        CCeff = ImpHam.H1["cc"] - H1energy["cc"] - tempCC
        H0eff = ImpHam.H0 - H0energy - tempH0
        Efrag = E - np.sum(CDeff[0] * rhoA + CDeff[1] * rhoB) - \
                2 * np.sum(CCeff[0] * kappaBA.T) - H0eff
    else:
        Efrag = None
    return GRhoImp, Efrag, nelec
