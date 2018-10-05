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
from libdmet import settings


def embBasis(lattice, GRho, local = True, **kwargs):
    if local:
        return __embBasis_proj(lattice, GRho, **kwargs)
    else:
        return __embBasis_phsymm(lattice, GRho, **kwargs)

def __embBasis_proj(lattice, GRho, **kwargs):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    if "sites" in kwargs:
        Imps = kwargs["sites"]
        ImpIdx = Imps + map(lambda i: i+nscsites, Imps)
        EnvIdx = filter(lambda i: not i in ImpIdx, range(2*nscsites*ncells))
        nImp = len(Imps)
        basis = np.zeros((2, ncells*nscsites*2, nImp*2))
        GRhoImpEnv = np.delete(np.transpose(GRho, \
                (1, 0, 2)).reshape(nscsites*2, nscsites*ncells*2)[ImpIdx], ImpIdx, 1)
        _, _, vt = la.svd(GRhoImpEnv, full_matrices = False)
        log.debug(1, "bath orbitals\n%s", vt)
        B = vt.T
        basis[np.ix_([0], Imps, range(nImp))] = np.eye(nImp)
        basis[np.ix_([1], Imps, range(nImp))] = np.eye(nImp)
        BathIdxV = range(nscsites-nImp)
        BathIdxU = range(nscsites-nImp, 2*(nscsites-nImp))
        for i in range(ncells-1):
            BathIdxV += range(2*(nscsites-nImp)+nscsites*i*2, \
                    2*(nscsites-nImp)+nscsites*(2*i+1))
            BathIdxU += range(2*(nscsites-nImp)+nscsites*(2*i+1), \
                    2*(nscsites-nImp)+nscsites*(2*i+2))
        EnvIdxV = [EnvIdx[i] for i in BathIdxV]
        EnvIdxU = [EnvIdx[i] for i in BathIdxU]
        w = np.diag(np.dot(B[BathIdxV].T, B[BathIdxV]))
        order = np.argsort(w)[::-1]
        w1 = np.sort(w)[::-1]
        orderA, orderB = order[:nImp], order[nImp:]
        wA, wB = w1[:nImp], 1. - w1[nImp:]
        log.debug(0, "particle character:\nspin A max %.2f min %.2f mean %.2f"
                "\nspin B max %.2f min %.2f mean %.2f", np.max(wA), np.min(wA), \
                np.average(wA), np.max(wB), np.min(wB), np.average(wB))
        basis[np.ix_([0], EnvIdx, range(nImp, nImp*2))] = B[:, orderA]
        basis[np.ix_([1], EnvIdxV, range(nImp, nImp*2))] = B[np.ix_(BathIdxU, orderB)]
        basis[np.ix_([1], EnvIdxU, range(nImp, nImp*2))] = B[np.ix_(BathIdxV, orderB)]
        basis = basis.reshape((2, ncells, nscsites*2, nImp*2))
    else:
        # spins give an additional factor of 2
        basis = np.zeros((2, ncells, nscsites*2, nscsites*2))
        # A is square root of impurity part
        #A = MatSqrt(GRho[0])
        #B = np.swapaxes(np.tensordot(la.inv(A), GRho[1:], axes = (1,1)), 0, 1)
        #B = np.swapaxes(B, 1, 2)
        #B = orthonormalizeBasis(B)
        GRhoImpEnv = np.transpose(GRho[1:], (1, 0, 2)).reshape(nscsites*2, nscsites*(ncells-1)*2)
        _, s, vt = la.svd(GRhoImpEnv, full_matrices = False)
        log.debug(1, "bath orbitals\n%s", vt)
        # ZHC NOTE first two site basis, third emb basis (dim contract with singular value)
        B = np.transpose(vt.reshape((nscsites*2, ncells-1, nscsites*2)), (1, 2, 0))
        if "localize_bath" in kwargs:
            if kwargs["localize_bath"] == True:
                # PM localization of bath
                from localizer import localize_bath
                B = localize_bath(B)
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
        log.info("Bath coupling strength\n%s\n%s", s[orderA], s[orderB])
    if "return_bath" in kwargs:
        return B
    else:
        return basis

def __embBasis_phsymm(lattice, GRho, **kwargs):
    if "sites" in kwargs:
        log.error('keyword "sites" not supported.')
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
    H0 = H0_from1e + H0_from2e
    H0_energy = H0_energy_from1e + H0_from2e
    if Int1e_from2e is not None:
        Int1e["cd"] += Int1e_from2e["cd"]
        Int1e["cc"] += Int1e_from2e["cc"]
        Int1e_energy["cd"] += Int1e_from2e["cd"]
        Int1e_energy["cc"] += Int1e_from2e["cc"]
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

    if settings.save_mem:
        if local:
            return {"ccdd": lattice.getH2()[np.newaxis,:], "cccd": None, "cccc": None}, \
                    None, 0.
        else:
            log.warning("Basis nonlocal, ignoring memory saving option")
            settings.save_mem = False

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
        return {"ccdd": ccdd, "cccd": cccd, "cccc": cccc}, None, 0.
    else:
        from libdmet.integral.integral_nonlocal_emb import transform
        VA, VB, UA, UB = separate_basis(basis)
        H01, cd1, cc1, ccdd1, cccd1, cccc1 = \
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
        for Hidx in mask_GRho:
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

def FitVcorEmb(GRho, lattice, basis, vcor, mu, MaxIter = 300, CG_check = False, **kwargs):
    param_begin = vcor.param.copy()
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
        # add contribution of chemical potential # ZHC NOTE
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
    param, err_end, converge_pattern = minimize(errfunc, vcor.param, MaxIter, gradfunc, **kwargs)
    
    # ZHC NOTE
    gnorm_res = la.norm(gradfunc(param))
    
    vcor.update(param)
    
    print "Minimizer converge pattern: %d "%converge_pattern
    print "Current function value: %15.8f"%err_end
    print "Norm of gradients: %15.8f"%gnorm_res
    print "Norm diff of x: %15.8f"%(la.norm(param- param_begin))
    
    if CG_check and (converge_pattern == 0 or gnorm_res > 1.0e-4):
        
        print "Not converge in Bo-Xiao's minimizer, try mixed solver in scipy..."

        param_new = param.copy()
        gtol = 5.0e-5

        from scipy import optimize as opt
        min_result = opt.minimize(errfunc, param_new, method = 'CG', jac = gradfunc ,\
                options={'maxiter': 10 * len(param_new), 'disp': True, 'gtol': gtol})
        param_new_2 = min_result.x
    
        print "CG Final Diff: ", min_result.fun, "Converged: ",min_result.status,\
                " Jacobian: ", la.norm(min_result.jac)      
        if(not min_result.success):
            print "WARNING: Minimization unsuccessful. Message: ",min_result.message
    
        gnorm_new = la.norm(min_result.jac)
        diff_CG_BX = np.max(np.abs(param_new_2 - param_new))
        print "max diff in x between CG and BX", diff_CG_BX
        if (gnorm_new < gnorm_res * 0.5) and (min_result.fun < err_end) and (diff_CG_BX < 1.0):
            print "CG result used"
            #vcor.param = param_new_2
            vcor.update(param_new_2)
            err_end = min_result.fun
        else:
            print "BX result used"
    else:
        print "BX result used"


    return vcor, err_begin, err_end

def FitVcorEmb_triu(GRho, lattice, basis, vcor, mu, MaxIter = 300, CG_check = False, **kwargs):
    param_begin = vcor.param.copy()
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[-1]
    
    # ZHC transform lattice H -> emb space
    # ZHC NOTE use more accurate one?
    (embHA, embHB), embD, _ = transform_trans_inv_sparse(basis, lattice, \
            lattice.getFock(kspace = False))
    #(embHA, embHB), embD, _ = transform_trans_inv(basis, lattice, \
    #        lattice.getFock(kspace = False))

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
    
    ll = np.triu_indices(GRho.shape[0])

    def Vemb_param(param):
        V = np.tensordot(param, dV_dparam, axes = (0, 0))
        # add contribution of chemical potential # ZHC NOTE
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
        return la.norm((GRho - GRho1)[ll])**2

    def gradfunc(param):
        vcor.update(param)
        embHeff = embH + Vemb_param(param)
        ew, ev = la.eigh(embHeff)
        nocc = 1 * (ew < 0.)
        GRho1 = mdot(ev, np.diag(nocc), ev.T)
        
        jac = np.zeros((len(param), embHeff.shape[1], embHeff.shape[1]), dtype = param.dtype)

        #Only concerned with corners
        for k, vep in enumerate(dV_dparam):
            drho = analyticGradientO(ev, ew, vep, nbasis)
            jac[k] = drho

        #Gradient function: careful with complex stuff
        gradfn = np.zeros(len(param), dtype=np.float64)
        diffrdm = (GRho1 - GRho)[ll] 
        diffrdmR = diffrdm.real
        diffrdmI = diffrdm.imag
        
        for k in xrange(len(param)):
            J = jac[k][ll]
            gradfn[k] = 2.*np.sum(np.multiply(J.real,diffrdmR) + np.multiply(J.imag,diffrdmI))
        
        return gradfn

    def analyticGradientO(C, E, dH, nocc):
        
         L = dH.shape[1]
         
         Cocc = C[:, :nocc]
         Cvir = C[:, nocc:]
         
         de_ov = E[:nocc] - E[nocc:][:, None]
         zero_element = np.abs(de_ov) < 1.0e-5
         if zero_element.any():
             print "WARNING: Degeneracy occurs when evaluate gradients! "
             de_ov[zero_element] = np.sign(de_ov[zero_element] + 1.0e-20) * 0.01
         
         Zm = np.divide(np.dot(Cvir.conjugate().T, np.dot(dH,Cocc)), de_ov)
     
         Cmocc = np.dot(Cvir, Zm)
         CCT = np.dot(Cocc, Cmocc.conjugate().T) 
         result = CCT + CCT.conj().T
         
         return result

    """
    # ZHC NOTE test numerical gradients
    du = 1e-5
    param_0 = vcor.param.copy()
    err_ref = errfunc(param_0)
    grad_ref = gradfunc(param_0)
    grad_num = np.zeros_like(grad_ref)
    for i in xrange(len(param_0)):
        param_i = (param_0.copy())
        param_i[i] += du
        err_i = errfunc(param_i)
        grad_num[i] = (err_i - err_ref) / du
    
    print "GRAD DIFF"
    print np.linalg.norm(grad_ref - grad_num)
    
    """


    err_begin = errfunc(vcor.param)
    log.info("Using analytic gradient")
    param, err_end, converge_pattern = minimize(errfunc, vcor.param, MaxIter, gradfunc, **kwargs)
   
    
    # ZHC NOTE
    gnorm_res = la.norm(gradfunc(param))
    
    vcor.update(param)
    
    print "Minimizer converge pattern: %d "%converge_pattern
    print "Current function value: %15.8f"%err_end
    print "Norm of gradients: %15.8f"%gnorm_res
    print "Norm diff of x: %15.8f"%(la.norm(param- param_begin))
    
    if CG_check and (gnorm_res > 1.0e-4):
        
        print "Not fully converge in Bo-Xiao's minimizer, try mixed solver in scipy..."

        param_new = param.copy()
        gtol = 5.0e-5

        from scipy import optimize as opt
        min_result = opt.minimize(errfunc, param_new, method = 'CG', jac = gradfunc ,\
                options={'maxiter': len(param_new), 'disp': True, 'gtol': gtol})
        param_new_2 = min_result.x
    
        print "CG Final Diff: ", min_result.fun, "Converged: ",min_result.status,\
                " Jacobian: ", la.norm(min_result.jac)      
        if(not min_result.success):
            print "WARNING: Minimization unsuccessful. Message: ",min_result.message
    
        gnorm_new = la.norm(min_result.jac)
        diff_CG_BX = np.max(np.abs(param_new_2 - param_new))
        print "max diff in x between CG and BX", diff_CG_BX
        if (gnorm_new < gnorm_res * 0.5) and (min_result.fun < err_end) and (diff_CG_BX < 1.0):
            print "CG result used"
            vcor.update(param_new_2)
            err_end = min_result.fun
        else:
            print "BX result used"
    else:
        print "BX result used"


    return vcor, err_begin, err_end


def FitVcorFull(GRho, lattice, basis, vcor, mu, MaxIter, **kwargs):
    nbasis = basis.shape[-1]
    verbose = log.verbose

    def callback(param):
        vcor.update(param)
        log.verbose = "RESULT"
        GRhoTRef, _, _ = HFB(lattice, vcor, False, mu = mu, beta = np.inf)
        log.verbose = verbose
        GRho1Ref = foldRho(GRhoTRef, lattice, basis, thr = 1e-8)
        return {"GRhoTRef": GRhoTRef, "GRho1Ref": GRho1Ref}

    def errfunc(param, ref = None):
        vcor.update(param)
        log.verbose = "RESULT"
        GRhoT, _, _ = HFB(lattice, vcor, False, mu = mu, beta = np.inf)
        log.verbose = verbose
        if ref is None:
            GRho1 = foldRho(GRhoT, lattice, basis, thr = 1e-8)
        else:
            GRho1 = foldRho(GRhoT - ref["GRhoTRef"], lattice, basis, thr = 1e-8) \
                    + ref["GRho1Ref"]
        return la.norm(GRho - GRho1) / sqrt(2.)

    err_begin = errfunc(vcor.param)
    param, err_end, converge_pattern = minimize(errfunc, vcor.param, MaxIter, callback = callback, **kwargs)
    vcor.update(param)
    return vcor, err_begin, err_end

def FitVcorFullK(GRho, lattice, basis, vcor, mu, MaxIter, **kwargs):
    nscsites = lattice.supercell.nsites

    def costfunc(param, v = False):
        vcor.update(param)
        verbose = log.verbose
        log.verbose = "RESULT"
        GRhoT, _, _ = HFB(lattice, vcor, False, mu = mu, beta = np.inf)
        log.verbose = verbose

        tempRdm = map(extractRdm, GRhoT)
        rhoAT = np.asarray([rhoA for (rhoA, rhoB, kappaBA) in tempRdm])
        rhoBT = np.asarray([rhoB for (rhoA, rhoB, kappaBA) in tempRdm])
        kappaBA0 = tempRdm[0][2]

        kinetic = np.sum((rhoAT+rhoBT) * lattice.getFock(kspace = False))

        rhoA, rhoB, kappaBA = extractRdm(GRho)
        rhoAImp = rhoA[:nscsites, :nscsites]
        rhoBImp = rhoB[:nscsites, :nscsites]
        kappaBAImp = kappaBA[:nscsites, :nscsites]

        constraint = np.sum((vcor.get()[0]-mu*np.eye(nscsites)) * (rhoAT[0] - rhoAImp)) + \
                np.sum((vcor.get()[1]-mu*np.eye(nscsites)) * (rhoBT[0] - rhoBImp)) + \
                np.sum(vcor.get()[2] * (kappaBA0 - kappaBAImp).T) + \
                np.sum(vcor.get()[2].T * (kappaBA0 - kappaBAImp))

        if v:
            return kinetic, constraint
        else:
            return -(kinetic + constraint)

    def grad(param):
        vcor.update(param)
        verbose = log.verbose
        log.verbose = "RESULT"
        GRhoT, _, _ = HFB(lattice, vcor, False, mu = mu, beta = np.inf)
        log.verbose = verbose

        rhoA0, rhoB0, kappaBA0 = extractRdm(GRhoT[0])
        rhoA, rhoB, kappaBA = extractRdm(GRho)

        dRho = np.asarray([rhoA0 - rhoA[:nscsites, :nscsites], \
                rhoB0 - rhoB[:nscsites, :nscsites], \
                2 * (kappaBA0.T - kappaBA.T[:nscsites, :nscsites])])
        return -np.tensordot(vcor.gradient(), dRho, axes = ((1,2,3), (0,1,2)))

    from scipy.optimize import minimize
    ke_begin, c_begin = costfunc(vcor.param, v = True)
    log.info("begin: \nkinetic energy = %20.12f    constraint = %20.12f", ke_begin, c_begin)
    param = minimize(costfunc, vcor.param, jac = grad).x
    ke_end, c_end = costfunc(param, v = True)
    log.info("end: \nkinetic energy = %20.12f    constraint = %20.12f", ke_end, c_end)

    vcor.update(param)
    return vcor, c_begin, c_end


def FitVcorTwoStep(GRho, lattice, basis, vcor, mu, MaxIter1 = 300, MaxIter2 = 0, kinetic = False, triu = True, CG_check
        = False):
    vcor_new = deepcopy(vcor)
    log.result("Using two-step vcor fitting")
    err_begin = None
    if kinetic:
        log.check(MaxIter1 > 0, "Embedding fitting with kinetic energy minimization does not work!\n"
                "Skipping Embedding fitting")
        if MaxIter2 == 0:
            log.warning("Setting MaxIter2 to 1")
            MaxIter2 = 1

        vcor_new, err_begin, err_end = FitVcorFullK(GRho, lattice, basis, vcor_new, \
                    mu, MaxIter = MaxIter2)
    else:
        if MaxIter1 > 0:
            log.info("Impurity model stage  max %d steps", MaxIter1)
            if triu:
                vcor_new, err_begin1, err_end1 = FitVcorEmb_triu(GRho, lattice, basis, vcor_new, \
                        mu, MaxIter = MaxIter1, CG_check = CG_check, serial = True)
            else:
                vcor_new, err_begin1, err_end1 = FitVcorEmb(GRho, lattice, basis, vcor_new, \
                        mu, MaxIter = MaxIter1, CG_check = CG_check, serial = True)

            log.info("Embedding Stage:\nbegin %20.12f    end %20.12f" % (err_begin1, err_end1))
        if MaxIter2 > 0:
            log.info("Full lattice stage  max %d steps", MaxIter2)
            vcor_new, err_begin2, err_end2 = FitVcorFull(GRho, lattice, basis, vcor_new, \
                    mu, MaxIter = MaxIter2)
            log.info("Full Lattice Stage:\nbegin %20.12f    end %20.12f" % (err_begin2, err_end2))
        if MaxIter1 > 0:
            err_begin = err_begin1
        else:
            err_begin = err_begin2
        if MaxIter2 > 0:
            err_end = err_end2
        else:
            err_end = err_end1


    log.result("residue (begin) = %20.12f", err_begin)
    log.result("residue (end)   = %20.12f", err_end)
    #return vcor_new, err_begin
    return vcor_new, err_end

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


def pack_1rdm_fromdmrg(P, nscsites):
    '''
        Given P in BCS format from dmrg repack to GHF format
    '''
    nbs = P.shape[0]/2
    ne = nbs-nscsites

    rdm1 = np.zeros_like(P, dtype=P.dtype)
    rdm1[:nscsites, :nscsites] = P[:nscsites, :nscsites] 
    rdm1[:nscsites, nscsites:nscsites*2] = P[:nscsites, nbs:nbs+nscsites] 
    rdm1[nscsites:nscsites*2, :nscsites] = P[nbs:nbs+nscsites, :nscsites] 
    rdm1[nscsites:nscsites*2, nscsites:nscsites*2] = P[nbs:nbs+nscsites,nbs:nbs+nscsites] 

    rdm1[:nscsites, 2*nscsites:2*nscsites+ne] = P[:nscsites, nscsites:nbs] 
    rdm1[:nscsites, 2*nscsites+ne:] = P[:nscsites, nbs+nscsites:] 
    rdm1[nscsites:nscsites*2, 2*nscsites:2*nscsites+ne] = P[nbs:nbs+nscsites, nscsites:nbs] 
    rdm1[nscsites:nscsites*2, 2*nscsites+ne:] = P[nbs:nbs+nscsites, nbs+nscsites:] 

    rdm1[2*nscsites:,:2*nscsites] = rdm1[:2*nscsites, 2*nscsites:].conj().T 

    rdm1[2*nscsites:-ne, 2*nscsites:-ne] = P[nscsites:nbs, nscsites:nbs] 
    rdm1[2*nscsites:-ne, -ne:] = P[nscsites:nbs, -ne:] 
    rdm1[-ne:, 2*nscsites:-ne] = P[-ne:, nscsites:nbs] 
    rdm1[-ne:, -ne:] = P[-ne:, -ne:]

    return rdm1
 

def transformResults_new(GRhoEmb, E, lattice, basis, ImpHam, H_energy, last_dmu, Mu):
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
        # ZHC NOTE: the following energy is from defination of Edmet.
        # Efrag = E1 + E2
        # where E1 = partial Tr(rho, H1), H1 should NOT include contribution from Mu and last_dmu
        # E2 = E_dmrg - <psi | himp| psi>, psi is dmrg wavefunction

        rhoA, rhoB, kappaBA = extractRdm(GRhoEmb)
        
        E2 = E - np.sum(ImpHam.H1["cd"][0] * rhoA + ImpHam.H1["cd"][1] * rhoB) - \
                2 * np.sum(ImpHam.H1["cc"][0] * kappaBA.T) - ImpHam.H0

        # remove the contribution of last_dmu
        from libdmet.dmet.Hubbard import apply_dmu
        ImpHam_no_last_dmu = apply_dmu(lattice, deepcopy(ImpHam), basis, -last_dmu)
        
        H1_scaled = deepcopy(ImpHam_no_last_dmu.H1)
        
        # add back the global mu 
        v = np.zeros((3, nscsites, nscsites))
        v[0] = Mu * np.eye(nscsites)
        v[1] = Mu * np.eye(nscsites)
        tempCD, tempCC, tempH0 = transform_local(basis, lattice, v)
        H1_scaled["cd"] += tempCD
        H1_scaled["cc"][0] += tempCC

        # scale by the number of imp indices
        H1_scaled["cd"][0][:nscsites, nscsites:] *= 0.5
        H1_scaled["cd"][0][nscsites:, :nscsites] *= 0.5
        H1_scaled["cd"][0][nscsites:, nscsites:] = 0.0
        H1_scaled["cd"][1][:nscsites, nscsites:] *= 0.5
        H1_scaled["cd"][1][nscsites:, :nscsites] *= 0.5
        H1_scaled["cd"][1][nscsites:, nscsites:] = 0.0
        H1_scaled["cc"][0][:nscsites, nscsites:] *= 0.5
        H1_scaled["cc"][0][nscsites:, :nscsites] *= 0.5
        H1_scaled["cc"][0][nscsites:, nscsites:] = 0.0
        
        E1 = np.sum(H1_scaled["cd"][0] * rhoA + H1_scaled["cd"][1] * rhoB) + \
                2 * np.sum(H1_scaled["cc"][0] * kappaBA.T)

        Efrag = E1 + E2
    else:
        Efrag = None
    return GRhoImp, Efrag, nelec


def pack_1rdm_fromdmrg(P, nscsites):
    '''
        Given P in BCS format from dmrg repack to GHF format
    '''
    nbs = P.shape[0]/2
    ne = nbs-nscsites

    rdm1 = np.zeros_like(P, dtype=P.dtype)
    rdm1[:nscsites, :nscsites] = P[:nscsites, :nscsites] 
    rdm1[:nscsites, nscsites:nscsites*2] = P[:nscsites, nbs:nbs+nscsites] 
    rdm1[nscsites:nscsites*2, :nscsites] = P[nbs:nbs+nscsites, :nscsites] 
    rdm1[nscsites:nscsites*2, nscsites:nscsites*2] = P[nbs:nbs+nscsites,nbs:nbs+nscsites] 

    rdm1[:nscsites, 2*nscsites:2*nscsites+ne] = P[:nscsites, nscsites:nbs] 
    rdm1[:nscsites, 2*nscsites+ne:] = P[:nscsites, nbs+nscsites:] 
    rdm1[nscsites:nscsites*2, 2*nscsites:2*nscsites+ne] = P[nbs:nbs+nscsites, nscsites:nbs] 
    rdm1[nscsites:nscsites*2, 2*nscsites+ne:] = P[nbs:nbs+nscsites, nbs+nscsites:] 

    rdm1[2*nscsites:,:2*nscsites] = rdm1[:2*nscsites, 2*nscsites:].conj().T 

    rdm1[2*nscsites:-ne, 2*nscsites:-ne] = P[nscsites:nbs, nscsites:nbs] 
    rdm1[2*nscsites:-ne, -ne:] = P[nscsites:nbs, -ne:] 
    rdm1[-ne:, 2*nscsites:-ne] = P[-ne:, nscsites:nbs] 
    rdm1[-ne:, -ne:] = P[-ne:, -ne:]

    return rdm1
 
