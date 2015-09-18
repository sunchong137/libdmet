import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log
from libdmet.utils.misc import mdot
from copy import deepcopy

def extractRdm(GRho):
    norbs = GRho.shape[0] / 2
    log.eassert(norbs * 2 == GRho.shape[0], \
            "generalized density matrix dimension error")
    rhoA = deepcopy(GRho[:norbs, :norbs])
    rhoB = np.eye(norbs) - GRho[norbs:, norbs:]
    kappaBA = GRho[norbs:,:norbs]
    return rhoA, rhoB, kappaBA

def extractH1(GFock):
    norbs = GFock.shape[0] / 2
    log.eassert(norbs * 2 == GFock.shape[0], \
            "generalized density matrix dimension error")
    HA = deepcopy(GFock[:norbs, :norbs])
    HB =  - GFock[norbs:, norbs:]
    HDT = deepcopy(GFock[norbs:,:norbs])
    return HA, HB, HDT

def combineRdm(rhoA, rhoB, kappaAB):
    norbs = rhoA.shape[0]
    return np.vstack((
        np.hstack((      rhoA,             -kappaAB)),
        np.hstack((-kappaAB.T, np.eye(norbs) - rhoB))
    ))

def swapSpin(GRho):
    rhoA, rhoB, kappaBA = extractRdm(GRho)
    norbs = rhoA.shape[0]
    return np.vstack((
        np.hstack((      rhoB,             -kappaBA)),
        np.hstack((-kappaBA.T, np.eye(norbs) - rhoA))
    ))

def basisToCanonical(basis):
    assert(basis.shape[0] == 2)
    shape = list(basis.shape[1:])
    nbasis = shape[-1]
    nsites = shape[-2] / 2
    shape[-1] *= 2
    newbasis = np.empty(tuple(shape))
    newbasis[...,:nbasis] = basis[0]
    newbasis[...,:nsites,nbasis:], newbasis[...,nsites:,nbasis:] = \
            basis[1,...,nsites:,:], basis[1,...,:nsites,:]
    return newbasis

def basisToSpin(basis):
    shape = [2] + list(basis.shape)
    shape[-1] /= 2
    nbasis = shape[-1]
    nsites = shape[-2] / 2
    newbasis = np.empty(tuple(shape))
    newbasis[0] = basis[...,:nbasis]
    newbasis[1,...,:nsites,:], newbasis[1,...,nsites:,:] = \
            basis[...,nsites:,nbasis:], basis[...,:nsites,nbasis:]
    return newbasis

def mono_fit(fn, y0, x0, thr, increase = True):
    if not increase:
        return mono_fit(lambda x: -fn(x), -y0, x0, thr, True)

    from libdmet.utils.misc import counted
    @counted
    def evaluate(xx):
        yy = fn(xx)
        log.debug(1, "Iter %2d, x = %20.12f, f(x) = %20.12f", \
                evaluate.count-1, xx, yy)
        return yy

    log.debug(0, "target f(x) = %20.12f", y0)
    # first section search
    x = x0
    y = evaluate(x)
    if abs(y - y0) < thr:
        return x

    if y > y0:
        dx = -1.
    else:
        dx = 1.

    while 1:
        x1 = x + dx
        y1 = evaluate(x1)
        if abs(y1 - y0) < thr:
            return x1
        if (y-y0) * (y1-y0) < 0:
            break
        else:
            x = x1
            y = y1

    if x < x1:
        sec_x, sec_y = [x, x1], [y, y1]
    else:
        sec_x, sec_y = [x1, x], [y1, y]

    while sec_x[1] - sec_x[0] > 0.1 * thr:
        f = (y0-sec_y[0]) / (sec_y[1] - sec_y[0])
        x1 = sec_x[0] * (1.-f) + sec_x[1] * f
        y1 = evaluate(x1)
        if abs(y1 - y0) < thr:
            return x1
        if (y1 - y0) * (sec_y[0] - y0) < 0:
            sec_x = [sec_x[0], x1]
            sec_y = [sec_y[0], y1]
        else:
            sec_x = [x1, sec_x[1]]
            sec_y = [y1, sec_y[1]]

    return 0.5 * (sec_x[0] + sec_x[1])

def separate_basis(basis):
    nscsites = basis.shape[2] / 2
    # VA, VB, UA, UB
    return basis[0, :, :nscsites], basis[1, :, :nscsites], \
            basis[1, :, nscsites:], basis[0, :, nscsites:]

def contract_trans_inv(basisL, basisR, lattice, H):
    ncells = lattice.ncells
    nbasisL = basisL.shape[2]
    nbasisR = basisR.shape[2]
    res = np.zeros((nbasisL, nbasisR))
    for i, j in it.product(range(ncells), repeat = 2):
        res += mdot(basisL[i].T, H[lattice.subtract(j,i)], basisR[j])
    return res

def transform_trans_inv(basis, lattice, H):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 3:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    D_T = lattice.transpose(D)
    ctr = lambda L, H, R: contract_trans_inv(L, R, lattice, H)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D_T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D_T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D_T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D_T, UA))
    return np.asarray([resHA, resHB]), resD, resE0

def contract_trans_inv_sparse(basisL, basisR, lattice, H, thr = 1e-7):
    ncells = lattice.ncells
    nbasisL = basisL.shape[2]
    nbasisR = basisR.shape[2]
    res = np.zeros((nbasisL, nbasisR))
    from libdmet.utils.misc import find
    mask_basisL = find(True, map(lambda a: la.norm(a) > thr, basisL))
    mask_basisR = find(True, map(lambda a: la.norm(a) > thr, basisR))
    mask_H = find(True, map(lambda a: la.norm(a) > thr, H))

    for i, j in it.product(mask_basisL, mask_basisR):
        Hidx = lattice.subtract(j, i)
        if Hidx in mask_H:
            res += mdot(basisL[i].T, H[Hidx], basisR[j])
    return res

def transform_trans_inv_sparse(basis, lattice, H, thr = 1e-7):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 3:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    D_T = lattice.transpose(D)
    ctr = lambda L, H, R: contract_trans_inv_sparse(L, R, lattice, H, thr = thr)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D_T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D_T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D_T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D_T, UA))
    return np.asarray([resHA, resHB]), resD, resE0

def contract_local(basisL, basisR, lattice, H):
    ncells = lattice.ncells
    return reduce(np.ndarray.__add__, \
            map(lambda i: mdot(basisL[i].T, H, basisR[i]), range(ncells)))

def transform_local(basis, lattice, H):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 2:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    ctr = lambda L, H, R: contract_local(L, R, lattice, H)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D.T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D.T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D.T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D.T, UA))
    return np.asarray([resHA, resHB]), resD, resE0

def contract_local_sparseH(basisL, basisR, lattice, H, thr = 1e-7):
    ncells = lattice.ncells
    nbasisL = basisL.shape[-1]
    nbasisR = basisR.shape[-1]
    res = np.zeros((nbasisL, nbasisR))
    mask_H = np.nonzero(abs(H) > thr)
    mask_H = zip(*map(lambda a: a.tolist(), mask_H))
    for j, k in mask_H:
        res += np.dot(basisL[:,j].T, basisR[:,k]) * H[j,k]
    return res

def transform_local_sparseH(basis, lattice, H, thr = 1e-7):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 2:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    ctr = lambda L, H, R: contract_local_sparseH(L, R, lattice, H, thr = 1e-7)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D.T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D.T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D.T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D.T, UA))
    return np.asarray([resHA, resHB]), resD, resE0

def transform_imp(basis, lattice, H):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 2:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    ctr = lambda L, H, R: mdot(L[0].T, H, R[0])
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D.T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D.T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D.T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D.T, UA))
    return np.asarray([resHA, resHB]), resD, resE0

def contract_imp_env(basisL, basisR, lattice, H):
    ncells = lattice.ncells
    res1 = reduce(np.ndarray.__add__, \
            map(lambda i: mdot(basisL[0].T, H[i], basisR[i]), range(ncells)))
    res2 = reduce(np.ndarray.__add__, \
            map(lambda i: mdot(basisL[i].T, H[lattice.subtract(0,i)], basisR[0]), \
            range(ncells)))
    return 0.5 * (res1 + res2)

def transform_imp_env(basis, lattice, H):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 3:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    D_T = lattice.transpose(D)
    ctr = lambda L, H, R: contract_imp_env(L, R, lattice, H)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D_T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D_T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D_T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D_T, UA))
    return np.asarray([resHA, resHB]), resD, resE0
