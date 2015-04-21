import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log

def MatSqrt(M):
    log.eassert(la.norm(M - M.T.conj()) < 1e-10, "matrix must be symmetric")
    ew, ev = la.eigh(M)
    log.eassert((ew >= 0).all(), "matrix must be positive definite")
    log.check(ew[0] > 1e-10, \
        "small eigenvalue for rho_imp, cut-off is recommended\nthe first 5 eigenvalues are %s", ew[:5])
    ewsq = np.sqrt(ew).real
    return np.dot(ev, np.diag(ewsq))

def normBasis(b):
    n = np.diag(1./np.sqrt(np.diag(np.dot(b.T, b))).real)
    return np.dot(b, n)
    

def embBasis(lattice, rho, local = True, *args):
    if local:
        return embBasis_proj(lattice, rho, *args)
    else:
        return embBasis_phsymm(lattice, rho, *args)

def embBasis_proj(lattice, rho, *args):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    spin = rho.shape[0]
    basis = np.zeros((spin, nscsites * ncells, nscsites * 2))
    for s in range(spin):
        A = MatSqrt(rho[s,0])
        B = np.swapaxes(rho[s,1:], 0, 1).reshape((nscsites, -1))
        B = np.dot(la.inv(A), B).T
        B = normBasis(B)
        basis[s, :nscsites, :nscsites] = np.eye(nscsites)
        basis[s, nscsites:, nscsites:] = B
    return basis
