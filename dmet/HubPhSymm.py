from libdmet.system.lattice import ChainLattice, SquareLattice, CubicLattice, HoneycombLattice, BipartiteSquare
from libdmet.system.hamiltonian import HubbardHamiltonian as Ham
from libdmet.routine import vcor, slater
from libdmet.routine.mfd import HF
from libdmet.routine.diis import FDiisContext
import libdmet.utils.logger as log
from libdmet.solver import impurity_solver
import types
import numpy as np
import numpy.linalg as la
import itertools as it
from libdmet.routine.slater_helper import transform_trans_inv_sparse

def HartreeFock(Lat, v, U):
    rho, mu, E, res = HF(Lat, v, 0.5, False, mu0 = U/2, beta = np.inf, ires = True)
    log.result("Local density matrix (mean-field):\n%s\n%s", rho[0][0], rho[1][0])
    log.result("Chemical potential (mean-field) = %20.12f", mu)
    log.result("Energy per site (mean-field) = %20.12f", E/Lat.supercell.nsites)
    log.result("Gap (mean-field) = %20.12f" % res["gap"])
    return rho, mu

def basisMatching(basis):
    basisA, basisB = basis[0], basis[1]
    S = np.tensordot(basisA, basisB, axes = ((0,1), (0,1)))
    # S=A^T*B svd of S is UGV^T then we let A'=AU, B'=BV
    # yields A'^T*B'=G diagonal and optimally overlapped
    u, gamma, vt = la.svd(S)
    log.result("overlap statistics:\n larger than 0.9: %3d  smaller than 0.9: %3d\n"
            " average: %10.6f  min: %10.6f", \
            np.sum(gamma > 0.9), np.sum(gamma < 0.9), np.average(gamma), np.min(gamma))
    basisA = np.tensordot(basisA, u, axes = (2, 0))
    basisB = np.tensordot(basisB, vt, axes = (2, 1))
    return np.asarray([basisA, basisB])

def ConstructImpHam(Lat, rho, v, matching = True, local = True, split = False, **kwargs):
    log.result("Making embedding basis")
    basis = slater.embBasis(Lat, rho, local = local)
    if matching and basis.shape[0] == 2:
        log.result("Rotate bath orbitals to match alpha and beta basis")
        nscsites = Lat.supercell.nsites
        if local:
            basis[:, :, :, nscsites:] = basisMatching(basis[:, :, :, nscsites:])
        else:
            # split matching occ and virt
            if split:
                basis[:, :, :, :nscsites] = basisMatching(basis[:, :, :, :nscsites])
                basis[:, :, :, nscsites:] = basisMatching(basis[:, :, :, nscsites:])
            else:
                basis = basisMatching(basis)
            
    log.result("Constructing impurity Hamiltonian")
    ImpHam, H1e = slater.embHam(Lat, basis, v, local = local, **kwargs)

    return ImpHam, H1e, basis

def transformResults(rhoEmb, E, basis, ImpHam, H1e):
    spin = rhoEmb.shape[0]
    nscsites = basis.shape[2]
    rhoImp, Efrag, nelec = slater.transformResults(rhoEmb, E, basis, ImpHam, H1e)

    log.result("Local density matrix (impurity):")
    for s in range(spin):
        log.result("%s", rhoImp[s])
    log.result("nelec per site (impurity) = %20.12f", nelec/nscsites)
    log.result("Energy per site (impurity) = %20.12f", Efrag/nscsites)

    return rhoImp, Efrag/nscsites, nelec/nscsites

def InitGuess(ImpSize, U, polar = None):
    subA, subB = BipartiteSquare(ImpSize)
    v = VcorLocalPhSymm(U, False, subA, subB)
    nscsites = len(subA) + len(subB)
    if polar is None:
        polar = U/2
    init_v = np.diag(map(lambda s: polar if s in subA else -polar, range(nscsites)))
    v.assign(np.asarray([init_v, -init_v]))
    return v

def VcorLocalPhSymm(U, bogoliubov, subA, subB):
    # with particle-hole symmetry, on two sublattices
    # specifically for t'=0 Hubbard model at half-filling
    # unrestricted potential is assumed
    # the symmetry is
    # VA_{ij} + (-)^{i+j}VB_{ij} = 0
    # D_{ij} = (-)^{i+j}D_{ji}
    # AA=+, AB=-, BB=+
    subA, subB = set(subA), set(subB)
    log.eassert(len(subA) == len(subB), "number of sites in two sublattices are equal")
    nscsites = len(subA) * 2
    log.eassert(subA | subB == set(range(nscsites)), "sublattice designation problematic")
    nV = nscsites * (nscsites+1) / 2

    v = vcor.Vcor()
    v.grad = None

    def sign(i,j):
        if (i in subA) == (j in subA):
            return 1
        else:
            return -1

    if bogoliubov:
        nD = nscsites * (nscsites+1) / 2
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = -self.param[idx] * sign(i,j)
            for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                V[2,i,j] = self.param[idx+nV]
                if i != j:
                    V[2,j,i] = self.param[idx+nV] * sign(i,j)
            V[0] += np.eye(nscsites) * (U/2)
            V[1] += np.eye(nscsites) * (U/2)
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = -sign(i,j)
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx+nV,2,i,j] = 1
                    if i != j:
                        g[idx+nV,2,j,i] = sign(i,j)
                self.grad = g
            return self.grad
    else:
        nD = 0
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = -self.param[idx] * sign(i,j)
            V[0] += np.eye(nscsites) * (U/2)
            V[1] += np.eye(nscsites) * (U/2)
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = -sign(i,j)
                self.grad = g
            return self.grad

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.length = types.MethodType(lambda self: nV+nD, v)
    return v

def FitVcor(rho, lattice, basis, vcor, beta, MaxIter1 = 300, MaxIter2 = 20):
    return slater.FitVcorTwoStep(rho, lattice, basis, vcor, beta, 0.5, \
            MaxIter1, MaxIter2)

class IterHistory(object):
    def __init__(self):
        self.history = []

    def update(self, energy, err, nelec, dvcor, dc):
        self.history.append([energy, err, nelec, dvcor, dc.nDim, dc.iNext])
        log.section("\nDMET Progress\n")
        log.result("  Iter         Energy               RdmErr         " \
            "       Nelec                 dVcor      DIIS")
        for idx, item in enumerate(self.history):
            log.result(" %3d %20.12f %20.12f %20.12f %20.12f  %2d %2d", idx, *item)
        log.result("")

foldRho = slater.foldRho
