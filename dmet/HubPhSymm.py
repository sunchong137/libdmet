from libdmet.system.lattice import ChainLattice, SquareLattice, CubicLattice, HoneycombLattice, BipartiteSquare
from libdmet.system.hamiltonian import HubbardHamiltonian as Ham
from libdmet.routine import vcor, slater
from libdmet.routine.mfd import HF
from libdmet.solver import block
import libdmet.utils.logger as log
import types
import numpy as np
import itertools as it

solver = block.Block()
schedule = block.Schedule()

def HartreeFock(Lat, v, U):
    rho, mu, E, res = HF(Lat, v, 0.5, False, mu0 = U/2, beta = np.inf, ires = True)
    log.result("Local density matrix:\n%s\n%s", rho[0][0], rho[1][0])
    log.result("Chemical potential = %20.12f\tEnergy = %20.12f", mu, E)
    log.result("Gap = %20.12f" % res["gap"])
    return rho, mu

def ConstructImpHam(Lat, rho, v):
    log.result("Making embedding basis")    
    basis = slater.embBasis(Lat, rho, local = True)
    log.result("Constructing impurity Hamiltonian")    
    ImpHam, H1e = slater.embHam(Lat, basis, v, local = True)

    return ImpHam, H1e, basis

def SolveImpHam(ImpHam, basis, M):
    if not solver.sys_initialized:
        solver.set_system(ImpHam.norb, 0, False, False, False)     
    if not solver.optimized:
        schedule.gen_initial(minM = 100, maxM = M)
    else:
        schedule.gen_restart(M)
    solver.set_schedule(schedule)
    solver.set_integral(ImpHam)

    truncation, energy, onepdm = solver.optimize()
    return onepdm, energy

def InitGuess(ImpSize, U, polar = None):
    subA, subB = BipartiteSquare(ImpSize)
    v = VcorLocalPhSymm(U, False, subA, subB)
    nscsites = len(subA) + len(subB)
    if polar is None:
        polar = U/2
    init_v = np.diag(map(lambda s: polar if s in subA else -polar, range(nscsites)))
    vcor.VcorGuess(v, np.asarray([init_v, -init_v]))
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
