from HubPhSymm import *
from libdmet.routine.slater_helper import transform_imp
import numpy as np
from math import copysign

def RHartreeFock(Lat, v, filling, mu0):
    rho, mu, E, res = HF(Lat, v, filling, True, mu0 = mu0, beta = np.inf, ires = True)
    log.result("Local density matrix (mean-field):\n%s", rho[0][0])
    log.result("Chemical potential (mean-field) = %20.12f", mu)
    log.result("Energy per site (mean-field) = %20.12f", E/Lat.supercell.nsites)
    log.result("Gap (mean-field) = %20.12f" % res["gap"])
    return rho, mu

def HartreeFock(Lat, v, filling, mu0):
    rho, mu, E, res = HF(Lat, v, filling, False, mu0 = mu0, beta = np.inf, ires = True)
    log.result("Local density matrix (mean-field):\n%s\n%s", rho[0][0], rho[1][0])
    log.result("Chemical potential (mean-field) = %20.12f", mu)
    log.result("Energy per site (mean-field) = %20.12f", E/Lat.supercell.nsites)
    log.result("Gap (mean-field) = %20.12f" % res["gap"])
    return rho, mu

def transformResults(rhoEmb, E, basis, ImpHam, H1e):
    spin = rhoEmb.shape[0]
    nscsites = basis.shape[2]
    rhoImp, Efrag, nelec = slater.transformResults(rhoEmb, E, basis, ImpHam, H1e)
    if Efrag is None:
        return nelec/nscsites
    else:
        log.result("Local density matrix (impurity):")
        for s in range(spin):
            log.result("%s", rhoImp[s])
        log.result("nelec per site (impurity) = %20.12f", nelec/nscsites)
        log.result("Energy per site (impurity) = %20.12f", Efrag/nscsites)

        return rhoImp, Efrag/nscsites, nelec/nscsites

def apply_dmu(lattice, ImpHam, basis, dmu):
    nscsites = lattice.supercell.nsites  
    if ImpHam.restricted:
        ImpHam.H1["cd"][0] -= transform_imp(basis[0], lattice, dmu * np.eye(nscsites))
    else:
        ImpHam.H1["cd"][0] -= transform_imp(basis[0], lattice, dmu * np.eye(nscsites))
        ImpHam.H1["cd"][1] -= transform_imp(basis[1], lattice, dmu * np.eye(nscsites))
    ImpHam.H0 += dmu * nscsites * 2
    return ImpHam

def SolveImpHam_with_dmu(lattice, ImpHam, basis, dmu, solver, solver_args = {}):
    # H = H1 + Vcor - Mu
    # to keep H for mean-field Mu->Mu+dMu, Vcor->Vcor+dMu
    # In impurity Ham, equivalent to substracting dMu from impurity, but not bath
    # The evaluation of energy is not affected if using (corrected) ImpHam-dMu
    # alternatively, we can change ImpHam.H0 to compensate
    ImpHam = apply_dmu(lattice, ImpHam, basis, dmu)
    result = solver.run(ImpHam, **solver_args)
    ImpHam = apply_dmu(lattice, ImpHam, basis, -dmu)
    return result

def SolveImpHam_with_fitting(lattice, filling, ImpHam, basis, solver, \
        solver_args = {}, delta = 0.02, thrnelec = 1e-4, step = 0.05):
    solve_with_mu = lambda mu: SolveImpHam_with_dmu(lattice, ImpHam, basis, \
            mu, solver, solver_args)
    rhoEmb, EnergyEmb = solve_with_mu(0.)
    nelec = transformResults(rhoEmb, None, basis, None, None)
    log.result("nelec = %20.12f (target is %20.12f)", nelec, filling*2)

    if abs(nelec/(filling*2) - 1.) < thrnelec:
        log.info("chemical potential fitting unnecessary")
        return rhoEmb, EnergyEmb, ImpHam, 0.
    else:
        delta *= -1. if (nelec > filling*2) else 1.
        log.result("chemical potential fitting:\n" \
                "finite difference dMu = %20.12f" % delta)
        rhoEmb1, EnergyEmb1 = solve_with_mu(delta)
        nelec1 = transformResults(rhoEmb1, None, basis, None, None)
        log.result("nelec = %20.12f (target is %20.12f)", nelec1, filling*2)
        if abs(nelec1/(filling*2) - 1.) < thrnelec:
            ImpHam = apply_dmu(lattice, ImpHam, basis, delta)
            return rhoEmb1, EnergyEmb1, ImpHam, delta
        else:
            nprime = (nelec1 - nelec) / delta
            delta1 = (filling*2 - nelec) / nprime
            if abs(delta1) > step:
                delta1 = copysign(0.1, delta1)
            log.info("dMu = %20.12f nelec = %20.12f", 0., nelec)
            log.info("dMu = %20.12f nelec = %20.12f", delta, nelec1)
            log.result("extrapolated to dMu = %20.12f", delta1)
            rhoEmb2, EnergyEmb2 = solve_with_mu(delta1)
            ImpHam = apply_dmu(lattice, ImpHam, basis, delta1)
            return rhoEmb2, EnergyEmb2, ImpHam, delta1

def AFInitGuess(ImpSize, U, Filling, polar = None):
    subA, subB = BipartiteSquare(ImpSize)
    nscsites = len(subA) + len(subB)    
    v = VcorLocal(False, False, nscsites)
    shift = U * Filling
    if polar is None:
        polar = shift * Filling
    init_v = np.eye(nscsites) * shift
    init_p = np.diag(map(lambda s: polar if s in subA else -polar, range(nscsites)))
    v.assign(np.asarray([init_v+init_p, init_v-init_p]))
    return v

def addDiag(v, scalar):
    rep = v.get()
    spin = rep.shape[0]
    nscsize = rep.shape[1]
    for s in range(spin):
        rep[s] += np.eye(nscsize) * scalar 
    v.assign(rep)
    return v

def VcorLocal(restricted, bogoliubov, nscsites):
    if restricted:
        nV = nscsites * (nscsites + 1) / 2
    else:
        nV = nscsites * (nscsites + 1)

    if bogoliubov and restricted:
        nD = nscsites * (nscsites + 1) / 2
    elif bogoliubov and not restricted:
        nD = nscsites * nscsites
    else:
        nD = 0

    v = vcor.Vcor()
    v.grad = None

    if restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter, require %s", (nV,))
            V = np.zeros((1, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 1, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                self.grad = g
            return self.grad

    elif not restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx + nV/2]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx+nV/2,1,i,j] = g[idx+nV/2,1,j,i] = 1
                self.grad = g
            return self.grad

    elif restricted and bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx]
                V[2,i,j] = V[2,j,i] = self.param[idx+nV]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = 1
                    g[idx+nV,2,i,j] = g[idx+nV,2,j,i] = 1
                self.grad = g
            return self.grad

    else:
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx+nV/2]
            for idx, (i,j) in enumerate(it.product(range(nscsites), repeat = 2)):
                V[2,i,j] = self.param[idx+nV]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx+nV/2,1,i,j] = g[idx+nV/2,1,j,i] = 1
                for idx, (i,j) in enumerate(it.product(range(nscsites), repeat = 2)):
                    g[idx+nV,2,i,j] = 1
                self.grad = g
            return self.grad

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.length = types.MethodType(lambda self: nV+nD, v)
    return v

FitVcor = FitVcorTwoStep
