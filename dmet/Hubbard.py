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
    log.result("Local density matrix (mean-field): alpha and beta\n%s\n%s", \
            rho[0][0], rho[1][0])
    log.result("Chemical potential (mean-field) = %20.12f", mu)
    log.result("Energy per site (mean-field) = %20.12f", E/Lat.supercell.nsites)
    log.result("Gap (mean-field) = %20.12f" % res["gap"])
    return rho, mu

def transformResults(rhoEmb, E, basis, ImpHam, H1e):
    spin = rhoEmb.shape[0]
    nscsites = basis.shape[2]
    rhoImp, Efrag, nelec = slater.transformResults(rhoEmb, E, basis, ImpHam, H1e)
    log.debug(1, "impurity density matrix:\n%s", rhoImp)
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
    nbasis = basis.shape[-1]
    if ImpHam.restricted:
        ImpHam.H1["cd"][0] -= transform_imp(basis[0], lattice, dmu * np.eye(nscsites))
    else:
        ImpHam.H1["cd"][0] -= transform_imp(basis[0], lattice, dmu * np.eye(nscsites))
        ImpHam.H1["cd"][1] -= transform_imp(basis[1], lattice, dmu * np.eye(nscsites))
    ImpHam.H0 += dmu * nbasis
    return ImpHam

def SolveImpHam_with_dmu(lattice, ImpHam, basis, dmu, solver, solver_args = {}):
    # H = H1 + Vcor - Mu
    # to keep H for mean-field Mu->Mu+dMu, Vcor->Vcor+dMu
    # In impurity Ham, equivalent to subtracting dMu from impurity, but not bath
    # The evaluation of energy is not affected if using (corrected) ImpHam-dMu
    # alternatively, we can change ImpHam.H0 to compensate
    ImpHam = apply_dmu(lattice, ImpHam, basis, dmu)
    result = solver.run(ImpHam, **solver_args)
    ImpHam = apply_dmu(lattice, ImpHam, basis, -dmu)
    return result

# FIXME it is better to define this class in a file contained under the folder routine/
class MuSolver(object):
    def __init__(self, adaptive = True, trust_region = 2.5):
        self.adaptive = adaptive
        self.trust_region = trust_region
        self.history = []
        self.first_run = True

    def __call__(self, lattice, filling, ImpHam, basis, solver, \
            solver_args = {}, delta = 0.02, thrnelec = 1e-5, step = 0.05):
        solve_with_mu = lambda mu: SolveImpHam_with_dmu(lattice, ImpHam, basis, \
                mu, solver, solver_args)
        rhoEmb, EnergyEmb = solve_with_mu(0.)
        nelec = transformResults(rhoEmb, None, basis, None, None)
        record = [(0., nelec)]
        log.result("nelec = %20.12f (target is %20.12f)", nelec, filling*2)

        solver_args["similar"] = True
        #solver_args["similar"] = False

        if abs(nelec/(filling*2) - 1.) < thrnelec:
            log.info("chemical potential fitting unnecessary")
            self.history.append(record)
            return rhoEmb, EnergyEmb, ImpHam, 0.
        else:
            if self.adaptive:
                # predict delta using historic information
                temp_delta = self.predict(nelec, filling*2)
                if temp_delta is not None:
                    delta = temp_delta
                    step = abs(delta) * self.trust_region
                else:
                    delta = abs(delta) * (-1 if (nelec > filling*2) else 1)
            else:
                delta = abs(delta) * (-1 if (nelec > filling*2) else 1)

            log.result("chemical potential fitting:\n" \
                    "finite difference dMu = %20.12f" % delta)
            rhoEmb1, EnergyEmb1 = solve_with_mu(delta)
            nelec1 = transformResults(rhoEmb1, None, basis, None, None)
            record.append((delta, nelec1))
            log.result("nelec = %20.12f (target is %20.12f)", nelec1, filling*2)

            if abs(nelec1/(filling*2) - 1.) < thrnelec:
                ImpHam = apply_dmu(lattice, ImpHam, basis, delta)
                self.history.append(record)
                return rhoEmb1, EnergyEmb1, ImpHam, delta
            else:
                nprime = (nelec1 - nelec) / delta
                delta1 = (filling*2 - nelec) / nprime
                if abs(delta1) > step:
                    log.info("extrapolation dMu %20.12f more than trust step %20.12f", delta1, step)
                    delta1 = copysign(step, delta1)
                log.info("dMu = %20.12f nelec = %20.12f", 0., nelec)
                log.info("dMu = %20.12f nelec = %20.12f", delta, nelec1)
                log.result("extrapolated to dMu = %20.12f", delta1)
                rhoEmb2, EnergyEmb2 = solve_with_mu(delta1)
                nelec2 = transformResults(rhoEmb2, None, basis, None, None)
                record.append((delta1, nelec2))
                log.result("nelec = %20.12f (target is %20.12f)", nelec2, filling*2)
                
                if abs(nelec2/(filling*2) - 1.) < thrnelec:    

                    ImpHam = apply_dmu(lattice, ImpHam, basis, delta1)
                    self.history.append(record)
                    return rhoEmb2, EnergyEmb2, ImpHam, delta1
                else:
                    log.info("use quadratic fitting.")
                    from quad_fit import quad_fit
                    step *= 2.0
                    mus = np.array([0.0, delta, delta1])
                    delta_nelec = np.array([nelec, nelec1, nelec2]) - (filling * 2.0)
                    delta2, status = quad_fit(mus, delta_nelec, tol = 1e-12)
                    if not status:
                        log.info("quadratic fails, use linear extrapolation.")
                        slope = (nelec2 - nelec1) / (delta1 - delta)
                        intercept = nelec1 - slope * delta
                        delta2 = (filling * 2.0 - intercept) / slope

                    if abs(delta2) > step:
                        log.info("extrapolation dMu %20.12f more than trust step %20.12f", delta2, step)
                        delta2 = copysign(step, delta2)
                    
                    log.result("extrapolated to dMu = %20.12f", delta2)
                    rhoEmb3, EnergyEmb3 = solve_with_mu(delta2)
                    nelec3 = transformResults(rhoEmb3, None, basis, None, None)
                    record.append((delta2, nelec3))
                    log.result("nelec = %20.12f (target is %20.12f)", nelec3, filling * 2.0)
                    

                    ImpHam = apply_dmu(lattice, ImpHam, basis, delta2)
                    self.history.append(record)
                    return rhoEmb3, EnergyEmb3, ImpHam, delta2

    def save(self, filename):
        import pickle as p
        log.info("saving chemical potential fitting history to %s", filename)
        with open(filename, "w") as f:
            p.dump(self.history, f)

    def load(self, filename):
        import pickle as p
        log.info("loading chemical potential fitting history from %s", filename)
        with open(filename, "r") as f:
            self.history = p.load(f)

    def predict(self, nelec, target):
        # we assume the chemical potential landscape more or less the same for
        # previous fittings
        # the simplest thing to do is predicting a delta from each previous
        # fitting, and compute a weighted average. The weight should prefer
        # lattest runs, and prefer the fittigs that have has points close to
        # current and target filling
        from math import sqrt, exp
        vals = []
        weights = []

        # hyperparameters
        damp_factor = np.e
        sigma2, sigma3 = 0.00025, 0.0005

        for i, record in enumerate(self.history):
            # exponential
            weight = damp_factor ** (i + 1 - len(self.history))

            if len(record) == 1:
                val, weight = 0., 0.
                continue

            elif len(record) == 2:
                # we fit a line
                (mu1, n1), (mu2, n2) = record
                slope = (n2 - n1) / (mu2 - mu1)
                val = (target - nelec) / slope
                # weight factor
                metric = min(
                        (target-n1)**2 + (nelec-n2)**2,
                        (target-n2)**2 + (nelec-n1)**2)

                # Gaussian weight
                weight *= exp(- 0.5 * metric / sigma2)

            else: # len(record) == 3
                # we need to check data sanity: should be monotonic
                (mu1, n1), (mu2, n2), (mu3, n3) = sorted(record)
                if (not n1 < n2) or (not n2 < n3):
                    val, weight = 0., 0.
                    continue

                # parabola between mu1 and mu3, linear outside the region
                # with f' continuous
                a, b, c = np.dot(la.inv(np.asarray([
                    [mu1**2, mu1, 1],
                    [mu2**2, mu2, 1],
                    [mu3**2, mu3, 1]
                ])), np.asarray([n1,n2,n3]).reshape(-1,1)).reshape(-1)

                # if the parabola is not monotonic, use linear interpolation instead
                if mu1 < -0.5*b/a < mu3:
                    def find_mu(n):
                        if n < n2:
                            slope = (n2-n1) / (mu2-mu1)
                        else:
                            slope = (n2-n3) / (mu2-mu3)
                        return mu2 + (n-n2) / slope

                else:
                    def find_mu(n):
                        if n < n1:
                            slope = 2 * a * mu1 + b
                            return mu1 + (n-n1) / slope
                        elif n > n3:
                            slope = 2 * a * mu3 + b
                            return mu3 + (n-n3) / slope
                        else:
                            return 0.5 * (-b + sqrt(b**2 - 4 * a * (c-n))) / a

                val = find_mu(target) - find_mu(nelec)
                # weight factor
                metric = min(
                        (target-n1)**2 + (nelec-n2)**2,
                        (target-n1)**2 + (nelec-n3)**2,
                        (target-n2)**2 + (nelec-n1)**2,
                        (target-n2)**2 + (nelec-n3)**2,
                        (target-n3)**2 + (nelec-n1)**2,
                        (target-n3)**2 + (nelec-n2)**2,
                )
                weight *= exp(-0.5 * metric / sigma3)

            vals.append(val)
            weights.append(weight)

        log.debug(1, "dmu predictions:\n    value      weight")
        for v, w in zip(vals, weights):
            log.debug(1, "%10.6f %10.6f" % (v, w))

        if np.sum(weights) > 1e-3:
            dmu = np.dot(vals, weights) / np.sum(weights)
            if abs(dmu) > 0.5:
                dmu = copysign(0.5, dmu)
            log.info("adaptive chemical potential fitting, dmu = %20.12f", dmu)
            return dmu
        else:
            log.info("adaptive chemical potential fitting not used")
            return None

SolveImpHam_with_fitting = MuSolver(adaptive = True)

def AFInitGuess(ImpSize, U, Filling, polar = None, bogoliubov = False, rand = 0.):
    subA, subB = BipartiteSquare(ImpSize)
    nscsites = len(subA) + len(subB)    
    shift = U * Filling
    if polar is None:
        polar = shift * Filling
    init_v = np.eye(nscsites) * shift
    init_p = np.diag(map(lambda s: polar if s in subA else -polar, range(nscsites)))
    v = VcorLocal(False, bogoliubov, nscsites)
    if bogoliubov:
        np.random.seed(32499823)
        init_d = (np.random.rand(nscsites, nscsites) - 0.5) * rand
        v.assign(np.asarray([init_v+init_p, init_v-init_p, init_d]))
    else:
        v.assign(np.asarray([init_v+init_p, init_v-init_p]))
    return v

def PMInitGuess(ImpSize, U, Filling, bogoliubov = False, rand = 0.):
    nscsites = np.product(ImpSize)
    shift = U * Filling
    init_v = np.eye(nscsites) * shift
    v = VcorLocal(True, bogoliubov, nscsites)
    if bogoliubov:
        init_d = np.zeros((nscsites, nscsites)) 
        v.assign(np.asarray([init_v, int_v, init_d]))
    else:
        v.assign(np.asarray([init_v, init_v]))

    if rand > 0.:
        np.random.seed(32499823)
        v.update(v.param + (np.random.rand(v.length()) - 0.5) * rand)
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
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = 1
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

    else: # not restricted and bogoliubov
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

addDiag = slater.addDiag

FitVcor = slater.FitVcorTwoStep
