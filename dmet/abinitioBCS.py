from HubbardBCS import *
from abinitio import buildUnitCell, buildLattice, read_integral, \
        buildHamiltonian, AFInitGuessIdx, AFInitGuessOrbs
from abinitio import reportOccupation as report

def reportOccupation(lattice, GRho, names = None):
    rhoA, rhoB, kappaBA = extractRdm(GRho)
    report(lattice, np.asarray([rhoA, rhoB]), names)

def VcorRestricted(restricted, bogoliubov, active_sites, core_sites):
    # full correlation potential for active sites
    # diagonal potential for core sites
    nAct = len(active_sites)
    nCor = len(core_sites)
    nscsites = nAct + nCor

    if restricted:
        nV0 = nAct * (nAct + 1) / 2
        nV = nV0 + nCor
    else:
        nV0 = nAct * (nAct + 1)
        nV = nV0 + nCor * 2

    # no bogoliubov term on core sites
    if bogoliubov and restricted:
        nD = nAct * (nAct + 1) / 2
    elif bogoliubov and not restricted:
        nD = nAct * nAct
    else:
        nD = 0

    v = vcor.Vcor()
    v.grad = None

    if restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
            V = np.zeros((1, nscsites, nscsites))
            for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                V[0, i, j] = V[0, j, i] = self.param[idx]
            for idx, i in enumerate(core_sites):
                V[0, i, i] = self.param[nV0 + idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 1, nscsites, nscsites))
                for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                    g[idx, 0, i, j] = g[idx, 0, j, i] = 1
                for idx, i in enumerate(core_sites):
                    g[nV0 + idx, 0, i, i] = 1
                self.grad = g
            return self.grad

    elif not restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                V[0, i, j] = V[0, j, i] = self.param[idx]
                V[1, i, j] = V[1, j, i] = self.param[nV0/2 + idx]
            for idx, i in enumerate(core_sites):
                V[0, i, i] = self.param[nV0 + idx]
                V[1, i, i] = self.param[nV0 + nCor + idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                    g[idx, 0, i, j] = g[idx, 0, j, i] = 1
                    g[nV0/2 + idx, 1, i, j] = g[nV0/2 + idx, 1, j, i] = 1
                for idx, i in enumerate(core_sites):
                    g[nV0 + idx, 0, i, i] = 1
                    g[nV0 + nCor + idx, 1, i, i] = 1
                self.grad = g
            return self.grad

    elif restricted and bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                V[0, i, j] = V[0, j, i] = self.param[idx]
                V[1, i, j] = V[1, j, i] = self.param[idx]
                V[2, i, j] = V[2, j, i] = self.param[nV + idx]
            for idx, i in enumerate(core_sites):
                V[0, i, i] = V[1, i, i] = self.param[nV0 + idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                    g[idx, 0, i, j] = g[idx, 0, j, i] = 1
                    g[idx, 1, i, j] = g[idx, 1, j, i] = 1
                    g[nV + idx, 2, i, j] = g[nV + idx, 2, j, i] = 1
                for idx, i in enumerate(core_sites):
                    g[nV0 + idx, 0, i, i] = g[nV0 + idx, 1, i, i] = 1
                self.grad = g
            return self.grad

    else:
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                V[0, i, j] = V[0, j, i] = self.param[idx]
                V[1, i, j] = V[1, j, i] = self.param[nV0/2 + idx]
            for idx, i in enumerate(core_sites):
                V[0, i, i] = self.param[nV0 + idx]
                V[1, i, i] = self.param[nV0 + nCor + idx]
            for idx, (i, j) in enumerate(it.product(active_sites, repeat = 2)):
                V[2, i, j] = self.param[nV + idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                    g[idx, 0, i, j] = g[idx, 0, j, i] = 1
                    g[nV0/2 + idx, 1, i, j] = g[nV0/2 + idx, 1, j, i] = 1
                for idx, i in enumerate(core_sites):
                    g[nV0 + idx, 0, i, i] = 1
                    g[nV0 + nCor + idx, 1, i, i] = 1
                for idx, (i, j) in enumerate(it.product(active_sites, repeat = 2)):
                    g[nV + idx, 2, i, j] = 1
                self.grad = g
            return self.grad

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.length = types.MethodType(lambda self: nV + nD, v)
    return v
