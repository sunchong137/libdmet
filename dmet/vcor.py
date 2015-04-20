# define type/symmetry of correlation potential
# potential fitting algorithms using the symmetry
# initial guess

import itertools as it
import numpy as np
import libdmet.utils.logger as log
import types
from scipy.optimize import minimize

class Vcor(object):
    def __init__(self):
        self.param = None
        self.value = None

    def update(self, param):
        self.param = param
        self.value = self.evaluate()

    def islocal(self):
        return True

    def __call__(self, i = 0, kspace = True):
        log.eassert(self.value is not None, "Vcor not initialized yet")
        if kspace or i == 0:
            return self.value
        else:
            return np.zeros_like(self.value)

    def evaluate(self):
        log.error("function evaulate() is not implemented")

    def gradient(self):
        log.error("function gradient() is not implemented")

    def length(self):
        log.error("function len() is not implemented")

def VcorGuess(obj_v, v0):
    log.eassert(obj_v.islocal(), "This routine is for local vcor")
    log.eassert(v0.shape == obj_v.gradient().shape[1:], \
        "The initial guess should have shape %s, rather than %s",
        v0.shape, obj_v.gradient().shape[1:])

    def fn(x):
        obj_v.update(x)
        return np.sum((obj_v() - v0) ** 2)

    res = minimize(fn, np.zeros(obj_v.length()))
    log.check(fn(res.x) < 1e-6, "symmetrization imposed on initial guess")

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

    vcor = Vcor()
    vcor.grad = None

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

    vcor.evaluate = types.MethodType(evaluate, vcor)
    vcor.gradient = types.MethodType(gradient, vcor)
    vcor.length = types.MethodType(lambda self: nV+nD, vcor)
    return vcor

def VcorLocalPhSymm(bogoliubov, subA, subB):
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

    vcor = Vcor()
    vcor.grad = None

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
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(range(nscsites), 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = -sign(i,j)
                self.grad = g
            return self.grad

    vcor.evaluate = types.MethodType(evaluate, vcor)
    vcor.gradient = types.MethodType(gradient, vcor)
    vcor.length = types.MethodType(lambda self: nV+nD, vcor)
    return vcor

def VcorNonLocl(restricted, bogoliubov, ncells, nscsites):
    # need to replace __call__ function
    log.error("VcorNonLocal not implemented yet")

def test():
    log.result("Test resctricted potential")
    vcor = VcorLocal(True, False, 4)
    vcor.update(np.asarray([2,1,0,-1,3,4,2,1,2,3]))
    log.result("Vcor:\n%s", vcor())
    log.result("Gradient:\n%s", vcor.gradient())

    log.result("Test unresctricted potential")
    vcor = VcorLocal(False, False, 2)
    vcor.update(np.asarray([2,1,0,-1,3,4]))
    log.result("Vcor:\n%s", vcor())
    log.result("Gradient:\n%s", vcor.gradient())

    log.result("Test unresctricted Bogoliubov potential")
    vcor = VcorLocal(False, True, 2)
    vcor.update(np.asarray([1,2,3,4,5,6,7,8,9,10]))
    log.result("Vcor:\n%s", vcor())
    log.result("Gradient:\n%s", vcor.gradient())

    log.result("Test particle-hole symmetric potential")
    vcor = VcorLocalPhSymm(True, [0,3],[1,2])
    vcor.update(np.asarray(range(1,21)))
    log.result("Vcor:\n%s", vcor())

if __name__ == "__main__":
    test()
