# define type/symmetry of correlation potential
# potential fitting algorithms using the symmetry
# initial guess

import itertools as it
import numpy as np
import numpy.linalg as la
import libdmet.utils.logger as log
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

    def get(self, i = 0, kspace = True):
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

    def assign(self, v0):
        log.eassert(self.islocal(), "This routine is for local vcor only")
        log.eassert(v0.shape == self.gradient().shape[1:], \
            "The correlation potential should have shape %s, rather than %s",
            self.gradient().shape[1:], v0.shape)

        # v_empty
        self.update(np.zeros(self.length()))
        v_empty = self.get()
        v0prime = v0 - v_empty
        param = np.empty(self.length())
        g = self.gradient()
        for i in range(self.length()):
            param[i] = np.sum(g[i] * v0prime) / np.sum(g[i] * g[i])
        self.update(param)
        log.check(la.norm(v0-self.get()) < 1e-7, \
                "symmetrization imposed on initial guess")
        #def fn(x):
        #    self.update(x)
        #    return np.sum((self.get() - v0) ** 2)

        #res = minimize(fn, np.zeros(self.length()), tol = 1e-10)
        #log.check(fn(res.x) < 1e-6, "symmetrization imposed on initial guess")

    def __str__(self):
        return self.evaluate().__str__()

def VcorNonLocl(restricted, bogoliubov, ncells, nscsites):
    # need to replace __call__ function
    log.error("VcorNonLocal not implemented yet")

def test():
    log.result("Test resctricted potential")
    vcor = VcorLocal(True, False, 4)
    vcor.update(np.asarray([2,1,0,-1,3,4,2,1,2,3]))
    log.result("Vcor:\n%s", vcor.get())
    log.result("Gradient:\n%s", vcor.gradient())

    log.result("Test unresctricted potential")
    vcor = VcorLocal(False, False, 2)
    vcor.update(np.asarray([2,1,0,-1,3,4]))
    log.result("Vcor:\n%s", vcor.get())
    log.result("Gradient:\n%s", vcor.gradient())

    log.result("Test unresctricted Bogoliubov potential")
    vcor = VcorLocal(False, True, 2)
    vcor.update(np.asarray([1,2,3,4,5,6,7,8,9,10]))
    log.result("Vcor:\n%s", vcor.get())
    log.result("Gradient:\n%s", vcor.gradient())

if __name__ == "__main__":
    test()
