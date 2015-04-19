import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log

class HamNonInt(object):
    def __init__(self, lattice, H1, H2, Fock = None):
        ncells = lattice.ncells
        nscsites = lattice.supercell.nsites
        log.eassert(H1.shape == (ncells, nscsites, nscsites), \
            "H1 shape not compatible with lattice")
        self.H1 = H1
        if Fock is None:
            self.Fock = H1
        else:
            log.eassert(Fock.shape == H1.shape, "Fock shape not compatible with lattice")
            self.Fock = Fock
        log.eassert(H2.shape == (nscsites,) * 4, "H2 shape not compatible with supercell")
        self.H2 = H2

    def getH1(self):
        return self.H1

    def getH2(self):
        return self.H2

    def getFock(self):
        return self.Fock

def HubbardHamiltonian(lattice, U, tlist):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    H1 = np.zeros((ncells, nscsites, nscsites))

    for order, t in enumerate(tlist):
        if abs(t < 1e-7):
            continue
        log.eassert(order < len(lattice.neighborDist),
            "%dth near neighbor distance unspecified in Lattice object", order+1)
        dis = lattice.neighborDist[order]
        log.warning("Searching neighbor within only one supercell")
        pairs = lattice.neighbor(dis = dis, sitesA = range(nscsites))
        for i, j in pairs:
            H1[j / nscsites, i, j % nscsites] = -t

    H2 = np.zeros((nscsites,) * 4)
    for s in range(nscsites):
        H2[s,s,s,s] = U

    return HamNonInt(lattice, H1, H2)
