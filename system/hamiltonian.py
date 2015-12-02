import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log

class HamNonInt(object):
    def __init__(self, lattice, H1, H2, Fock = None, ImpJK = None, kspace_input = False):
        ncells = lattice.ncells
        nscsites = lattice.supercell.nsites
        log.eassert(H1.shape == (ncells, nscsites, nscsites), \
            "H1 shape not compatible with lattice")
        if kspace_input:
            self.H1 = lattice.FFTtoT(H1)
        else:
            self.H1 = H1
        if Fock is None:
            self.Fock = H1
        else:
            log.eassert(Fock.shape == H1.shape, "Fock shape not compatible with lattice")
            if kspace_input:
                self.Fock = lattice.FFTtoT(Fock)
            else:
                self.Fock = Fock
        if ImpJK is None:
            self.ImpJK = None
        else:
            log.eassert(ImpJK.shape == H1[0].shape, "ImpJK shape not compatible with supercell")
            self.ImpJK = ImpJK

        log.eassert(H2.shape == (nscsites,) * 4, "H2 shape not compatible with supercell")
        self.H2 = H2

    def getH1(self):
        return self.H1

    def getH2(self):
        return self.H2

    def getFock(self):
        return self.Fock

    def getImpJK(self):
        return self.ImpJK

def HubbardHamiltonian(lattice, U, tlist = [1.]):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    H1 = np.zeros((ncells, nscsites, nscsites))

    for order, t in enumerate(tlist):
        if abs(t) < 1e-7:
            continue
        log.eassert(order < len(lattice.neighborDist), \
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

def Hubbard3band(lattice, Ud, Up, ed, tpd, tpp):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    H1 = np.zeros((ncells, nscsites, nscsites))
    H2 = np.zeros((nscsites,) * 4)
    d_pd = lattice.neighborDist[0]
    d_pp = lattice.neighborDist[1]
    log.warning("Searching neighbor within only one supercell")
    pd_pairs = lattice.neighbor(dis = d_pd, sitesA = range(nscsites))
    for i, j in pd_pairs:
        H1[j / nscsites, i, j % nscsites] = tpd
    pp_pairs = lattice.neighbor(dis = d_pp, sitesA = range(nscsites))
    for i, j in pp_pairs:
        H1[j / nscsites, i, j % nscsites] = tpp
    for i, orb in enumerate(lattice.supercell.names):
        if orb == "Cu":
            H1[0,i,i] = ed
            H2[i,i,i,i] = Ud
        elif orb == "O":
            H2[i,i,i,i] = Up
        else:
            log.error("wrong orbital name %s in 3-band Hubbard model", orb)
    return HamNonInt(lattice, H1, H2)
