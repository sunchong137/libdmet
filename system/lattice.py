# File: lattice.py
# Author: Bo-Xiao Zheng <boxiao@princeton.edu>
#

import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log
import libdmet.system.hamiltonian as ham

def Frac2Real(cellsize, coord):
    assert(len(cellsize.shape) == 2 and cellsize.shape[0] == cellsize.shape[1])
    assert(len(coord.shape) == 1)
    return np.dot(coord, cellsize)

def Real2Frac(cellsize, coord):
    assert(len(cellsize.shape) == 2 and cellsize.shape[0] == cellsize.shape[1])
    assert(len(coord.shape) == 1)
    return np.dot(coord, la.inv(cellsize))

def ChainLattice(length, scsites):
    log.eassert(length % scsites == 0, "incompatible lattice size and supercell size")
    uc = UnitCell(np.eye(1), [(np.array([0]), "X")])
    sc = SuperCell(uc, np.array([scsites]))
    lat = Lattice(sc, np.array([length / scsites]))
    lat.neighborDist = [1.,2.,3.]
    return lat

def SquareLattice(lx, ly, scx, scy):
    log.eassert(lx % scx == 0 and ly % scy == 0, "incompatible lattice size and supercell size")
    uc = UnitCell(np.eye(2), [(np.array([0, 0]), "X")])
    sc = SuperCell(uc, np.array([scx, scy]))
    lat = Lattice(sc, np.array([lx / scx, ly / scy]))
    lat.neighborDist = [1., 2.**0.5, 2.]
    return lat

def HoneycombLattice(lx, ly, scx, scy):
    log.error("honeycomb lattice not implemented yet")

def CubicLattice(lx, ly, lz, scx, scy, scz):
    log.eassert(lx % scx == 0 and ly % scy == 0 and lz % scz == 0, "incompatible lattice size and supercell size")
    uc = UnitCell(np.eye(3), [(np.array([0, 0, 0]), "X")])
    sc = SuperCell(uc, np.array([scx, scy, scz]))
    lat = Lattice(sc, np.array([lx / scx, ly / scy, lz / scz]))
    lat.neighborDist = [1., 2.**0.5, 3.**0.5]
    return lat

def translateSites(baseSites, usize, csize):
    # csize = [3,3], then cells = [0,0], [0,1], [0,2], [0,3], [1,0], ..., [3,3]
    cells = map(np.asarray, it.product(*tuple(map(range, csize))))
    sites = list(it.chain.from_iterable(map(lambda c: \
            map(lambda s: np.dot(c, usize) + s, baseSites), cells)))
    return cells, sites

def BipartiteSquare(impsize):
    subA = []
    subB = []
    for idx, pos in enumerate(it.product(*map(range, impsize))):
        if np.sum(pos) % 2 == 0:
            subA.append(idx)
        else:
            subB.append(idx)
    log.eassert(len(subA) == len(subB), \
        "The impurity cannot be divided into two sublattices")
    return subA, subB

class UnitCell(object):
    def __init__(self, size, sites):
        # unit cell shape
        self.size = np.array(size)
        log.eassert(self.size.shape[0] == self.size.shape[1], "Invalid unitcell constants")
        self.dim = self.size.shape[0]
        for s in sites:
            log.eassert(s[0].shape == (self.dim,), "Invalid position for the site")
        self.sites = map(lambda site: np.asarray(site[0]), sites)
        self.names = map(lambda site: site[1], sites)
        self.nsites = len(self.sites)
        self.sitedict = dict(zip(map(tuple, self.sites), range(self.nsites)))

    def __str__(self):
        r = "UnitCell Shape\n%s\nSites:\n" % self.size
        for i in range(len(self.sites)):
          r += "%-10s%-10s\t" % (self.names[i], self.sites[i])
          if (i+1)%6 == 0:
            r+= "\n"
        r += "\n\n"
        return r

class SuperCell(object):
    def __init__(self, uc, size): # uc is unitcell
        self.unitcell = uc
        self.dim = uc.dim
        self.csize = np.array(size)
        self.size = np.dot(np.diag(self.csize), uc.size)
        self.ncells = np.product(self.csize)
        self.nsites = uc.nsites * self.ncells

        self.cells, self.sites = translateSites(uc.sites, uc.size, size)
        self.names = uc.names * self.ncells

        self.celldict = dict(zip(map(tuple, self.cells), range(self.ncells)))
        self.sitedict = dict(zip(map(tuple, self.sites), range(self.nsites)))

    def __str__(self):
        r = self.unitcell.__str__()
        r += "SuperCell Shape\n"
        r += self.size.__str__()
        r += "\nNumber of Sites:%d\n" % self.nsites
        r += "\n"
        return r

class Lattice(object):
    def __init__(self, sc, size):
        self.supercell = sc
        self.dim = sc.dim
        self.csize = np.array(size)
        self.size = np.dot(np.diag(self.csize), sc.size)
        self.ncells = np.product(self.csize)
        self.nsites = sc.nsites * self.ncells

        self.cells, self.sites = translateSites(sc.sites, sc.size, size)
        self.names = sc.names * self.ncells

        self.celldict = dict(zip(map(tuple, self.cells), range(self.ncells)))
        self.sitedict = dict(zip(map(tuple, self.sites), range(self.nsites)))
        self.neighborDist = []

    def __str__(self):
        r = self.supercell.__str__()
        r += "Lattice Shape\n%s\n" % self.csize
        r += "Number of SuperCells: %4d\n" % self.ncells
        r += "Number of Sites:      %4d\n" % self.nsites
        return r

    """
    functions on translations
    """

    def cell_idx2pos(self, idx):
        return self.cells[idx % self.ncells]

    def cell_pos2idx(self, pos):
        return self.celldict[tuple(pos % self.csize)]

    def site_idx2pos(self, idx):
        return self.sites[idx % self.ncells]

    def site_pos2idx(self, pos):
        return self.sitedict[tuple(pos % self.size)]

    def add(self, i, j):
        return self.cell_pos2idx(self.cell_idx2pos(i) + self.cell_idx2pos(j))

    def substract(self, i, j):
        return self.cell_pos2idx(self.cell_idx2pos(i) - self.cell_idx2pos(j))
    """
    functions on matrices in the system
    """

    def FFTtoK(self, A):
        # before FFT ncells * nscsites * nscsites where the first index is cell
        # after FFT first index is k-point
        return np.fft.fftn(A.reshape(tuple(self.csize) + A.shape[-2:]), \
            axes = range(self.dim)).reshape(A.shape)

    def FFTtoT(self, B):
        A = np.fft.ifftn(B.reshape(tuple(self.csize) + B.shape[-2:]), \
            axes = range(self.dim)).reshape(B.shape)
        if np.allclose(A.imag, 0.):
            return A.real
        else:
            return A

    def kpoints(self):
        return map(lambda d: np.fft.fftfreq(self.csize[d], 1./(2*np.pi)), range(self.dim))

    def expand(self, A, dense = False):
        # expand ncells * nscsites * nscsites translational invariant matrix to full
        # nsites * nsites matrix
        bigA = np.zeros((self.nsites, self.nsites))
        nscsites = self.supercell.nsites
        if dense:
            for i, j in it.product(range(self.ncells), repeat = 2):
                idx = self.add(i, j)
                bigA[i*nscsites:(i+1)*nscsites, idx*nscsites:(idx+1)*nscsites] = A[j]
        else:
            nonzero = filter(lambda j: not np.allclose(A[j], 0.), range(self.ncells))
            for i, j in it.product(range(self.ncells), nonzero):
                idx = self.add(i, j)
                bigA[i*nscsites:(i+1)*nscsites, idx*nscsites:(idx+1)*nscsites] = A[j]
        return bigA

    def transpose(self, A):
        # return the transpose of ncells * nscsites * nscsites translational invariant matrix
        AT = np.zeros_like(A)
        for n in range(self.ncells):
            AT[n] = A[self.cell_pos2idx(-self.cell_idx2pos(n))].T
        return AT

    """
    get neighbor sites
    """

    def neighbor(self, dis = 1., max_range = 1, sitesA = None, sitesB = None):
        # siteA, siteB are indices, not positions
        if sitesA is None:
            sitesA = range(self.nsites)
        if sitesB is None:
            sitesB = range(self.nsites)

        nscsites = self.supercell.nsites
        cellshifts = map(lambda s: self.cell_pos2idx(np.asarray(s)), \
            it.product(range(-max_range, max_range+1), repeat = self.dim))

        shifts = map(lambda s: np.asarray(s), it.product([-1, 0, 1], repeat = self.dim))

        neighbors = []
        for siteA in sitesA:
            cellA = siteA / nscsites
            cellB = map(lambda x: self.add(cellA, x), cellshifts)
            psitesB = list(set(sitesB) & \
                set(it.chain.from_iterable(map(lambda c:range(c*nscsites, (c+1)*nscsites), cellB))))

            for siteB in psitesB:
                for shift in shifts:
                    if abs(la.norm(self.sites[siteA] - self.sites[siteB] - np.dot(shift, self.size)) \
                        - dis) < 1e-5:
                        neighbors.append((siteA, siteB))
                        break
        return neighbors

    def setHam(self, Ham):
        self.Ham = Ham
        self.H1 = self.Ham.getH1()
        self.H1_kspace = self.FFTtoK(self.H1)
        self.Fock = self.Ham.getFock()
        self.Fock_kspace = self.FFTtoK(self.Fock)

    def getH1(self, kspace = True):
        if kspace:
            return self.H1_kspace
        else:
            return self.H1

    def getFock(self, kspace = True):
        if kspace:
            return self.Fock_kspace
        else:
            return self.Fock

    def getH2(self):
        return self.Ham.getH2()

    def getImpJK(self):
        return self.Ham.getImpJK()

def test():
    chain = ChainLattice(240, 4)
    log.result("%s", chain)
    log.result("kpoints: %s", chain.kpoints())
    log.result("neighbors: %s", chain.neighbor(sitesA = range(4)))
    log.result("")

    square = SquareLattice(72, 72, 2, 2)
    log.result("%s", square)
    log.result("1st neigbors: %s", square.neighbor(dis = 1., sitesA = range(4)))
    log.result("2nd neigbors: %s", square.neighbor(dis = 2**0.5, sitesA = range(4)))

    Ham = ham.HubbardHamiltonian(square, 4, [1, 0.])
    square.setHam(Ham)
    log.result("%s", square.getH1(kspace = True))
    log.result("%s", square.getFock(kspace = True))
    log.result("%s", square.getH2())

class Indexer(object):
    def __init__(self, size):
        self.size = size
        self.nsites = np.product(size)
        self.sites = map(np.asarray, list(it.product(*map(range, self.size))))
        self.sitedict = dict(zip(map(tuple, self.sites), range(self.nsites)))

    def idx2pos(self, idx):
        return self.sites[idx % self.nsites]

    def pos2idx(self, pos):
        return self.sitedict[tuple(pos % self.size)]

    def add(self, i, j):
        return self.pos2idx(self.idx2pos(i) + self.idx2pos(j))

    def substract(self, i, j):
        return self.pos2idx(self.idx2pos(i) - self.idx2pos(j))

if __name__ == "__main__":
    test()
