import numpy as np
import numpy.linalg as la
from libdmet.system.lattice import Indexer
from libdmet.system import integral
from libdmet.system.hamiltonian import HamNonInt
from libdmet.routine.mfd import DiagRHF, assignocc
import libdmet.dmet.abinitio as dmet
import itertools as it
import matplotlib.pyplot as plt

def plotdos(ew, fermi, eta = 0.02, label = "DOS"):
    ew_sorted = sorted(ew.ravel())
    grid = np.linspace(ew_sorted[0]-eta*10, ew_sorted[-1]+eta*10, 1001)
    vals = np.zeros_like(grid)
    for e in ew_sorted:
        nonzero = np.nonzero((grid >= e-eta*10) * (grid <= e+eta*10))[0]
        for i in nonzero:
            vals[i] += eta / (eta**2 + (grid[i]-e)**2)
    vals /= np.sum(vals)
    plt.plot(grid - fermi, vals, label = label)


size0 = np.asarray([3,3])
size = np.asarray([10,10])

nelec_per_cell = 20

for x in size0:
    if x%2 != 1:
        print "Even number of original cells is not implemented yet."
        exit()

Ind0 = Indexer(size0)
Ind = Indexer(size)

center = size0 / 2

Fock0 = np.load("Fockoriginal.npy")
H10 = np.load("H1original.npy")

assert(Fock0.shape == H10.shape)
assert(Fock0.shape[0] == np.product(size0))
assert(Fock0.shape[1] == Fock0.shape[2])

nbasis = Fock0.shape[1]
Fock = np.zeros((np.product(size), nbasis, nbasis))
H1 = np.zeros((np.product(size), nbasis, nbasis))

for i in range(np.product(size0)):
    pos = Ind0.idx2pos(i) - center
    Fock[Ind.pos2idx(pos)] = Fock0[Ind0.pos2idx(pos)]
    H1[Ind.pos2idx(pos)] = H10[Ind0.pos2idx(pos)]

for i in range(np.product(size)):
    assert(np.allclose(Fock[i] - Fock[Ind.pos2idx(-Ind.idx2pos(i))].T, 0))

ImpSize = np.asarray([1,1])
Cell = np.eye(2) # specify geometry
Atoms = [(np.asarray([0,0]), "X")]
AtomicBasis = {"X": map(lambda x: "%d" % x, range(nbasis))}
Lat0 = dmet.buildLattice(size0, ImpSize, Cell, Atoms, AtomicBasis)
Lat = dmet.buildLattice(size, ImpSize, Cell, Atoms, AtomicBasis)
Ham0 = HamNonInt(Lat0, H1 = Fock0, H2 = np.empty((nbasis,nbasis,nbasis,nbasis)), \
        Fock = Fock0, ImpJK = None)
Ham = HamNonInt(Lat, H1 = Fock, H2 = np.empty((nbasis,nbasis,nbasis,nbasis)), \
        Fock = Fock, ImpJK = None)
Lat0.setHam(Ham0)
Lat.setHam(Ham)
ew0, ev0 = DiagRHF(Lat0.getFock(), None)
ew, ev = DiagRHF(Lat.getFock(), None)

nelec0 = nelec_per_cell * Lat0.ncells / 2
nelec = nelec_per_cell * Lat.ncells / 2

occs0, fermi0, nerr0 = assignocc(ew0, nelec0, np.inf, 0)
occs, fermi, nerr = assignocc(ew, nelec, np.inf, 0)

plotdos(ew0, fermi0, label = "original")
plotdos(ew, fermi, label = "expand")
plt.legend()
plt.savefig("DOS.png")

np.save("Fock.npy", Fock)
np.save("H1.npy", H1)
