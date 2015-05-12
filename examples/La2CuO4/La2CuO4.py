import libdmet.utils.logger as log
import libdmet.dmet.abinitio as dmet
from libdmet.solver import block
import numpy as np
import numpy.linalg as la
import itertools as it
from libdmet.system import integral

block.Block.set_nproc(16)
log.verbose = "INFO"

# control variables
MaxIter = 1
DiisStart = 6
TraceStart = 4
DiisDim = 8

# for solvers
thrNatOrb = 5e-3
M = 600

# first define lattice shape, atoms and basis
Dim = 2
Cell = np.asarray([[7.6011851839 , 0], [0.11342497103, 7.60033887244]])
Atoms = [
    (np.asarray([ 0.00000,  0.00000]), "Cu"),
    (np.asarray([ 0.50000,  0.00000]), "Cu"),
    (np.asarray([ 0.00000,  0.50000]), "Cu"),
    (np.asarray([ 0.50000,  0.50000]), "Cu"),
    (np.asarray([ 0.25000-0.01680/4,  0.00000-0.01680/4]), "O" ),
    (np.asarray([ 0.00000-0.01680/4,  0.25000-0.01680/4]), "O" ),
    (np.asarray([ 0.75000+0.01680/4,  0.00000+0.01680/4]), "O" ),
    (np.asarray([ 0.50000+0.01680/4,  0.25000+0.01680/4]), "O" ),
    (np.asarray([ 0.25000+0.01680/4,  0.50000+0.01680/4]), "O" ),
    (np.asarray([ 0.00000+0.01680/4,  0.75000+0.01680/4]), "O" ),
    (np.asarray([ 0.75000-0.01680/4,  0.50000-0.01680/4]), "O" ),
    (np.asarray([ 0.50000-0.01680/4,  0.75000-0.01680/4]), "O" ),
]
AtomicBasis = {
    "Cu": ["4p3", "4p2", "4d1", "4p1", "4s", "3d5", "3d4", "3d3", "3d2", "3d1"],
    "O": ["3p3", "3p2", "3p1", "2p3", "2p2", "2p1"],
}

# for the form of vcor
AForbs = [["Cu1_3d5", "Cu4_3d5"], ["Cu2_3d5", "Cu3_3d5"]]
PMorbs = map(lambda s: "Cu%1d_3d%1d" % s, it.product(range(1,5), range(1,5)))
doping = 0
Filling = 10.5/22 - 0.5 * doping

# then lattice and impurity sizes
LatSize = np.asarray([8, 8])
ImpSize = np.asarray([1, 1])

Lat = dmet.buildLattice(LatSize, ImpSize, Cell, Atoms, AtomicBasis)
Ham = dmet.buildHamiltonian("integrals", Lat, 2)
Lat.setHam(Ham)

vcor = dmet.VcorLocal(False, False, Lat.supercell.nsites)
dmet.AFInitGuessOrbs(vcor, Lat, AForbs, PMorbs, shift = 0.4, polar = 0.4)
Mu = 0. # I don't know what is Mu

dc = dmet.FDiisContext(DiisDim)

conv = False

history = dmet.IterHistory()

log.section("\nsolving mean-field problem\n")
log.result("Vcor =\n%s", vcor.get())
log.result("Mu (guess) = %20.12f", Mu)
rho, Mu = dmet.HartreeFock(Lat, vcor, Filling, Mu)
dmet.reportOccupation(Lat, rho)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)
    log.section("\nconstructing impurity problem\n")
    log.result("Making embedding basis")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching = True)
    log.section("\nsolving impurity problem\n")
    rhoEmb, EnergyEmb = dmet.SolveImpCAS(ImpHam, M, Lat, basis, rho, thrRdm = thrNatOrb)
    with open("rdmCAS.npy", "w") as f:
        np.save(f, rhoEmb)
