from HubbardBCS import *
from abinitio import buildUnitCell, buildLattice, read_integral, \
        buildHamiltonian, AFInitGuessIdx, AFInitGuessOrbs
from abinitio import reportOccupation as report

def reportOccupation(lattice, GRho, names = None):
    rhoA, rhoB, kappaBA = extractRdm(GRho)
    report(lattice, np.asarray([rhoA, rhoB]), names)
